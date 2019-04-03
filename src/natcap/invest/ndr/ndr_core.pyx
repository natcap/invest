# cython: profile=False
import tempfile
import logging
import os
import collections

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal
from cython.operator cimport dereference as deref

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp.list cimport list as clist
from libcpp.stack cimport stack
from libcpp.map cimport map
from libc.math cimport atan
from libc.math cimport atan2
from libc.math cimport tan
from libc.math cimport sqrt
from libc.math cimport ceil
from libc.math cimport exp

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger('ndr core')

cdef double PI = 3.141592653589793238462643383279502884
# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8
# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

cdef int is_close(double x, double y):
    return abs(x-y) <= (1e-8+1e-05*abs(y))

# this is a least recently used cache written in C++ in an external file,
# exposing here so _ManagedRaster can use it
cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)

# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.  Copied from src/pygeoprocessing/routing/routing.pyx,
# revision 891288683889237cfd3a3d0a1f09483c23489fca.
cdef class _ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
    cdef cset[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int block_xmod
    cdef int block_ymod
    cdef int block_xbits
    cdef int block_ybits
    cdef long raster_x_size
    cdef long raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef bytes raster_path
    cdef int band_id
    cdef int closed

    def __cinit__(self, raster_path, band_id, write_mode):
        """Create new instance of Managed Raster.

        Parameters:
            raster_path (char*): path to raster that has block sizes that are
                powers of 2. If not, an exception is raised.
            band_id (int): which band in `raster_path` to index. Uses GDAL
                notation that starts at 1.
            write_mode (boolean): if true, this raster is writable and dirty
                memory blocks will be written back to the raster as blocks
                are swapped out of the cache or when the object deconstructs.

        Returns:
            None.
        """
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_xmod = self.block_xsize-1
        self.block_ymod = self.block_ysize-1

        if not (1 <= band_id <= raster_info['n_bands']):
            err_msg = (
                "Error: band ID (%s) is not a valid band number. "
                "This exception is happening in Cython, so it will cause a "
                "hard seg-fault, but it's otherwise meant to be a "
                "ValueError." % (band_id))
            print(err_msg)
            raise ValueError(err_msg)
        self.band_id = band_id

        if (self.block_xsize & (self.block_xsize - 1) != 0) or (
                self.block_ysize & (self.block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s. This exception is happening"
                "in Cython, so it will cause a hard seg-fault, but it's"
                "otherwise meant to be a ValueError." % (
                    self.block_xsize, self.block_ysize, raster_path))
            print(err_msg)
            raise ValueError(err_msg)

        self.block_xbits = numpy.log2(self.block_xsize)
        self.block_ybits = numpy.log2(self.block_ysize)
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) / self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) / self.block_ysize

        self.lru_cache = new LRUCache[int, double*](MANAGED_RASTER_N_BLOCKS)
        self.raster_path = <bytes> raster_path
        self.write_mode = write_mode
        self.closed = 0

    def __dealloc__(self):
        """Deallocate _ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
        """Close the _ManagedRaster and free up resources.

            This call writes any dirty blocks to disk, frees up the memory
            allocated as part of the cache, and frees all GDAL references.

            Any subsequent calls to any other functions in _ManagedRaster will
            have undefined behavior.
        """
        if self.closed:
            return
        self.closed = 1
        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array = numpy.empty(
            (self.block_ysize, self.block_xsize))
        cdef double *double_buffer
        cdef int block_xi
        cdef int block_yi
        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize
        cdef int win_ysize

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff
        cdef int yoff

        cdef clist[BlockBufferPair].iterator it = self.lru_cache.begin()
        cdef clist[BlockBufferPair].iterator end = self.lru_cache.end()
        if not self.write_mode:
            while it != end:
                # write the changed value back if desired
                PyMem_Free(deref(it).second)
                inc(it)
            return

        raster = gdal.OpenEx(
            self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)

        # if we get here, we're in write_mode
        cdef cset[int].iterator dirty_itr
        while it != end:
            double_buffer = deref(it).second
            block_index = deref(it).first

            # write to disk if block is dirty
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr != self.dirty_blocks.end():
                self.dirty_blocks.erase(dirty_itr)
                block_xi = block_index % self.block_nx
                block_yi = block_index / self.block_nx

                # we need the offsets to subtract from global indexes for
                # cached array
                xoff = block_xi << self.block_xbits
                yoff = block_yi << self.block_ybits

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # clip window sizes if necessary
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (
                        xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (
                        yoff+win_ysize - self.raster_y_size)

                for xi_copy in xrange(win_xsize):
                    for yi_copy in xrange(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef inline void set(self, long xi, long yi, double value):
        """Set the pixel at `xi,yi` to `value`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod))<<self.block_xbits) +
                (xi & (self.block_xmod))] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef inline double get(self, long xi, long yi):
        """Return the value of the pixel at `xi,yi`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        return self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod))<<self.block_xbits) +
                (xi & (self.block_xmod))]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index / self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array
        cdef double *double_buffer
        cdef clist[BlockBufferPair] removed_value_list

        # determine the block aligned xoffset for read as array

        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize = self.block_xsize
        cdef int win_ysize = self.block_ysize

        # load a new block
        if xoff+win_xsize > self.raster_x_size:
            win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
        if yoff+win_ysize > self.raster_y_size:
            win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

        raster = gdal.OpenEx(self.raster_path, gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(
            numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            (sizeof(double) << self.block_xbits) * win_ysize)
        for xi_copy in xrange(win_xsize):
            for yi_copy in xrange(win_ysize):
                double_buffer[(yi_copy<<self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            raster = gdal.OpenEx(
                self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index / self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in xrange(win_xsize):
                        for yi_copy in xrange(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None


def ndr_eff_calculation(
        mfd_flow_direction_path, stream_path, retention_eff_lulc_path,
        crit_len_path, effective_retention_path):
    """Calculate flow downhill effective_retention to the channel.

        Parameters:
            mfd_flow_direction_path (string) - (input) a path to a raster with
                pygeoprocessing.routing MFD flow direction values.
            stream_path (string) - (input) a raster where 1 indicates a stream
                all other values ignored must be same dimensions and projection
                as mfd_flow_direction_path.
            retention_eff_lulc_path (string) - (input) a raster indicating the
                maximum retention efficiency that the landcover on that pixel
                can accumulate.
            crit_len_path (string) - (input) a raster indicating the critical length
                of the retention efficiency that the landcover on this pixel.

            effective_retention_path (string) - (output) a raster showing
                the effective retention on that pixel to the stream.

        Returns:
            nothing"""

    cdef float effective_retention_nodata = -1.0
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, effective_retention_path, gdal.GDT_Float32,
        [effective_retention_nodata])
    fp, to_process_flow_directions_path = tempfile.mkstemp(
        suffix='.tif', prefix='flow_to_process',
        dir=os.path.dirname(effective_retention_path))
    os.close(fp)

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int n_cols, n_rows
    flow_dir_info = pygeoprocessing.get_raster_info(mfd_flow_direction_path)
    n_rows, n_cols = flow_dir_info['raster_size']

    cdef stack[int] processing_stack
    stream_info = pygeoprocessing.get_raster_info(stream_path)
    # cell sizes must be square, so no reason to test at this point.
    cdef float cell_size = abs(stream_info['pixel_size'][0])

    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, False)
    cdef _ManagedRaster crit_len_raster = _ManagedRaster(
        crit_len_path, 1, False)
    cdef float crit_len_nodata = pygeoprocessing.get_raster_info(
        crit_len_path)['nodata'][0]
    cdef _ManagedRaster retention_eff_lulc_raster = _ManagedRaster(
        retention_eff_lulc_path, 1, False)
    cdef float retention_eff_nodata = pygeoprocessing.get_raster_info(
        retention_eff_lulc_path)['nodata'][0]
    cdef _ManagedRaster effective_retention_raster = _ManagedRaster(
        effective_retention_path, 1, True)
    cdef _ManagedRaster mfd_flow_direction_raster = _ManagedRaster(
        mfd_flow_direction_path, 1, False)

    # create direction raster in bytes
    def _mfd_to_flow_dir_op(mfd_array):
        result = numpy.zeros(mfd_array.shape, dtype=numpy.int8)
        for i in range(8):
            result[:] |= (((mfd_array >> (i*4)) & 0xF) > 0) << i
        return result

    pygeoprocessing.raster_calculator(
        [(mfd_flow_direction_path, 1)], _mfd_to_flow_dir_op,
        to_process_flow_directions_path, gdal.GDT_Byte, None)

    cdef _ManagedRaster to_process_flow_directions_raster = _ManagedRaster(
        to_process_flow_directions_path, 1, True)

    cdef int col_index, row_index, win_xsize, win_ysize, xoff, yoff
    cdef int global_col, global_row
    cdef int flat_index, outflow_weight, outflow_weight_sum, flow_dir
    cdef int ds_col, ds_row, i
    cdef float current_step_factor, step_size, crit_len
    cdef int neighbor_row, neighbor_col, neighbor_outflow_dir
    cdef int neighbor_outflow_dir_mask, neighbor_process_flow_dir
    cdef int outflow_dirs, dir_mask

    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        for row_index in range(win_ysize):
            global_row = yoff + row_index
            for col_index in range(win_xsize):
                global_col = xoff + col_index
                outflow_dirs = <int>to_process_flow_directions_raster.get(
                    global_col, global_row)
                should_seed = 0
                # see if this pixel drains to nodata or the edge, if so it's
                # a drain
                for i in range(8):
                    dir_mask = 1 << i
                    if outflow_dirs & dir_mask > 0:
                        neighbor_col = col_offsets[i] + global_col
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            should_seed = 1
                            outflow_dirs &= ~dir_mask
                        neighbor_row = row_offsets[i] + global_row
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            should_seed = 1
                            outflow_dirs &= ~dir_mask
                        neighbor_flow_dirs = (
                            to_process_flow_directions_raster.get(
                                neighbor_col, neighbor_row))
                        if neighbor_flow_dirs == 0:
                            should_seed = 1
                            outflow_dirs &= ~dir_mask

                if should_seed:
                    # mark all outflow directions processed
                    to_process_flow_directions_raster.set(
                        global_col, global_row, outflow_dirs)
                    processing_stack.push(global_row*n_cols+global_col)

        while processing_stack.size() > 0:
            # loop invariant, we don't push a cell on the stack that
            # hasn't already been set for processing.
            flat_index = processing_stack.top()
            processing_stack.pop()
            global_row = flat_index / n_cols
            global_col = flat_index % n_cols

            crit_len = <float>crit_len_raster.get(global_col, global_row)
            retention_eff_lulc = retention_eff_lulc_raster.get(
                global_col, global_row)
            flow_dir = <int>mfd_flow_direction_raster.get(
                    global_col, global_row)
            if stream_raster.get(global_col, global_row) == 1 or (
                    is_close(crit_len_nodata, crit_len) or
                    is_close(retention_eff_lulc, retention_eff_nodata) or
                    flow_dir == 0):
                effective_retention_raster.set(global_col, global_row, 0)
            else:
                working_retention_eff = 0.0
                outflow_weight_sum = 0
                LOGGER.info('start loop')
                for i in range(8):
                    outflow_weight = (flow_dir >> (i*4)) & 0xF
                    if outflow_weight == 0:
                        continue
                    outflow_weight_sum += outflow_weight
                    ds_col = col_offsets[i] + global_col
                    if ds_col < 0 or ds_col >= n_cols:
                        LOGGER.info('skip because ds_col < 0 or >= n_cols %s %s', ds_col, n_cols)
                        continue
                    ds_row = row_offsets[i] + global_row
                    if ds_row < 0 or ds_row >= n_rows:
                        LOGGER.info('skip because ds_row < 0 or >= n_rows %s %s', ds_row, n_rows)
                        continue
                    if i % 2 == 1:
                        step_size = cell_size*1.41421356237
                    else:
                        step_size = cell_size
                    # guard against an area that has flow but no landcover
                    current_step_factor = exp(-5*step_size/crit_len)

                    neighbor_effective_retention = (
                        effective_retention_raster.get(ds_col, ds_row))
                    if neighbor_effective_retention >= retention_eff_lulc:
                        working_retention_eff += (
                            neighbor_effective_retention) * outflow_weight
                    else:
                        intermediate_retention = (
                            (neighbor_effective_retention *
                             current_step_factor) +
                            retention_eff_lulc * (1 - current_step_factor))
                        if intermediate_retention > retention_eff_lulc:
                            intermediate_retention = retention_eff_lulc
                        working_retention_eff += (
                            intermediate_retention * outflow_weight)
                    LOGGER.debug('working_retention_eff %s %s', working_retention_eff, i)
                LOGGER.info('end loop %s %s %s', outflow_weight_sum, global_col, global_row)
                if outflow_weight_sum > 0:
                    old_eff = working_retention_eff
                    working_retention_eff /= float(outflow_weight_sum)
                    if working_retention_eff == 0:
                        LOGGER.error('setting %s %s %s %s %s %s %s %s', retention_eff_lulc, intermediate_retention, outflow_weight, outflow_weight_sum, old_eff, retention_eff_lulc, neighbor_effective_retention, current_step_factor)
                    effective_retention_raster.set(
                        global_col, global_row, working_retention_eff)
                else:
                    LOGGER.error('outflow_weight_sum %s', outflow_weight_sum)
                    raise Exception("got to a cell that has no outflow!")
            # search upstream to see if we need to push a cell on the stack
            for i in range(8):
                neighbor_col = col_offsets[i] + global_col
                if neighbor_col < 0 or neighbor_col >= n_cols:
                        continue
                neighbor_row = row_offsets[i] + global_row
                if neighbor_row < 0 or neighbor_row >= n_rows:
                    continue
                neighbor_outflow_dir = inflow_offsets[i]
                neighbor_outflow_dir_mask = 1 << neighbor_outflow_dir
                neighbor_process_flow_dir = <int>(
                    to_process_flow_directions_raster.get(
                        neighbor_col, neighbor_row))
                if neighbor_process_flow_dir == 0:
                    # skip, due to loop invariant this must be a nodata pixel
                    continue
                if neighbor_process_flow_dir & neighbor_outflow_dir_mask == 0:
                    # no outflow
                    continue
                # mask out the outflow dir that this iteration processed
                neighbor_process_flow_dir &= ~neighbor_outflow_dir_mask
                to_process_flow_directions_raster.set(
                    neighbor_col, neighbor_row, neighbor_process_flow_dir)
                if neighbor_process_flow_dir == 0:
                    # if 0 then all downstream have been processed,
                    # push on stack, otherwise another downstream pixel will
                    # pick it up
                    processing_stack.push(neighbor_row*n_cols + neighbor_col)

    to_process_flow_directions_raster.close()
    #os.remove(to_process_flow_directions_path)
    return
    ####################
    """
    outflow_weights_path = natcap.invest.pygeoprocessing_0_3_3.temporary_filename()
    outflow_direction_path = natcap.invest.pygeoprocessing_0_3_3.temporary_filename()
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.calculate_flow_weights(
        flow_direction_path, outflow_weights_path, outflow_direction_path)
    outflow_weights_ds = gdal.OpenEx(outflow_weights_path)
    outflow_weights_band = outflow_weights_ds.GetRasterBand(1)
    cdef float outflow_weights_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_weights_path)
    outflow_direction_ds = gdal.OpenEx(outflow_direction_path)
    outflow_direction_band = outflow_direction_ds.GetRasterBand(1)
    cdef int outflow_direction_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_direction_path)
    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = stream_band.GetBlockSize()
    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] stream_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int8))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] effective_retention_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] retention_eff_lulc_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] crit_len_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)


    cdef numpy.ndarray[numpy.npy_int8, ndim=4] processed_cell_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int8))

    band_list = [stream_band, outflow_direction_band, outflow_weights_band,
                 effective_retention_band, processed_cell_band,
                 retention_eff_lulc_band, crit_len_band]
    block_list = [stream_block, outflow_direction_block, outflow_weights_block,
                  effective_retention_block, processed_cell_block,
                  retention_eff_lulc_block, crit_len_block]
    update_list = [False, False, False, True, True, False, False]

    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = (
        numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte))

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    #center point of global index
    cdef int global_row, global_col
    cdef int row_index, col_index
    cdef int row_block_offset, col_block_offset
    cdef int global_block_row, global_block_col

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col
    cdef int neighbor_row_index, neighbor_col_index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset
    cdef int flat_index


    cdef c_set[int] cells_in_queue

    #build up the stream pixel indexes as starting seed points for the search
    cdef time_t last_time, current_time
    time(&last_time)
    for global_block_row in xrange(n_global_block_rows):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                "find_sinks %.1f%% complete",
                (global_block_row + 1.0) / n_global_block_rows * 100)
            last_time = current_time
        for global_block_col in xrange(n_global_block_cols):
            for global_row in xrange(
                    global_block_row*block_row_size,
                    min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(
                        global_block_col*block_col_size,
                        min((global_block_col+1)*block_col_size, n_cols)):
                    block_cache.update_cache(
                        global_row, global_col, &row_index, &col_index,
                        &row_block_offset, &col_block_offset)
                    if stream_block[
                            row_index, col_index, row_block_offset,
                            col_block_offset] == 1:
                        flat_index = global_row * n_cols + global_col
                        visit_stack.push_front(global_row * n_cols + global_col)
                        cells_in_queue.insert(flat_index)

                        effective_retention_block[row_index, col_index,
                            row_block_offset, col_block_offset] = 0
                        processed_cell_block[row_index, col_index,
                            row_block_offset, col_block_offset] = 1
                        cache_dirty[row_index, col_index] = 1

    cdef int neighbor_outflow_direction, neighbor_index, outflow_direction
    cdef float neighbor_outflow_weight, current_effective_retention
    cdef float outflow_weight, neighbor_effective_retention, step_size
    cdef float downstream_effective_retention, current_stream
    cdef float original_effective_retention
    cdef float retention_eff_lulc, crit_len, intermediate_retention
    cdef float current_step_factor
    cdef int it_flows_here

    while visit_stack.size() > 0:
        flat_index = visit_stack.front()
        visit_stack.pop_front()
        cells_in_queue.erase(flat_index)
        global_row = flat_index / n_cols
        global_col = flat_index % n_cols

        block_cache.update_cache(
            global_row, global_col, &row_index, &col_index,
            &row_block_offset, &col_block_offset)

        update_downstream = False
        current_effective_retention = 0.0

        time(&current_time)
        if current_time - last_time > 0.0001:
            last_time = current_time
            LOGGER.info(
                'visit_stack on stream effective_retention size: %d ', visit_stack.size())

        current_stream = stream_block[
            row_index, col_index, row_block_offset, col_block_offset]
        outflow_direction = outflow_direction_block[
            row_index, col_index, row_block_offset,
            col_block_offset]
        if current_stream == 1:
            effective_retention_block[row_index, col_index,
                row_block_offset, col_block_offset] = 0
            processed_cell_block[row_index, col_index,
                row_block_offset, col_block_offset] = 1
            cache_dirty[row_index, col_index] = 1
        elif outflow_direction == outflow_direction_nodata:
            current_effective_retention = 1.0
        elif processed_cell_block[row_index, col_index, row_block_offset,
                col_block_offset] == 0:
            #add downstream effective_retention to current effective_retention

            outflow_weight = outflow_weights_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            retention_eff_lulc = retention_eff_lulc_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            crit_len = crit_len_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            for neighbor_index in xrange(2):
                #check if downstream neighbors are calcualted
                if neighbor_index == 1:
                    outflow_direction = (outflow_direction + 1) % 8
                    outflow_weight = (1.0 - outflow_weight)

                if outflow_weight <= 0.0:
                    continue

                neighbor_row = global_row + row_offsets[outflow_direction]
                neighbor_col = global_col + col_offsets[outflow_direction]
                if (neighbor_row < 0 or neighbor_row >= n_rows or
                        neighbor_col < 0 or neighbor_col >= n_cols):
                    #out of bounds
                    continue

                block_cache.update_cache(
                    neighbor_row, neighbor_col, &neighbor_row_index,
                    &neighbor_col_index, &neighbor_row_block_offset,
                    &neighbor_col_block_offset)

                if stream_block[neighbor_row_index,
                        neighbor_col_index, neighbor_row_block_offset,
                        neighbor_col_block_offset] == stream_nodata:
                    #out of the valid raster entirely
                    continue

                neighbor_effective_retention = effective_retention_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                neighbor_outflow_direction = outflow_direction_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                neighbor_outflow_weight = outflow_weights_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                if processed_cell_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == 0:
                    neighbor_flat_index = neighbor_row * n_cols + neighbor_col
                    #insert into the processing queue if it's not already there
                    if (cells_in_queue.find(flat_index) ==
                            cells_in_queue.end()):
                        visit_stack.push_back(flat_index)
                        cells_in_queue.insert(flat_index)

                    if (cells_in_queue.find(neighbor_flat_index) ==
                            cells_in_queue.end()):
                        visit_stack.push_front(neighbor_flat_index)
                        cells_in_queue.insert(neighbor_flat_index)

                    update_downstream = True
                    neighbor_effective_retention = 0.0

                if outflow_direction % 2 == 1:
                    #increase effective_retention by a square root of 2 for diagonal
                    step_size = cell_size * 1.41421356237
                else:
                    step_size = cell_size

                current_step_factor = exp(-5 * step_size / crit_len)
                if neighbor_effective_retention >= retention_eff_lulc:
                    current_effective_retention += (
                        neighbor_effective_retention) * outflow_weight
                else:
                    intermediate_retention =  (
                        neighbor_effective_retention * current_step_factor +
                        retention_eff_lulc * (1 - current_step_factor))
                    if intermediate_retention > retention_eff_lulc:
                        intermediate_retention = retention_eff_lulc
                    current_effective_retention += (
                        intermediate_retention * outflow_weight)

        if not update_downstream:
            #mark flat_index as processed
            block_cache.update_cache(
                global_row, global_col, &row_index, &col_index,
                &row_block_offset, &col_block_offset)
            processed_cell_block[row_index, col_index,
                row_block_offset, col_block_offset] = 1
            effective_retention_block[row_index, col_index,
                row_block_offset, col_block_offset] = (
                    current_effective_retention)
            cache_dirty[row_index, col_index] = 1

            #update any upstream neighbors with this effective_retention
            for neighbor_index in range(8):
                neighbor_row = global_row + row_offsets[neighbor_index]
                neighbor_col = global_col + col_offsets[neighbor_index]
                if (neighbor_row < 0 or neighbor_row >= n_rows or
                        neighbor_col < 0 or neighbor_col >= n_cols):
                    #out of bounds
                    continue

                block_cache.update_cache(
                    neighbor_row, neighbor_col, &neighbor_row_index,
                    &neighbor_col_index, &neighbor_row_block_offset,
                    &neighbor_col_block_offset)

                #streams were already added, skip if they are in the queue
                if (stream_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == 1 or
                    stream_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == stream_nodata):
                    continue

                if processed_cell_block[
                        neighbor_row_index,
                        neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == 1:
                    #don't reprocess it, it's already been updated by two valid
                    #children
                    continue

                neighbor_outflow_direction = outflow_direction_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]
                if neighbor_outflow_direction == outflow_direction_nodata:
                    #if the neighbor has no flow, we can't flow here
                    continue

                neighbor_outflow_weight = outflow_weights_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                it_flows_here = False
                if (neighbor_outflow_direction ==
                        inflow_offsets[neighbor_index]):
                    it_flows_here = True
                elif ((neighbor_outflow_direction + 1) % 8 ==
                        inflow_offsets[neighbor_index]):
                    it_flows_here = True
                    neighbor_outflow_weight = 1.0 - neighbor_outflow_weight

                neighbor_flat_index = neighbor_row * n_cols + neighbor_col
                if (it_flows_here and neighbor_outflow_weight > 0.0 and
                    cells_in_queue.find(neighbor_flat_index) ==
                        cells_in_queue.end()):
                    visit_stack.push_back(neighbor_flat_index)
                    cells_in_queue.insert(neighbor_flat_index)

    block_cache.flush_cache()

    for dataset in [outflow_weights_ds, outflow_direction_ds]:
        gdal.Dataset.__swig_destroy__(dataset)
    for dataset_uri in [outflow_weights_path, outflow_direction_path]:
        os.remove(dataset_uri)
    """
