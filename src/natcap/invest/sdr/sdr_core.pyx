# cython: profile=False
# cython: language_level=3
import logging
import os

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp.list cimport list as clist
from libcpp.stack cimport stack
cimport libc.math as cmath

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef double PI = 3.141592653589793238462643383279502884
# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8
# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# These offsets are for the neighbor rows and columns according to the
# ordering: 3 2 1
#           4 x 0
#           5 6 7
cdef int *ROW_OFFSETS = [0, -1, -1, -1,  0,  1, 1, 1]
cdef int *COL_OFFSETS = [1,  1,  0, -1, -1, -1, 0, 1]


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
            self.raster_x_size + (self.block_xsize) - 1) // self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) // self.block_ysize

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
        cdef int block_yi = block_index // self.block_nx

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
                    block_yi = block_index // self.block_nx

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


def calculate_sediment_deposition(
        mfd_flow_direction_path, e_prime_path, f_path, sdr_path,
        target_sediment_deposition_path):
    """Calculate sediment deposition layer

        Parameters:
            mfd_flow_direction_path (string): a path to a raster with
                pygeoprocessing.routing MFD flow direction values.
            e_prime_path (string): path to a raster that shows sources of
                sediment that wash off a pixel but do not reach the stream.
            f_path (string): path to a raster that shows the sediment flux
                on a pixel for sediment that does not reach the stream.
            sdr_path (string): path to Sediment Delivery Ratio raster.
            target_sediment_deposition_path (string): path to created that
                shows where the E' sources end up across the landscape.

        Returns:
            None.

    """
    LOGGER.info('calculate sediment deposition')
    cdef float sediment_deposition_nodata = -1.0
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, target_sediment_deposition_path,
        gdal.GDT_Float32, [sediment_deposition_nodata])
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, f_path,
        gdal.GDT_Float32, [sediment_deposition_nodata])

    cdef _ManagedRaster mfd_flow_direction_raster = _ManagedRaster(
        mfd_flow_direction_path, 1, False)
    cdef _ManagedRaster e_prime_raster = _ManagedRaster(
        e_prime_path, 1, False)
    cdef _ManagedRaster sdr_raster = _ManagedRaster(sdr_path, 1, False)
    cdef _ManagedRaster f_raster = _ManagedRaster(f_path, 1, True)
    cdef _ManagedRaster sediment_deposition_raster = _ManagedRaster(
        target_sediment_deposition_path, 1, True)

    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int n_cols, n_rows
    flow_dir_info = pygeoprocessing.get_raster_info(mfd_flow_direction_path)
    n_cols, n_rows = flow_dir_info['raster_size']
    cdef stack[int] processing_stack
    cdef float sdr_nodata = pygeoprocessing.get_raster_info(
        sdr_path)['nodata'][0]
    cdef float e_prime_nodata = pygeoprocessing.get_raster_info(
        e_prime_path)['nodata'][0]
    cdef int col_index, row_index, win_xsize, win_ysize, xoff, yoff
    cdef int global_col, global_row, flat_index, j, k
    cdef int seed_col = 0
    cdef int seed_row = 0
    cdef int neighbor_row, neighbor_col
    cdef int flow_val, neighbor_flow_val, ds_neighbor_flow_val
    cdef int flow_weight, neighbor_flow_weight
    cdef float flow_sum, neighbor_flow_sum
    cdef float downstream_sdr_weighted_sum, sdr_i, sdr_j
    cdef float r_j, r_j_weighted_sum, p_j, p_val

    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('%.2f%% complete', 100.0 * (
            (seed_row * seed_col) / float(n_cols*n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index
                # search to see if this is a good seed
                if mfd_flow_direction_raster.get(seed_col, seed_row) == 0:
                    continue
                seed_pixel = 1
                for j in range(8):
                    neighbor_row = seed_row + ROW_OFFSETS[j]
                    if neighbor_row < 0 or neighbor_row >= n_rows:
                        continue
                    neighbor_col = seed_col + COL_OFFSETS[j]
                    if neighbor_col < 0 or neighbor_col >= n_cols:
                        continue
                    neighbor_flow_val = <int>mfd_flow_direction_raster.get(
                        neighbor_col, neighbor_row)
                    if neighbor_flow_val == 0:
                        continue
                    neighbor_flow_weight = (
                        neighbor_flow_val >> (inflow_offsets[j]*4)) & 0xF
                    if neighbor_flow_weight > 0:
                        # neighbor flows in, not a seed
                        seed_pixel = 0
                        break
                if seed_pixel and (
                        sediment_deposition_raster.get(
                            seed_col, seed_row) ==
                        sediment_deposition_nodata):
                    processing_stack.push(seed_row * n_cols + seed_col)

                while processing_stack.size() > 0:
                    # loop invariant: cell has all upstream neighbors
                    # processed
                    flat_index = processing_stack.top()
                    processing_stack.pop()
                    global_row = flat_index // n_cols
                    global_col = flat_index % n_cols

                    # calculate the upstream Fj contribution to this pixel
                    f_j_weighted_sum = 0
                    for j in range(8):
                        neighbor_row = global_row + ROW_OFFSETS[j]
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            continue
                        neighbor_col = global_col + COL_OFFSETS[j]
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            continue

                        # see if there's an inflow
                        neighbor_flow_val = (
                            <int>mfd_flow_direction_raster.get(
                                neighbor_col, neighbor_row))
                        neighbor_flow_weight = (
                            neighbor_flow_val >> (inflow_offsets[j]*4)) & 0xF
                        if neighbor_flow_weight > 0:
                            f_j = f_raster.get(neighbor_col, neighbor_row)
                            neighbor_flow_sum = 0
                            for k in range(8):
                                neighbor_flow_sum += (
                                    neighbor_flow_val >> (k*4)) & 0xF
                            p_val = neighbor_flow_weight / neighbor_flow_sum
                            f_j_weighted_sum += p_val * f_j

                    # calculate the differential downstream change in sdr
                    # from this pixel
                    downstream_sdr_weighted_sum = 0.0
                    flow_val = <int>mfd_flow_direction_raster.get(
                        global_col, global_row)
                    flow_sum = 0.0
                    for k in range(8):
                        flow_sum += (flow_val >> (k*4)) & 0xF

                    for j in range(8):
                        neighbor_row = global_row + ROW_OFFSETS[j]
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            continue
                        neighbor_col = global_col + COL_OFFSETS[j]
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            continue
                        # if this direction flows out, add to weighted sum
                        flow_weight = (flow_val >> (j*4)) & 0xF
                        if flow_weight > 0:
                            sdr_j = sdr_raster.get(neighbor_col, neighbor_row)
                            if sdr_j == 0.0:
                                # this means it's a stream, for SDR deposition
                                # purposes, we set sdr to 1 to indicate this
                                # is the last step on which to retain sediment
                                sdr_j = 1.0
                            if sdr_j == sdr_nodata:
                                sdr_j = 0.0
                            p_j = flow_weight / flow_sum
                            downstream_sdr_weighted_sum += sdr_j * p_j

                            # if there is a downstream neighbor it
                            # couldn't have been pushed on the processing
                            # stack yet, because the upstream was just
                            # completed
                            upstream_neighbors_processed = 1
                            for k in range(8):
                                if inflow_offsets[k] == j:
                                    # we don't need to process the one
                                    # we're currently calculating
                                    continue
                                # see if there's an inflow
                                ds_neighbor_row = (
                                    neighbor_row + ROW_OFFSETS[k])
                                if ds_neighbor_row < 0 or ds_neighbor_row >= n_rows:
                                    continue
                                ds_neighbor_col = (
                                    neighbor_col + COL_OFFSETS[k])
                                if ds_neighbor_col < 0 or ds_neighbor_col >= n_cols:
                                    continue
                                ds_neighbor_flow_val = (
                                    <int>mfd_flow_direction_raster.get(
                                        ds_neighbor_col, ds_neighbor_row))
                                if (ds_neighbor_flow_val >> (
                                        inflow_offsets[k]*4)) & 0xF > 0:
                                    if sediment_deposition_raster.get(
                                            ds_neighbor_col,
                                            ds_neighbor_row) == (
                                                sediment_deposition_nodata):
                                        # can't push it because not
                                        # processed yet
                                        upstream_neighbors_processed = 0
                                        break
                            if upstream_neighbors_processed:
                                processing_stack.push(
                                    neighbor_row * n_cols +
                                    neighbor_col)

                    sdr_i = sdr_raster.get(global_col, global_row)
                    if sdr_i == sdr_nodata:
                        sdr_i = 0.0
                    e_prime_i = e_prime_raster.get(global_col, global_row)
                    if e_prime_i == e_prime_nodata:
                        e_prime_i = 0.0

                    if downstream_sdr_weighted_sum < sdr_i:
                        # i think this happens because of our low resolution
                        # flow direction, it's okay to zero out.
                        downstream_sdr_weighted_sum = sdr_i
                    d_ri = (downstream_sdr_weighted_sum - sdr_i) / (1 - sdr_i)
                    r_i = d_ri * (e_prime_i + f_j_weighted_sum)
                    f_i = (1-d_ri) * (e_prime_i + f_j_weighted_sum)
                    sediment_deposition_raster.set(
                        global_col, global_row, r_i)
                    f_raster.set(global_col, global_row, f_i)

    LOGGER.info('100% complete')
    sediment_deposition_raster.close()


def calculate_average_aspect(
    mfd_flow_direction_path, target_average_aspect_path):
    """Calculate the Average Proportional Flow from MFD.

    Calculates the weighted average of proportional flow for every MFD flow
    direction pixel provided.

    Parameters:
        mfd_flow_direction_path (string): The path to an MFD flow direction
            raster.
        target_average_aspect_path (string): The path to where the calculated
            weighted average aspect raster should be written.

    Returns:
        ``None``.

    """
    LOGGER.info('Calculating average aspect')

    cdef float average_aspect_nodata = -1.0
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, target_average_aspect_path,
        gdal.GDT_Float32, [average_aspect_nodata], [average_aspect_nodata])

    flow_direction_info = pygeoprocessing.get_raster_info(
        mfd_flow_direction_path)
    cdef int mfd_flow_direction_nodata = flow_direction_info['nodata'][0]
    cdef int n_cols, n_rows
    n_cols, n_rows = flow_direction_info['raster_size']

    cdef _ManagedRaster mfd_flow_direction_raster = _ManagedRaster(
        mfd_flow_direction_path, 1, False)

    cdef _ManagedRaster average_aspect_raster = _ManagedRaster(
        target_average_aspect_path, 1, True)

    cdef int* neighbor_weights = [0, 0, 0, 0, 0, 0, 0, 0]
    cdef int seed_row = 0
    cdef int seed_col = 0
    cdef int n_pixels_visited = 0
    cdef int win_xsize, win_ysize, xoff, yoff
    cdef int row_index, col_index, neighbor_index
    cdef int flow_weight_in_direction
    cdef int weight_sum
    cdef int seed_flow_value
    cdef float aspect_weighted_average
    cdef float proportional_flow
    cdef float flow_length
    cdef float* flow_lengths = [
        1.0, <float>cmath.M_SQRT2,
        1.0, <float>cmath.M_SQRT2,
        1.0, <float>cmath.M_SQRT2,
        1.0, <float>cmath.M_SQRT2
    ]

    # Loop over iterblocks to maintain cache locality
    # Find each non-nodata pixel and calculate proportional flow
    # Multiply proportional flow times the flow length x_d
    # write the final value to the raster.
    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Average aspect %.2f%% complete', 100.0 * (
            n_pixels_visited / float(n_cols * n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index
                seed_flow_value = <int>mfd_flow_direction_raster.get(
                    seed_col, seed_row)

                # Skip this seed if it's nodata (Currently expected to be 0).
                # No need to set the nodata value here since we have already
                # filled the raster with nodata values at creation time.
                if seed_flow_value == mfd_flow_direction_nodata:
                    continue

                weight_sum = 0
                for neighbor_index in range(8):
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row < 0 or neighbor_row >= n_rows:
                        neighbor_weights[neighbor_index] = 0
                        continue

                    neighbor_col = seed_col + ROW_OFFSETS[neighbor_index]
                    if neighbor_col < 0 or neighbor_col >= n_cols:
                        neighbor_weights[neighbor_index] = 0
                        continue

                    flow_weight_in_direction = (seed_flow_value >> (
                        neighbor_index * 4) & 0xF)
                    weight_sum += flow_weight_in_direction
                    neighbor_weights[neighbor_index] = flow_weight_in_direction

                # Weight sum should never be less than 0.
                # Since it's an int, we can compare it directly against the
                # value of 0.
                if weight_sum == 0:
                    aspect_weighted_average = average_aspect_nodata
                else:
                    aspect_weighted_average = 0.0
                    for neighbor_index in range(8):
                        # the flow_lengths array is the functional equivalent
                        # of calculating |sin(alpha)| + |cos(alpha)|.
                        aspect_weighted_average += (
                            flow_lengths[neighbor_index] *
                            neighbor_weights[neighbor_index])

                    # We already know that weight_sum will be > 0 because we
                    # check for it in the condition above.
                    with cython.cdivision(True):
                        aspect_weighted_average = aspect_weighted_average / <float>weight_sum

                average_aspect_raster.set(
                    seed_col, seed_row, aspect_weighted_average)

        n_pixels_visited += win_xsize * win_ysize

    LOGGER.info('Average aspect 100.00% complete')

    mfd_flow_direction_raster.close()
    average_aspect_raster.close()
