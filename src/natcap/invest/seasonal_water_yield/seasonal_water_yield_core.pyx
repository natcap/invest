# cython: profile=False

import logging
import os
import collections
import sys
import gc

import numpy
cimport numpy
cimport cython
import osgeo
from osgeo import gdal
from cython.operator cimport dereference as deref

from libcpp.set cimport set as c_set
from libcpp.deque cimport deque
from libcpp.map cimport map
from libcpp.stack cimport stack
from libc.math cimport atan
from libc.math cimport atan2
from libc.math cimport tan
from libc.math cimport sqrt
from libc.math cimport ceil

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

import natcap.invest.pygeoprocessing_0_3_3
cimport natcap.invest.pygeoprocessing_0_3_3.routing.routing_core
from natcap.invest.pygeoprocessing_0_3_3.routing.routing_core cimport BlockCache


LOGGER = logging.getLogger('natcap.invest.pygeoprocessing_0_3_3.routing.routing_core')

cdef int N_MONTHS = 12

cdef double PI = 3.141592653589793238462643383279502884
cdef double INF = numpy.inf
cdef int N_BLOCK_ROWS = 4
cdef int N_BLOCK_COLS = 4

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



@cython.boundscheck(False)
@cython.wraparound(False)
cdef route_local_recharge(
        precip_path_list, et0_path_list, kc_path_list, li_path,
        li_avail_path, l_sum_avail_path, aet_path, numpy.ndarray alpha_month,
        float beta_i, float gamma, qfi_path_list, outflow_direction_path,
        outflow_weights_path, stream_path, deque[int] &sink_cell_deque):

    #Pass transport
    cdef time_t start
    time(&start)

    #load a base raster so we can determine the n_rows/cols
    outflow_direction_raster = gdal.OpenEx(outflow_direction_path)
    cdef int n_cols = outflow_direction_raster.RasterXSize
    cdef int n_rows = outflow_direction_raster.RasterYSize
    outflow_direction_band = outflow_direction_raster.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = outflow_direction_band.GetBlockSize()

    #center point of global index
    cdef int global_row, global_col #index into the overall raster
    cdef int row_index, col_index #the index of the cache block
    cdef int row_block_offset, col_block_offset #index into the cache block
    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the single caches
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] li_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] li_avail_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] l_sum_avail_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] aet_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] stream_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)

    #these are 12 band blocks
    cdef numpy.ndarray[numpy.npy_float32, ndim=5] precip_block_list = numpy.zeros(
        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=5] et0_block_list = numpy.zeros(
        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=5] qfi_block_list = numpy.zeros(
        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=5] kc_block_list = numpy.zeros(
        (N_MONTHS, N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)

    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)

    cdef int outflow_direction_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_direction_path)

    #load the et0 and precip bands
    et0_raster_list = []
    et0_band_list = []
    precip_datset_list = []
    precip_band_list = []

    for path_list, raster_list, band_list in [
            (et0_path_list, et0_raster_list, et0_band_list),
            (precip_path_list, precip_datset_list, precip_band_list)]:
        for index, path in enumerate(path_list):
            raster_list.append(gdal.OpenEx(path))
            band_list.append(raster_list[index].GetRasterBand(1))

    cdef float precip_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(precip_path_list[0])
    cdef float et0_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(et0_path_list[0])

    qfi_datset_list = []
    qfi_band_list = []

    outflow_weights_raster = gdal.OpenEx(outflow_weights_path)
    outflow_weights_band = outflow_weights_raster.GetRasterBand(1)
    cdef float outflow_weights_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_weights_path)
    stream_raster = gdal.OpenEx(stream_path)
    stream_band = stream_raster.GetRasterBand(1)

    #Create output arrays qfi and local_recharge and local_recharge_avail
    cdef float local_recharge_nodata = -99999
    natcap.invest.pygeoprocessing_0_3_3.new_raster_from_base_uri(
        outflow_direction_path, li_path, 'GTiff', local_recharge_nodata,
        gdal.GDT_Float32)
    li_raster = gdal.OpenEx(li_path, gdal.GA_Update)
    li_band = li_raster.GetRasterBand(1)
    natcap.invest.pygeoprocessing_0_3_3.new_raster_from_base_uri(
        outflow_direction_path, li_avail_path, 'GTiff', local_recharge_nodata,
        gdal.GDT_Float32)
    li_avail_raster = gdal.OpenEx(li_avail_path, gdal.GA_Update)
    li_avail_band = li_avail_raster.GetRasterBand(1)
    natcap.invest.pygeoprocessing_0_3_3.new_raster_from_base_uri(
       outflow_direction_path, l_sum_avail_path, 'GTiff', local_recharge_nodata,
       gdal.GDT_Float32)
    l_sum_avail_raster = gdal.OpenEx(l_sum_avail_path, gdal.GA_Update)
    l_sum_avail_band = l_sum_avail_raster.GetRasterBand(1)

    cdef float aet_nodata = -99999
    natcap.invest.pygeoprocessing_0_3_3.new_raster_from_base_uri(
        outflow_direction_path, aet_path, 'GTiff', aet_nodata,
        gdal.GDT_Float32)
    aet_raster = gdal.OpenEx(aet_path, gdal.GA_Update)
    aet_band = aet_raster.GetRasterBand(1)

    qfi_raster_list = []
    qfi_band_list = []
    kc_raster_list = []
    kc_band_list = []
    cdef float qfi_nodata = natcap.invest.pygeoprocessing_0_3_3.geoprocessing.get_nodata_from_uri(
        qfi_path_list[0])
    for index, (qfi_path, kc_path) in enumerate(
            zip(qfi_path_list, kc_path_list)):
        qfi_raster_list.append(gdal.OpenEx(qfi_path, gdal.GA_ReadOnly))
        qfi_band_list.append(qfi_raster_list[index].GetRasterBand(1))
        kc_raster_list.append(gdal.OpenEx(kc_path, gdal.GA_ReadOnly))
        kc_band_list.append(kc_raster_list[index].GetRasterBand(1))

    band_list = ([
        outflow_direction_band, outflow_weights_band, stream_band] +
        precip_band_list + et0_band_list + qfi_band_list + kc_band_list +
        [li_band, li_avail_band, l_sum_avail_band, aet_band])

    block_list = [
        outflow_direction_block, outflow_weights_block, stream_block]
    block_list.extend([precip_block_list[i] for i in xrange(N_MONTHS)])
    block_list.extend([et0_block_list[i] for i in xrange(N_MONTHS)])
    block_list.extend([qfi_block_list[i] for i in xrange(N_MONTHS)])
    block_list.extend([kc_block_list[i] for i in xrange(N_MONTHS)])
    block_list.append(li_block)
    block_list.append(li_avail_block)
    block_list.append(l_sum_avail_block)
    block_list.append(aet_block)

    update_list = (
        [False] * (3 + len(precip_band_list) + len(et0_band_list) +
            len(qfi_band_list) + len(kc_band_list)) + [True, True, True, True])

    cache_dirty[:] = 0

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols,
        block_row_size, block_col_size,
        band_list, block_list, update_list, cache_dirty)

    #Process flux through the grid
    cdef stack[int] cells_to_process
    cdef stack[int] cell_neighbor_to_process
    cdef stack[float] r_sum_stack

    for cell in sink_cell_deque:
        cells_to_process.push(cell)
        cell_neighbor_to_process.push(0)
        r_sum_stack.push(0.0)

    #Diagonal offsets are based off the following index notation for neighbors
    #    3 2 1
    #    4 p 0
    #    5 6 7

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int neighbor_direction
    cdef double absorption_rate
    cdef double outflow_weight
    cdef double in_flux
    cdef int current_neighbor_index
    cdef int current_index
    cdef float current_l_sum_avail
    cdef float qf_nodata = natcap.invest.pygeoprocessing_0_3_3.geoprocessing.get_nodata_from_uri(
        qfi_path_list[0])
    cdef int month_index
    cdef float aet_sum
    cdef float pet_m
    cdef float aet_m
    cdef float p_i
    cdef float qf_i
    cdef float qfi_m
    cdef float p_m
    cdef float l_i
    cdef float li_avail_value
    cdef float l_sum_avail_value
    cdef int neighbors_calculated = 0

    cdef time_t last_time, current_time
    time(&last_time)
    while not cells_to_process.empty():
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info('route_local_recharge work queue size = %d' % (
                cells_to_process.size()))
            last_time = current_time

        current_index = cells_to_process.top()
        cells_to_process.pop()
        with cython.cdivision(True):
            global_row = current_index / n_cols
            global_col = current_index % n_cols
        #see if we need to update the row cache

        current_neighbor_index = cell_neighbor_to_process.top()
        cell_neighbor_to_process.pop()
        current_l_sum_avail = r_sum_stack.top()
        r_sum_stack.pop()
        neighbors_calculated = 1

        block_cache.update_cache(
            global_row, global_col, &row_index, &col_index, &row_block_offset,
            &col_block_offset)

        #Ensure we are working on a valid pixel, if not set everything to 0
        #check quickflow nodata? month 0? qfi_nodata
        if qfi_block_list[0, row_index, col_index, row_block_offset, col_block_offset] == qfi_nodata:
            li_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            li_avail_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            l_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            cache_dirty[row_index, col_index] = 1
            continue

        for direction_index in xrange(current_neighbor_index, 8):
            #get percent flow from neighbor to current cell
            neighbor_row = global_row + row_offsets[direction_index]
            neighbor_col = global_col + col_offsets[direction_index]

            #See if neighbor out of bounds
            if (neighbor_row < 0 or neighbor_row >= n_rows or neighbor_col < 0 or neighbor_col >= n_cols):
                continue

            block_cache.update_cache(
                neighbor_row, neighbor_col, &neighbor_row_index,
                &neighbor_col_index, &neighbor_row_block_offset,
                &neighbor_col_block_offset)
            #if neighbor inflows
            neighbor_direction = outflow_direction_block[
                neighbor_row_index, neighbor_col_index,
                neighbor_row_block_offset, neighbor_col_block_offset]
            if neighbor_direction == outflow_direction_nodata:
                continue

            #check if the cell flows directly, or is one index off
            if (inflow_offsets[direction_index] != neighbor_direction and
                    ((inflow_offsets[direction_index] - 1) % 8) != neighbor_direction):
                #then neighbor doesn't inflow into current cell
                continue

            #Calculate the outflow weight
            outflow_weight = outflow_weights_block[
                neighbor_row_index, neighbor_col_index,
                neighbor_row_block_offset, neighbor_col_block_offset]

            if ((inflow_offsets[direction_index] - 1) % 8) == neighbor_direction:
                outflow_weight = 1.0 - outflow_weight

            if outflow_weight <= 0.0:
                continue

            if l_sum_avail_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == local_recharge_nodata:
                #push current cell and and loop
                cells_to_process.push(current_index)
                cell_neighbor_to_process.push(direction_index)
                r_sum_stack.push(current_l_sum_avail)
                cells_to_process.push(neighbor_row * n_cols + neighbor_col)
                cell_neighbor_to_process.push(0)
                r_sum_stack.push(0.0)
                neighbors_calculated = 0
                break
            else:
                #'calculate l_avail_i and l_i'
                #add the contribution of the upstream to l_avail and l_i eq [7]
                # in cases of bad user data we can sometimes loop and still
                # get nodata, treat it as zero flow.
                li_avail_value = li_avail_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]
                if li_avail_value == local_recharge_nodata:
                    li_avail_value = 0.0
                l_sum_avail_value = l_sum_avail_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]
                if l_sum_avail_value == local_recharge_nodata:
                    l_sum_avail_value = 0.0
                current_l_sum_avail += (
                    li_avail_value + l_sum_avail_value) * outflow_weight

        if not neighbors_calculated:
            continue

        #if we got here current_l_sum_avail is correct
        block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)
        p_i = 0.0
        qf_i = 0.0
        aet_sum = 0.0
        for month_index in xrange(N_MONTHS):
            p_m = precip_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset]
            if abs(p_m-precip_nodata) > 1e-6:  # it's too far apart to be nodata
                p_i += p_m
            else:
                p_m = 0.0 # don't add a nodata value later
            # Eq [6]
            # This check for nodata came up when several users had ill aligned data
            if abs(et0_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset]-et0_nodata) > 1e-6:
                pet_m = (
                    kc_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset] *
                    et0_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset])
            else:
                pet_m = 0.0
            qfi_m = qfi_block_list[month_index, row_index, col_index, row_block_offset, col_block_offset]
            qf_i += qfi_m
            # Eq [5]
            aet_m = min(
                pet_m, p_m - qfi_m + alpha_month[month_index] * beta_i *
                current_l_sum_avail)
            aet_sum += aet_m
        # Eq [3]
        l_i = p_i - qf_i - aet_sum

        #if it's a stream, set all recharge to 0 and aet to nodata
        if stream_block[row_index, col_index, row_block_offset, col_block_offset] == 1:
            l_i = 0
            current_l_sum_avail = 0
            aet_sum = aet_nodata

        # Eq [8]
        li_avail_block[row_index, col_index, row_block_offset, col_block_offset] = max(gamma * l_i, l_i)

        l_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset] = current_l_sum_avail
        li_block[row_index, col_index, row_block_offset, col_block_offset] = l_i
        aet_block[row_index, col_index, row_block_offset, col_block_offset] = aet_sum
        cache_dirty[row_index, col_index] = 1

    block_cache.flush_cache()


def calculate_local_recharge(
        precip_path_list, et0_path_list, qfm_path_list, flow_dir_path,
        outflow_weights_path, outflow_direction_path, dem_path, lulc_path,
        alpha_month, beta_i, gamma, stream_path, li_path,
        li_avail_path, l_sum_avail_path, aet_path, kc_path_list):
    """wrapper for local recharge so we can statically type outlet lists"""
    cdef deque[int] outlet_cell_deque
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.find_outlets(dem_path, flow_dir_path, outlet_cell_deque)
    # convert a dictionary of month -> alpha to an array of alphas in sorted
    # order by month
    cdef numpy.ndarray alpha_month_array = numpy.array(
        [x[1] for x in sorted(alpha_month.iteritems())])
    route_local_recharge(
        precip_path_list, et0_path_list, kc_path_list, li_path,
        li_avail_path, l_sum_avail_path, aet_path, alpha_month_array, beta_i,
        gamma, qfm_path_list, outflow_direction_path, outflow_weights_path,
        stream_path, outlet_cell_deque)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_r_sum_avail_pour(
        l_sum_avail_path, outflow_weights_path, outflow_direction_path,
        l_sum_avail_pour_path):
    """Calculate how r_sum_avail r_sum_avail_pours directly into its neighbors"""
    out_dir = os.path.dirname(l_sum_avail_path)
    l_sum_avail_raster = gdal.OpenEx(l_sum_avail_path)
    l_sum_avail_band = l_sum_avail_raster.GetRasterBand(1)
    block_col_size, block_row_size = l_sum_avail_band.GetBlockSize()
    r_sum_nodata = natcap.invest.pygeoprocessing_0_3_3.geoprocessing.get_nodata_from_uri(
        l_sum_avail_path)

    cdef float r_sum_avail_pour_nodata = -1.0
    natcap.invest.pygeoprocessing_0_3_3.new_raster_from_base_uri(
        l_sum_avail_path, l_sum_avail_pour_path, 'GTiff', r_sum_avail_pour_nodata,
        gdal.GDT_Float32)
    l_sum_avail_pour_raster = gdal.OpenEx(l_sum_avail_pour_path, gdal.GA_Update)
    l_sum_avail_pour_band = l_sum_avail_pour_raster.GetRasterBand(1)

    n_rows = l_sum_avail_band.YSize
    n_cols = l_sum_avail_band.XSize

    n_global_block_rows = int(numpy.ceil(float(n_rows) / block_row_size))
    n_global_block_cols = int(numpy.ceil(float(n_cols) / block_col_size))

    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] l_sum_avail_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] l_sum_avail_pour_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)

    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)

    outflow_direction_raster = gdal.OpenEx(outflow_direction_path)
    outflow_direction_band = outflow_direction_raster.GetRasterBand(1)
    cdef float outflow_direction_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_direction_path)
    outflow_weights_raster = gdal.OpenEx(outflow_weights_path)
    outflow_weights_band = outflow_weights_raster.GetRasterBand(1)
    cdef float outflow_weights_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_weights_path)
    #make the memory block
    band_list = [
        l_sum_avail_band, outflow_direction_band, outflow_weights_band,
        l_sum_avail_pour_band]
    block_list = [
        l_sum_avail_block, outflow_direction_block, outflow_weights_block,
        l_sum_avail_pour_block]

    update_list = [False, False, False, True]
    cache_dirty[:] = 0

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols,
        block_row_size, block_col_size,
        band_list, block_list, update_list, cache_dirty)

    #center point of global index
    cdef int global_row, global_col #index into the overall raster
    cdef int row_index, col_index #the index of the cache block
    cdef int row_block_offset, col_block_offset #index into the cache block
    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    for global_block_row in xrange(n_global_block_rows):
        for global_block_col in xrange(n_global_block_cols):
            xoff = global_block_col * block_col_size
            yoff = global_block_row * block_row_size
            win_xsize = min(block_col_size, n_cols - xoff)
            win_ysize = min(block_row_size, n_rows - yoff)

            for global_row in xrange(yoff, yoff+win_ysize):
                for global_col in xrange(xoff, xoff+win_xsize):

                    block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)
                    if l_sum_avail_block[row_index, col_index, row_block_offset, col_block_offset] == r_sum_nodata:
                        l_sum_avail_pour_block[row_index, col_index, row_block_offset, col_block_offset] = r_sum_avail_pour_nodata
                        cache_dirty[row_index, col_index] = 1
                        continue

                    r_sum_avail_pour_sum = 0.0
                    for direction_index in xrange(8):
                        #get percent flow from neighbor to current cell
                        neighbor_row = global_row + row_offsets[direction_index]
                        neighbor_col = global_col + col_offsets[direction_index]

                        #See if neighbor out of bounds
                        if (neighbor_row < 0 or neighbor_row >= n_rows or neighbor_col < 0 or neighbor_col >= n_cols):
                            continue

                        block_cache.update_cache(neighbor_row, neighbor_col, &neighbor_row_index, &neighbor_col_index, &neighbor_row_block_offset, &neighbor_col_block_offset)
                        #if neighbor inflows
                        neighbor_direction = outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
                        if neighbor_direction == outflow_direction_nodata:
                            continue

                        if l_sum_avail_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == r_sum_nodata:
                            continue

                        #check if the cell flows directly, or is one index off
                        if (inflow_offsets[direction_index] != neighbor_direction and
                                ((inflow_offsets[direction_index] - 1) % 8) != neighbor_direction):
                            #then neighbor doesn't inflow into current cell
                            continue

                        #Calculate the outflow weight
                        outflow_weight = outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]

                        if ((inflow_offsets[direction_index] - 1) % 8) == neighbor_direction:
                            outflow_weight = 1.0 - outflow_weight

                        if outflow_weight <= 0.0:
                            continue
                        r_sum_avail_pour_sum += l_sum_avail_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] * outflow_weight

                    block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)
                    l_sum_avail_pour_block[row_index, col_index, row_block_offset, col_block_offset] = r_sum_avail_pour_sum
                    cache_dirty[row_index, col_index] = 1
    block_cache.flush_cache()


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def route_baseflow_sum(
    dem_path, l_path, l_avail_path, l_sum_path,
    outflow_direction_path, outflow_weights_path, stream_path, b_sum_path):

    #Pass transport
    cdef time_t start
    time(&start)

    cdef deque[int] cells_to_process
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.find_outlets(dem_path, outflow_direction_path, cells_to_process)

    cdef c_set[int] cells_in_queue
    for cell in cells_to_process:
        cells_in_queue.insert(cell)

    cdef float pixel_area = (
        natcap.invest.pygeoprocessing_0_3_3.geoprocessing.get_cell_size_from_uri(dem_path) ** 2)

    #load a base raster so we can determine the n_rows/cols
    outflow_direction_raster = gdal.OpenEx(outflow_direction_path, gdal.GA_ReadOnly)
    cdef int n_cols = outflow_direction_raster.RasterXSize
    cdef int n_rows = outflow_direction_raster.RasterYSize
    outflow_direction_band = outflow_direction_raster.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = outflow_direction_band.GetBlockSize()

    #center point of global index
    cdef int global_row, global_col  # index into the overall raster
    cdef int row_index, col_index  # the index of the cache block
    cdef int row_block_offset, col_block_offset  # index into the cache block
    cdef int global_block_row, global_block_col  # used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col  # neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index  # neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset  # index into the neighbor cache block

    #define all the single caches
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] l_sum_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] l_avail_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] l_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] b_sum_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] stream_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)

    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)

    cdef int outflow_direction_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_direction_path)

    outflow_weights_raster = gdal.OpenEx(outflow_weights_path)
    outflow_weights_band = outflow_weights_raster.GetRasterBand(1)
    cdef float outflow_weights_nodata = natcap.invest.pygeoprocessing_0_3_3.get_nodata_from_uri(
        outflow_weights_path)

    #Create output arrays qfi and local_recharge and local_recharge_avail
    l_raster = gdal.OpenEx(l_path)
    l_band = l_raster.GetRasterBand(1)
    l_avail_raster = gdal.OpenEx(l_avail_path)
    l_avail_band = l_avail_raster.GetRasterBand(1)
    l_sum_raster = gdal.OpenEx(l_sum_path)
    l_sum_band = l_sum_raster.GetRasterBand(1)
    cdef float l_sum_nodata = l_sum_band.GetNoDataValue()

    stream_raster = gdal.OpenEx(stream_path)
    stream_band = stream_raster.GetRasterBand(1)

    cdef float b_sum_nodata = -9999.0
    natcap.invest.pygeoprocessing_0_3_3.new_raster_from_base_uri(
        outflow_direction_path, b_sum_path, 'GTiff', b_sum_nodata,
        gdal.GDT_Float32, fill_value=b_sum_nodata)
    b_sum_raster = gdal.OpenEx(b_sum_path, gdal.GA_Update)
    b_sum_band = b_sum_raster.GetRasterBand(1)

    band_list = [
        outflow_direction_band, outflow_weights_band, l_avail_band,
        l_sum_band, l_band, stream_band, b_sum_band]
    block_list = [
        outflow_direction_block, outflow_weights_block, l_avail_block,
        l_sum_block, l_block, stream_block, b_sum_block]
    update_list = [False] * 6 + [True]
    cache_dirty[:] = 0

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols,
        block_row_size, block_col_size,
        band_list, block_list, update_list, cache_dirty)

    #Diagonal offsets are based off the following index notation for neighbors
    #    3 2 1
    #    4 p 0
    #    5 6 7

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int flat_index
    cdef float outflow_weight
    cdef int outflow_direction
    cdef float neighbor_b_sum_i
    cdef float neighbor_l_i
    cdef float neighbor_l_avail_i
    cdef float neighbor_l_sum_i
    cdef float b_sum_i
    cdef float l_i
    cdef float l_avail_i
    cdef float l_sum_i
    cdef int neighbor_direction

    cdef time_t last_time, current_time
    time(&last_time)
    LOGGER.info(
                'START cells_to_process on B route size: %d',
                cells_to_process.size())
    while cells_to_process.size() > 0:
        flat_index = cells_to_process.front()
        cells_to_process.pop_front()
        cells_in_queue.erase(flat_index)
        global_row = flat_index / n_cols
        global_col = flat_index % n_cols

        block_cache.update_cache(
            global_row, global_col, &row_index, &col_index,
            &row_block_offset, &col_block_offset)

        outflow_weight = outflow_weights_block[
            row_index, col_index, row_block_offset, col_block_offset]
        outflow_direction = outflow_direction_block[
            row_index, col_index, row_block_offset, col_block_offset]

        time(&current_time)
        if current_time - last_time > 5.0:
            last_time = current_time
            LOGGER.info(
                'cells_to_process on B route size: %d',
                cells_to_process.size())

        #if cell is processed, then skip
        if b_sum_block[row_index, col_index, row_block_offset, col_block_offset] != b_sum_nodata:
            continue

        l_sum_i = l_sum_block[
            row_index, col_index, row_block_offset, col_block_offset]

        # if current cell doesn't outflow, base case is B == l_sum
        if outflow_direction == outflow_direction_nodata:
            if l_sum_i == l_sum_nodata:
                b_sum_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            else:
                b_sum_block[row_index, col_index, row_block_offset, col_block_offset] = l_sum_i
            cache_dirty[row_index, col_index] = 1
        elif stream_block[row_index, col_index, row_block_offset, col_block_offset] == 1:
            b_sum_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            cache_dirty[row_index, col_index] = 1
        else:
            downstream_calculated = 1
            b_sum_i = 0.0
            for neighbor_index in xrange(2):
                if neighbor_index == 1:
                    outflow_direction = (outflow_direction + 1) % 8
                    outflow_weight = 1.0 - outflow_weight

                if outflow_weight <= 0.0:
                    #doesn't flow here, so skip
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

                if stream_block[
                        neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset, neighbor_col_block_offset] == 1:
                    #calc base case
                    b_sum_i += outflow_weight
                else:
                    if b_sum_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset, neighbor_col_block_offset] == b_sum_nodata:
                        #push neighbor on stack
                        downstream_calculated = 0
                        neighbor_flat_index = neighbor_row * n_cols + neighbor_col
                        #push original on the end of the deque
                        if (cells_in_queue.find(flat_index) ==
                            cells_in_queue.end()):
                            cells_to_process.push_back(flat_index)
                            cells_in_queue.insert(flat_index)

                        #push neighbor on front of deque
                        if (cells_in_queue.find(neighbor_flat_index) ==
                            cells_in_queue.end()):
                            cells_to_process.push_front(neighbor_flat_index)
                            cells_in_queue.insert(neighbor_flat_index)
                    else:
                        #calculate downstream contribution
                        neighbor_l_i = l_block[
                            neighbor_row_index, neighbor_col_index,
                            neighbor_row_block_offset, neighbor_col_block_offset]
                        neighbor_l_sum_i = l_sum_block[
                            neighbor_row_index, neighbor_col_index,
                            neighbor_row_block_offset, neighbor_col_block_offset]
                        neighbor_l_avail_i = l_avail_block[
                            neighbor_row_index, neighbor_col_index,
                            neighbor_row_block_offset, neighbor_col_block_offset]
                        neighbor_b_sum_i = b_sum_block[
                            neighbor_row_index, neighbor_col_index,
                            neighbor_row_block_offset, neighbor_col_block_offset]
                        # make sure there's no zero in the denominator; if so
                        # result would be zero so don't add anything in
                        if (neighbor_l_sum_i - neighbor_l_i > 0) and neighbor_l_sum_i > 0:
                            b_sum_i += (outflow_weight * (
                                1 - neighbor_l_avail_i / neighbor_l_sum_i) *
                                neighbor_b_sum_i / (neighbor_l_sum_i - neighbor_l_i))

            if downstream_calculated:
                b_sum_i *= l_sum_i
                block_cache.update_cache(
                    global_row, global_col, &row_index, &col_index,
                    &row_block_offset, &col_block_offset)
                b_sum_block[row_index, col_index, row_block_offset, col_block_offset] = b_sum_i
                cache_dirty[row_index, col_index] = 1

        #put upstream neighbors on stack for processing
        for neighbor_index in xrange(8):
            neighbor_row = neighbor_row_offset[neighbor_index] + global_row
            neighbor_col = neighbor_col_offset[neighbor_index] + global_col

            if (neighbor_row >= n_rows or neighbor_row < 0 or
                    neighbor_col >= n_cols or neighbor_col < 0):
                continue

            block_cache.update_cache(
                neighbor_row, neighbor_col,
                &neighbor_row_index, &neighbor_col_index,
                &neighbor_row_block_offset,
                &neighbor_col_block_offset)

            neighbor_direction = outflow_direction_block[
                neighbor_row_index, neighbor_col_index,
                neighbor_row_block_offset, neighbor_col_block_offset]
            if neighbor_direction == outflow_direction_nodata:
                continue

            #check if the cell flows directly, or is one index off
            if (inflow_offsets[neighbor_index] != neighbor_direction and
                    ((inflow_offsets[neighbor_index] - 1) % 8) != neighbor_direction):
                #then neighbor doesn't inflow into current cell
                continue

            #Calculate the outflow weight
            outflow_weight = outflow_weights_block[
                neighbor_row_index, neighbor_col_index,
                neighbor_row_block_offset, neighbor_col_block_offset]

            if ((inflow_offsets[neighbor_index] - 1) % 8) == neighbor_direction:
                outflow_weight = 1.0 - outflow_weight

            if outflow_weight <= 0.0:
                continue

            #already processed, no need to loop on it again
            if b_sum_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset] != b_sum_nodata:
                continue

            neighbor_flat_index = neighbor_row * n_cols + neighbor_col
            if cells_in_queue.find(neighbor_flat_index) == cells_in_queue.end():
                cells_to_process.push_back(neighbor_flat_index)
                cells_in_queue.insert(neighbor_flat_index)

    block_cache.flush_cache()
