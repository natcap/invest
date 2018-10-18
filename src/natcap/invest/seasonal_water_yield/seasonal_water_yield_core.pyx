# cython: profile=False

import logging
import os
import collections
import sys
import gc
import pygeoprocessing

import numpy
cimport numpy
cimport cython
import osgeo
from osgeo import gdal
from cython.operator cimport dereference as deref

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time_t
from libc.time cimport time as ctime
from libcpp.list cimport list as clist
from libcpp.set cimport set as cset
from libcpp.pair cimport pair

from libc.time cimport time as ctime

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef int N_MONTHS = 12

cdef double PI = 3.141592653589793238462643383279502884
cdef double INF = numpy.inf
cdef double IMPROBABLE_FLOAT_NOATA = -1.23789789e29

# used to loop over neighbors and offset the x/y values as defined below
#  321
#  4x0
#  567
cdef int* NEIGHBOR_OFFSET_ARRAY = [
    1, 0,  # 0
    1, -1,  # 1
    0, -1,  # 2
    -1, -1,  # 3
    -1, 0,  # 4
    -1, 1,  # 5
    0, 1,  # 6
    1, 1  # 7
    ]


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


def calculate_local_recharge(
        precip_path_list, et0_path_list, qfm_path_list, flow_dir_path,
        dem_path, lulc_path, alpha_month, beta_i, gamma, stream_path, li_path,
        kc_path_list, li_avail_path, l_sum_avail_path, aet_path):
    """
        # call through to a cython function that does the necessary routing
        # between AET and L.sum.avail in equation [7], [4], and [3]

target_path_list=[
                file_registry['l_path'],
                file_registry['l_avail_path'],
                file_registry['l_sum_avail_path'],
                file_registry['aet_path']],
    """

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    LOGGER.debug('calculate_local_recharge')

    # determine dem nodata in the working type, or set an improbable value
    # if one can't be determined
    dem_raster_info = pygeoprocessing.get_raster_info(dem_path)
    base_nodata = dem_raster_info['nodata'][0]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    cdef _ManagedRaster flow_raster = _ManagedRaster(
        flow_dir_path, 1, 0)
    cdef _ManagedRaster dem_raster = _ManagedRaster(
        dem_path, 1, 0)

    LOGGER.debug(flow_raster)

    for offset_dict in pygeoprocessing.iterblocks(
            dem_path, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.2f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # search block for locally undrained pixels
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_val = dem_raster.get(yi_root+yi, xi_root+xi)
                if center_val == dem_nodata:
                    continue

                # this value is set in case it turns out to be the root of a
                # pit, we'll start the fill from this pixel in the last phase
                # of the algorithm
                xi_root = xi-1+xoff
                yi_root = yi-1+yoff

                # search neighbors for downhill or nodata
                downhill_neighbor = 0
                nodata_neighbor = 0

                for i_n in xrange(8):
                    xi_n = xi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                    if (xi_n < 0 or xi_n >= raster_x_size or
                            yi_n < 0 or yi_n >= raster_y_size):
                        # it'll drain off the edge of the raster
                        nodata_neighbor = 1
                        break
                    n_height = dem_raster.get(xi_n, yi_n)
                    if n_height == dem_nodata:
                        # it'll drain to nodata
                        nodata_neighbor = 1
                        break
                    if n_height < center_val:
                        # it'll drain downhill
                        downhill_neighbor = 1
                        break


    """
    cdef deque[int] outlet_cell_deque
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.find_outlets(
        dem_path, flow_dir_path, outlet_cell_deque)
    # convert a dictionary of month -> alpha to an array of alphas in sorted
    # order by month
    cdef numpy.ndarray alpha_month_array = numpy.array(
        [x[1] for x in sorted(alpha_month.iteritems())])
    route_local_recharge(
        precip_path_list, et0_path_list, kc_path_list, li_path,
        li_avail_path, l_sum_avail_path, aet_path, alpha_month_array, beta_i,
        gamma, qfm_path_list, stream_path, outlet_cell_deque)
    """


def route_baseflow_sum(
        dem_path, l_path, l_avail_path, l_sum_path,
        stream_path, b_sum_path):
    LOGGER.error('implement route_baseflow_sum')
    cdef _ManagedRaster dem_raster = _ManagedRaster(dem_path, 1, 0)

    cdef time_t start
    time(&start)


def _generate_read_bounds(offset_dict, raster_x_size, raster_y_size):
    """Helper function to expand GDAL memory block read bound by 1 pixel.

    This function is used in the context of reading a memory block on a GDAL
    raster plus an additional 1 pixel boundary if it fits into an existing
    numpy array of size (2+offset_dict['y_size'], 2+offset_dict['x_size']).

    Parameters:
        offset_dict (dict): dictionary that has values for 'win_xsize',
            'win_ysize', 'xoff', and 'yoff' to describe the bounding box
            to read from the raster.
        raster_x_size, raster_y_size (int): these are the global x/y sizes
            of the raster that's being read.

    Returns:
        (xa, xb, ya, yb) (tuple of int): bounds that can be used to slice a
            numpy array of size
                (2+offset_dict['y_size'], 2+offset_dict['x_size'])
        modified_offset_dict (dict): a copy of `offset_dict` with the
            `win_*size` keys expanded if the modified bounding box will still
            fit on the array.
    """
    xa = 1
    xb = -1
    ya = 1
    yb = -1
    target_offset_dict = offset_dict.copy()
    if offset_dict['xoff'] > 0:
        xa = None
        target_offset_dict['xoff'] -= 1
        target_offset_dict['win_xsize'] += 1
    if offset_dict['yoff'] > 0:
        ya = None
        target_offset_dict['yoff'] -= 1
        target_offset_dict['win_ysize'] += 1
    if (offset_dict['xoff'] + offset_dict['win_xsize'] < raster_x_size):
        xb = None
        target_offset_dict['win_xsize'] += 1
    if (offset_dict['yoff'] + offset_dict['win_ysize'] < raster_y_size):
        yb = None
        target_offset_dict['win_ysize'] += 1
    return (xa, xb, ya, yb), target_offset_dict
