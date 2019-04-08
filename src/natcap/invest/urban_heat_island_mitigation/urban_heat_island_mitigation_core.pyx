"""Urban Heat Island Cython Core."""
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

ctypedef pair[int, int] PointPair

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

def blob_mask(
        mask_raster_path_band, target_blob_id_raster_path,
        id_count_map_pickle_path):
    """Blob up touching raster fronts.

    Parameters:
        mask_raster_path_band (tuple): path/band tuple to a raster that has
            1 or 0 or nodata. Intent is to blob all the contiguous

    """
    cdef int blob_id_nodata = -1
    pygeoprocessing.new_raster_from_base(
        mask_raster_path_band[0], target_blob_id_raster_path, gdal.GDT_Int32,
        [blob_id_nodata], fill_value_list=[blob_id_nodata])

    cdef int n_cols, n_rows
    n_cols, n_rows = pygeoprocessing.get_raster_info(
        target_blob_id_raster_path)['raster_size']

    cdef _ManagedRaster mask_raster = _ManagedRaster(
        mask_raster_path_band[0], mask_raster_path_band[1], False)
    cdef _ManagedRaster blob_id_raster = _ManagedRaster(
        target_blob_id_raster_path, 1, True)

    cdef stack[PointPair] search_stack

    cdef int i, j, x, y, xn, yn, d
    cdef int xmin, xmax, ymin, ymax
    cdef int current_blob_id = -1

    for offset_dict in pygeoprocessing.iterblocks(
            mask_raster_path_band, offset_only=True):
        xmin = offset_dict['xoff']
        xmax = xmin+offset_dict['win_xsize']
        ymin = offset_dict['yoff']
        ymax = ymin+offset_dict['win_ysize']

        for j in range(ymin, ymax):
            for i in range(xmin, xmax):
                if (blob_id_raster.get(i, j) == blob_id_nodata and
                        mask_raster.get(i, j) == 1):
                    current_blob_id += 1
                    search_stack.push(PointPair(i, j))
                    blob_id_raster.set(i, j, current_blob_id)
                    while search_stack.size() > 0:
                        x = search_stack.top().first
                        y = search_stack.top().second
                        search_stack.pop()
                        for d in range(8):
                            xn = x + NEIGHBOR_OFFSET_ARRAY[d*2]
                            if xn < 0 or xn >= n_cols:
                                continue
                            yn = y + NEIGHBOR_OFFSET_ARRAY[d*2+1]
                            if yn < 0 or yn >= n_rows:
                                continue
                            if (mask_raster.get(xn, yn) == 1 and
                                    blob_id_raster.get(xn, yn) ==
                                    blob_id_nodata):
                                blob_id_raster.set(xn, yn, current_blob_id)
                                search_stack.push(PointPair(xn, yn))
