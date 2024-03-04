# cython: language_level=3
# distutils: language = c++
import os

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time as ctime
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp.list cimport list as clist
from libcpp.stack cimport stack
from libcpp.vector cimport vector


cdef void route(object flow_dir_path, function_type seed_fn,
        function_type route_fn, object seed_fn_args, object route_fn_args):
    """
    Args:
        seed_fn (callable): function that accepts an (x, y) coordinate
            and returns a bool indicating if the pixel is a seed
        route_fn (callable): function that accepts an (x, y) coordinate
            and performs whatever routing operation is needed on that pixel.

    Returns:
        None
    """

    cdef long win_xsize, win_ysize, xoff, yoff, flat_index
    cdef int col_index, row_index, global_col, global_row
    cdef stack[long] processing_stack
    cdef long n_cols, n_rows
    cdef vector[long] next_pixels

    flow_dir_info = pygeoprocessing.get_raster_info(flow_dir_path)
    n_cols, n_rows = flow_dir_info['raster_size']

    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        for row_index in range(win_ysize):
            global_row = yoff + row_index
            for col_index in range(win_xsize):

                global_col = xoff + col_index

                if seed_fn(global_col, global_row, *seed_fn_args):
                    processing_stack.push(global_row * n_cols + global_col)

        while processing_stack.size() > 0:
            # loop invariant, we don't push a cell on the stack that
            # hasn't already been set for processing.
            flat_index = processing_stack.top()
            processing_stack.pop()
            global_row = flat_index // n_cols
            global_col = flat_index % n_cols

            next_pixels = route_fn(global_col, global_row, *route_fn_args)
            for index in next_pixels:
                processing_stack.push(index)


# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair
# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# given the pixel neighbor numbering system
#  3 2 1
#  4 x 0
#  5 6 7
# These offsets are for the neighbor rows and columns
cdef int *ROW_OFFSETS = [0, -1, -1, -1,  0,  1, 1, 1]
cdef int *COL_OFFSETS = [1,  1,  0, -1, -1, -1, 0, 1]
cdef int *FLOW_DIR_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]

# if a pixel `x` has a neighbor `n` in position `i`,
# then `n`'s neighbor in position `inflow_offsets[i]`
# is the original pixel `x`
cdef int *INFLOW_OFFSETS = [4, 5, 6, 7, 0, 1, 2, 3]

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.  Copied from src/pygeoprocessing/routing/routing.pyx,
# revision 891288683889237cfd3a3d0a1f09483c23489fca.
cdef class ManagedRaster:

    def __cinit__(self, raster_path, band_id, write_mode):
        """Create new instance of Managed Raster.

        Args:
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
        self.block_xmod = self.block_xsize - 1
        self.block_ymod = self.block_ysize - 1
        self.pixel_x_size, pixel_y_size = raster_info['pixel_size']
        self.nodata = raster_info['nodata'][band_id - 1]

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
        """Deallocate ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
        """Close the ManagedRaster and free up resources.

            This call writes any dirty blocks to disk, frees up the memory
            allocated as part of the cache, and frees all GDAL references.

            Any subsequent calls to any other functions in ManagedRaster will
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


cdef class ManagedFlowDirRaster(ManagedRaster):

    cdef bint is_local_high_point(self, long xi, long yi):
        """Check if a given pixel is a local high point.

        Args:
            xi (int): x coord in pixel space of the pixel to consider
            yi (int): y coord in pixel space of the pixel to consider

        Returns:
            True if the pixel is a local high point, i.e. it has no
            upslope neighbors; False otherwise.
        """
        return self.get_upslope_neighbors(xi, yi).size() == 0

    @cython.cdivision(True)
    cdef vector[NeighborTuple] get_upslope_neighbors(
            ManagedFlowDirRaster self, long xi, long yi):
        """Return upslope neighbors of a given pixel.

        Args:
            xi (int): x coord in pixel space of the pixel to consider
            yi (int): y coord in pixel space of the pixel to consider

        Returns:
            libcpp.vector of NeighborTuples. Each NeighborTuple has
            the attributes ``direction`` (integer flow direction 0-7
            of the neighbor relative to the original pixel), ``x``
            and ``y`` (integer coordinates of the neighbor in pixel
            space), and ``flow_proportion`` (fraction of the flow
            from the neighbor that flows to the original pixel).
        """
        cdef int n_dir, flow_dir_j, idx
        cdef long xj, yj
        cdef float flow_ji, flow_dir_j_sum

        cdef NeighborTuple n
        cdef vector[NeighborTuple] upslope_neighbor_tuples

        for n_dir in range(8):
            xj = xi + COL_OFFSETS[n_dir]
            yj = yi + ROW_OFFSETS[n_dir]
            if (xj < 0 or xj >= self.raster_x_size or
                    yj < 0 or yj >= self.raster_y_size):
                continue
            flow_dir_j = <int>self.get(xj, yj)
            flow_ji = (0xF & (flow_dir_j >> (4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])))

            if flow_ji:
                flow_dir_j_sum = 0
                for idx in range(8):
                    flow_dir_j_sum += (flow_dir_j >> (idx * 4)) & 0xF
                n.direction = n_dir
                n.x = xj
                n.y = yj
                n.flow_proportion = flow_ji / flow_dir_j_sum
                upslope_neighbor_tuples.push_back(n)

        return upslope_neighbor_tuples

    @cython.cdivision(True)
    cdef vector[NeighborTuple] get_downslope_neighbors(
            ManagedFlowDirRaster self, long xi, long yi, bint skip_oob=True):
        """Return downslope neighbors of a given pixel.

        Args:
            xi (int): x coord in pixel space of the pixel to consider
            yi (int): y coord in pixel space of the pixel to consider
            skip_oob (bool): if True, do not return neighbors that fall
                outside the raster bounds.

        Returns:
            libcpp.vector of NeighborTuples. Each NeighborTuple has
            the attributes ``direction`` (integer flow direction 0-7
            of the neighbor relative to the original pixel), ``x``
            and ``y`` (integer coordinates of the neighbor in pixel
            space), and ``flow_proportion`` (fraction of the flow
            from the neighbor that flows to the original pixel).
        """
        cdef int n_dir
        cdef long xj, yj
        cdef float flow_ij

        cdef NeighborTuple n
        cdef vector[NeighborTuple] downslope_neighbor_tuples

        cdef int flow_dir = <int>self.get(xi, yi)
        cdef float flow_sum = 0

        cdef int i = 0
        for n_dir in range(8):
            # flows in this direction
            xj = xi + COL_OFFSETS[n_dir]
            yj = yi + ROW_OFFSETS[n_dir]
            if skip_oob and (xj < 0 or xj >= self.raster_x_size or
                    yj < 0 or yj >= self.raster_y_size):
                continue
            flow_ij = (flow_dir >> (n_dir * 4)) & 0xF
            flow_sum += flow_ij
            if flow_ij:
                n = NeighborTuple(
                    direction=n_dir,
                    x=xj,
                    y=yj,
                    flow_proportion=flow_ij)
                downslope_neighbor_tuples.push_back(n)
                i += 1

        for j in range(i):
            downslope_neighbor_tuples[j].flow_proportion = (
                downslope_neighbor_tuples[j].flow_proportion / flow_sum)

        return downslope_neighbor_tuples
