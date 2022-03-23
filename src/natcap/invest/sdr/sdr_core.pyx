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


# cmath is supposed to have M_SQRT2, but tests have been failing recently
# due to a missing symbol.
cdef double SQRT2 = cmath.sqrt(2)
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
    """Calculate sediment deposition layer.

    This algorithm outputs both sediment deposition (r_i) and flux (f_i)::

        r_i  =      dr_i  * (sum over j ∈ J of f_j * p(i,j)) + E'_i

        f_i  = (1 - dr_i) * (sum over j ∈ J of f_j * p(i,j)) + E'_i


                (sum over k ∈ K of SDR_k * p(i,k)) - SDR_i
        dr_i = --------------------------------------------
                              (1 - SDR_i)

    where:

    - ``p(i,j)`` is the proportion of flow from pixel ``i`` into pixel ``j``
    - ``J`` is the set of pixels that are immediate upslope neighbors of
      pixel ``i``
    - ``K`` is the set of pixels that are immediate downslope neighbors of
      pixel ``i``
    - ``E'`` is ``USLE * (1 - SDR)``, the amount of sediment loss from pixel
      ``i`` that doesn't reach a stream (``e_prime_path``)
    - ``SDR`` is the sediment delivery ratio (``sdr_path``)

    ``f_i`` is recursively defined in terms of ``i``'s upslope neighbors.
    The algorithm begins from seed pixels that are local high points and so
    have no upslope neighbors. It works downslope from each seed pixel,
    only adding a pixel to the stack when all its upslope neighbors are
    already calculated.

    Note that this function is designed to be used in the context of the SDR
    model. Because the algorithm is recursive upslope and downslope of each
    pixel, nodata values in the SDR input would propagate along the flow path.
    This case is not handled because we assume the SDR and flow dir inputs
    will come from the SDR model and have nodata in the same places.

    Args:
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
    LOGGER.info('Calculate sediment deposition')
    cdef float target_nodata = -1
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, target_sediment_deposition_path,
        gdal.GDT_Float32, [target_nodata])
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, f_path,
        gdal.GDT_Float32, [target_nodata])

    cdef _ManagedRaster mfd_flow_direction_raster = _ManagedRaster(
        mfd_flow_direction_path, 1, False)
    cdef _ManagedRaster e_prime_raster = _ManagedRaster(
        e_prime_path, 1, False)
    cdef _ManagedRaster sdr_raster = _ManagedRaster(sdr_path, 1, False)
    cdef _ManagedRaster f_raster = _ManagedRaster(f_path, 1, True)
    cdef _ManagedRaster sediment_deposition_raster = _ManagedRaster(
        target_sediment_deposition_path, 1, True)

    # given the pixel neighbor numbering system
    #  3 2 1
    #  4 x 0
    #  5 6 7
    # if a pixel `x` has a neighbor `n` in position `i`,
    # then `n`'s neighbor in position `inflow_offsets[i]`
    # is the original pixel `x`
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int n_cols, n_rows
    flow_dir_info = pygeoprocessing.get_raster_info(mfd_flow_direction_path)
    n_cols, n_rows = flow_dir_info['raster_size']
    cdef int mfd_nodata = 0
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
    cdef float downslope_sdr_weighted_sum, sdr_i, sdr_j
    cdef float p_j, p_val

    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Sediment deposition %.2f%% complete', 100 * (
            (xoff * yoff) / float(n_cols*n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index
                # check if this is a good seed pixel ( a local high point)
                if mfd_flow_direction_raster.get(seed_col, seed_row) == mfd_nodata:
                    continue
                seed_pixel = 1
                # iterate over each of the pixel's neighbors
                for j in range(8):
                    # skip if the neighbor is outside the raster bounds
                    neighbor_row = seed_row + ROW_OFFSETS[j]
                    if neighbor_row < 0 or neighbor_row >= n_rows:
                        continue
                    neighbor_col = seed_col + COL_OFFSETS[j]
                    if neighbor_col < 0 or neighbor_col >= n_cols:
                        continue
                    # skip if the neighbor's flow direction is undefined
                    neighbor_flow_val = <int>mfd_flow_direction_raster.get(
                        neighbor_col, neighbor_row)
                    if neighbor_flow_val == mfd_nodata:
                        continue
                    # if the neighbor flows into it, it's not a local high
                    # point and so can't be a seed pixel
                    neighbor_flow_weight = (
                        neighbor_flow_val >> (inflow_offsets[j]*4)) & 0xF
                    if neighbor_flow_weight > 0:
                        seed_pixel = 0  # neighbor flows in, not a seed
                        break

                # if this can be a seed pixel and hasn't already been
                # calculated, put it on the stack
                if seed_pixel and sediment_deposition_raster.get(
                        seed_col, seed_row) == target_nodata:
                    processing_stack.push(seed_row * n_cols + seed_col)

                while processing_stack.size() > 0:
                    # loop invariant: cell has all upslope neighbors
                    # processed. this is true for seed pixels because they
                    # have no upslope neighbors.
                    flat_index = processing_stack.top()
                    processing_stack.pop()
                    global_row = flat_index // n_cols
                    global_col = flat_index % n_cols

                    # (sum over j ∈ J of f_j * p(i,j) in the equation for r_i)
                    # calculate the upslope f_j contribution to this pixel,
                    # the weighted sum of flux flowing onto this pixel from
                    # all neighbors
                    f_j_weighted_sum = 0
                    for j in range(8):
                        neighbor_row = global_row + ROW_OFFSETS[j]
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            continue
                        neighbor_col = global_col + COL_OFFSETS[j]
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            continue

                        # see if there's an inflow from the neighbor to the
                        # pixel
                        neighbor_flow_val = (
                            <int>mfd_flow_direction_raster.get(
                                neighbor_col, neighbor_row))
                        neighbor_flow_weight = (
                            neighbor_flow_val >> (inflow_offsets[j]*4)) & 0xF
                        if neighbor_flow_weight > 0:
                            f_j = f_raster.get(neighbor_col, neighbor_row)
                            if f_j == target_nodata:
                                continue
                            # sum up the neighbor's flow dir values in each
                            # direction.
                            # flow dir values are relative to the total
                            neighbor_flow_sum = 0
                            for k in range(8):
                                neighbor_flow_sum += (
                                    neighbor_flow_val >> (k*4)) & 0xF
                            # get the proportion of the neighbor's flow that
                            # flows into the original pixel
                            p_val = neighbor_flow_weight / neighbor_flow_sum
                            # add the neighbor's flux value, weighted by the
                            # flow proportion
                            f_j_weighted_sum += p_val * f_j

                    # calculate sum of SDR values of immediate downslope
                    # neighbors, weighted by proportion of flow into each
                    # neighbor
                    # (sum over k ∈ K of SDR_k * p(i,k) in the equation above)
                    downslope_sdr_weighted_sum = 0
                    flow_val = <int>mfd_flow_direction_raster.get(
                        global_col, global_row)
                    flow_sum = 0
                    for k in range(8):
                        flow_sum += (flow_val >> (k*4)) & 0xF

                    # iterate over the neighbors again
                    for j in range(8):
                        # skip if neighbor is outside the raster boundaries
                        neighbor_row = global_row + ROW_OFFSETS[j]
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            continue
                        neighbor_col = global_col + COL_OFFSETS[j]
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            continue
                        # if it is a downslope neighbor, add to the sum and
                        # check if it can be pushed onto the stack yet
                        flow_weight = (flow_val >> (j*4)) & 0xF
                        if flow_weight > 0:
                            sdr_j = sdr_raster.get(neighbor_col, neighbor_row)
                            if sdr_j == sdr_nodata:
                                continue
                            if sdr_j == 0:
                                # this means it's a stream, for SDR deposition
                                # purposes, we set sdr to 1 to indicate this
                                # is the last step on which to retain sediment
                                sdr_j = 1
                            p_j = flow_weight / flow_sum
                            downslope_sdr_weighted_sum += sdr_j * p_j

                            # check if we can add neighbor j to the stack yet
                            #
                            # if there is a downslope neighbor it
                            # couldn't have been pushed on the processing
                            # stack yet, because the upslope was just
                            # completed
                            upslope_neighbors_processed = 1
                            # iterate over each neighbor-of-neighbor
                            for k in range(8):
                                # no need to push the one we're currently
                                # calculating back onto the stack
                                if inflow_offsets[k] == j:
                                    continue
                                # skip if neighbor-of-neighbor is outside
                                # raster bounds
                                ds_neighbor_row = (
                                    neighbor_row + ROW_OFFSETS[k])
                                if ds_neighbor_row < 0 or ds_neighbor_row >= n_rows:
                                    continue
                                ds_neighbor_col = (
                                    neighbor_col + COL_OFFSETS[k])
                                if ds_neighbor_col < 0 or ds_neighbor_col >= n_cols:
                                    continue
                                # if any upslope neighbor of j hasn't been
                                # calculated, we can't push j onto the stack
                                # yet
                                ds_neighbor_flow_val = (
                                    <int>mfd_flow_direction_raster.get(
                                        ds_neighbor_col, ds_neighbor_row))
                                if (ds_neighbor_flow_val >> (
                                        inflow_offsets[k]*4)) & 0xF > 0:
                                    if (sediment_deposition_raster.get(
                                            ds_neighbor_col, ds_neighbor_row) ==
                                            target_nodata):
                                        upslope_neighbors_processed = 0
                                        break
                            # if all upslope neighbors of neighbor j are
                            # processed, we can push j onto the stack.
                            if upslope_neighbors_processed:
                                processing_stack.push(
                                    neighbor_row * n_cols +
                                    neighbor_col)

                    # nodata pixels should propagate to the results
                    sdr_i = sdr_raster.get(global_col, global_row)
                    if sdr_i == sdr_nodata:
                        continue
                    e_prime_i = e_prime_raster.get(global_col, global_row)
                    if e_prime_i == e_prime_nodata:
                        continue

                    if downslope_sdr_weighted_sum < sdr_i:
                        # i think this happens because of our low resolution
                        # flow direction, it's okay to zero out.
                        downslope_sdr_weighted_sum = sdr_i

                    # these correspond to the full equations for
                    # dr_i, r_i, and f_i given in the docstring
                    dr_i = (downslope_sdr_weighted_sum - sdr_i) / (1 - sdr_i)
                    r_i = dr_i * (e_prime_i + f_j_weighted_sum)
                    f_i = (1 - dr_i) * (e_prime_i + f_j_weighted_sum)

                    # On large flow paths, it's possible for r_i and f_i to
                    # have very small negative values that are numerically
                    # equivalent to 0. These negative values were raising
                    # questions on the forums and it's easier to clamp the
                    # values here than to explain IEEE 754.
                    if r_i < 0:
                        r_i = 0
                    if f_i < 0:
                        f_i = 0

                    sediment_deposition_raster.set(global_col, global_row, r_i)
                    f_raster.set(global_col, global_row, f_i)

    LOGGER.info('Sediment deposition 100% complete')
    sediment_deposition_raster.close()


def calculate_average_aspect(
        mfd_flow_direction_path, target_average_aspect_path):
    """Calculate the Weighted Average Aspect Ratio from MFD.

    Calculates the average aspect ratio weighted by proportional flow
    direction.

    Args:
        mfd_flow_direction_path (string): The path to an MFD flow direction
            raster.
        target_average_aspect_path (string): The path to where the calculated
            weighted average aspect raster should be written.

    Returns:
        ``None``.

    """
    LOGGER.info('Calculating average aspect')

    cdef float average_aspect_nodata = -1
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

    cdef int seed_row = 0
    cdef int seed_col = 0
    cdef int n_pixels_visited = 0
    cdef int win_xsize, win_ysize, xoff, yoff
    cdef int row_index, col_index, neighbor_index
    cdef int flow_weight_in_direction
    cdef int weight_sum
    cdef int seed_flow_value
    cdef float aspect_weighted_average, aspect_weighted_sum

    # the flow_lengths array is the functional equivalent
    # of calculating |sin(alpha)| + |cos(alpha)|.
    cdef float* flow_lengths = [
        1, <float>SQRT2,
        1, <float>SQRT2,
        1, <float>SQRT2,
        1, <float>SQRT2
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

        LOGGER.info('Average aspect %.2f%% complete', 100 * (
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
                aspect_weighted_sum = 0
                for neighbor_index in range(8):
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row == -1 or neighbor_row == n_rows:
                        continue

                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col == -1 or neighbor_col == n_cols:
                        continue

                    flow_weight_in_direction = (seed_flow_value >> (
                        neighbor_index * 4) & 0xF)
                    weight_sum += flow_weight_in_direction

                    aspect_weighted_sum += (
                        flow_lengths[neighbor_index] *
                        flow_weight_in_direction)

                # Weight sum should never be less than 0.
                # Since it's an int, we can compare it directly against the
                # value of 0.
                if weight_sum == 0:
                    aspect_weighted_average = average_aspect_nodata
                else:
                    # We already know that weight_sum will be > 0 because we
                    # check for it in the condition above.
                    with cython.cdivision(True):
                        aspect_weighted_average = (
                            aspect_weighted_sum / <float>weight_sum)

                average_aspect_raster.set(
                    seed_col, seed_row, aspect_weighted_average)

        n_pixels_visited += win_xsize * win_ysize

    LOGGER.info('Average aspect 100.00% complete')

    mfd_flow_direction_raster.close()
    average_aspect_raster.close()
