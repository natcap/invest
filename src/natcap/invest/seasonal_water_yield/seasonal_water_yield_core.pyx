# cython: profile=False
# cython: language_level=2
import logging
import os
import collections
import sys
import gc
import pygeoprocessing

import numpy
cimport numpy
cimport cython
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from cython.operator cimport dereference as deref

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libcpp.list cimport list as clist
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp.stack cimport stack
from libcpp.queue cimport queue

from libc.time cimport time as ctime
cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

cdef int is_close(double x, double y):
    return abs(x-y) <= (1e-8+1e-05*abs(y))

cdef extern from "LRUCache.h":
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)


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

# index into this array with a direction and get the index for the reverse
# direction. Useful for determining the direction a neighbor flows into a
# cell.
cdef int* FLOW_DIR_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]

# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**4

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


cpdef calculate_local_recharge(
        precip_path_list, et0_path_list, qf_m_path_list, flow_dir_mfd_path,
        kc_path_list, alpha_month_map, float beta_i, float gamma, stream_path,
        target_li_path, target_li_avail_path, target_l_sum_avail_path,
        target_aet_path, target_pi_path):
    """
    Calculate the rasters defined by equations [3]-[7].

    Note all input rasters must be in the same coordinate system and
    have the same dimensions.

    Args:
        precip_path_list (list): list of paths to monthly precipitation
            rasters. (model input)
        et0_path_list (list): path to monthly ET0 rasters. (model input)
        qf_m_path_list (list): path to monthly quickflow rasters calculated by
            Equation [1].
        flow_dir_mfd_path (str): path to a PyGeoprocessing Multiple Flow
            Direction raster indicating flow directions for this analysis.
        alpha_month_map (dict): fraction of upslope annual available recharge
            that is available in month m (indexed from 1).
        beta_i (float):  fraction of the upgradient subsidy that is available
            for downgradient evapotranspiration.
        gamma (float): the fraction of pixel recharge that is available to
            downgradient pixels.
        stream_path (str): path to the stream raster where 1 is a stream,
            0 is not, and nodata is outside of the DEM.
        kc_path_list (str): list of rasters of the monthly crop factor for the
            pixel.
        target_li_path (str): created by this call, path to local recharge
            derived from the annual water budget. (Equation 3).
        target_li_avail_path (str): created by this call, path to raster
            indicating available recharge to a pixel.
        target_l_sum_avail_path (str): created by this call, the recursive
            upslope accumulation of target_li_avail_path.
        target_aet_path (str): created by this call, the annual actual
            evapotranspiration.
        target_pi_path (str): created by this call, the annual precipitation on
            a pixel.

        Returns:
            None.

    """
    cdef int i_n, flow_dir_nodata, flow_dir_mfd
    cdef int peak_pixel
    cdef int xs, ys, xs_root, ys_root, xoff, yoff, flow_dir_s
    cdef int xi, yi, xj, yj, flow_dir_j, p_ij_base
    cdef int win_xsize, win_ysize, n_dir
    cdef int raster_x_size, raster_y_size
    cdef double pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i, l_avail_i
    cdef float qf_nodata, kc_nodata

    cdef int j_neighbor_end_index, mfd_dir_sum
    cdef float mfd_direction_array[8]

    cdef queue[pair[int, int]] work_queue
    cdef _ManagedRaster et0_m_raster, qf_m_raster, kc_m_raster

    cdef numpy.ndarray[numpy.npy_float32, ndim=1] alpha_month_array = (
        numpy.array(
            [x[1] for x in sorted(alpha_month_map.items())],
            dtype=numpy.float32))

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # we know the PyGeoprocessing MFD raster flow dir type is a 32 bit int.
    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_mfd_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']
    cdef _ManagedRaster flow_raster = _ManagedRaster(flow_dir_mfd_path, 1, 0)

    # make sure that user input nodata values are defined
    # set to -1 if not defined
    # precipitation and evapotranspiration data should
    # always be non-negative
    et0_m_raster_list = []
    et0_m_nodata_list = []
    for et0_path in et0_path_list:
        et0_m_raster_list.append(_ManagedRaster(et0_path, 1, 0))
        nodata = pygeoprocessing.get_raster_info(et0_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        et0_m_nodata_list.append(nodata)

    precip_m_raster_list = []
    precip_m_nodata_list = []
    for precip_m_path in precip_path_list:
        precip_m_raster_list.append(_ManagedRaster(precip_m_path, 1, 0))
        nodata = pygeoprocessing.get_raster_info(precip_m_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        precip_m_nodata_list.append(nodata)

    qf_m_raster_list = []
    qf_m_nodata_list = []
    for qf_m_path in qf_m_path_list:
        qf_m_raster_list.append(_ManagedRaster(qf_m_path, 1, 0))
        qf_m_nodata_list.append(
            pygeoprocessing.get_raster_info(qf_m_path)['nodata'][0])

    kc_m_raster_list = []
    kc_m_nodata_list = []
    for kc_m_path in kc_path_list:
        kc_m_raster_list.append(_ManagedRaster(kc_m_path, 1, 0))
        kc_m_nodata_list.append(
            pygeoprocessing.get_raster_info(kc_m_path)['nodata'][0])

    target_nodata = -1e32
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_li_raster = _ManagedRaster(
        target_li_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef _ManagedRaster target_li_avail_raster = _ManagedRaster(
        target_li_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_l_sum_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef _ManagedRaster target_l_sum_avail_raster = _ManagedRaster(
        target_l_sum_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_aet_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_aet_raster = _ManagedRaster(
        target_aet_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_pi_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_pi_raster = _ManagedRaster(
        target_pi_path, 1, 1)


    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info(
                'peak point detection %.2f%% complete',
                100.0 * current_pixel / <float>(
                    raster_x_size * raster_y_size))

        # search block for a peak pixel where no other pixel drains to it.
        for ys in xrange(win_ysize):
            ys_root = yoff+ys
            for xs in xrange(win_xsize):
                xs_root = xoff+xs
                flow_dir_s = <int>flow_raster.get(xs_root, ys_root)
                if flow_dir_s == flow_dir_nodata:
                    continue
                # search neighbors for downhill or nodata
                peak_pixel = 1
                for n_dir in xrange(8):
                    # searching around the pattern:
                    # 321
                    # 4x0
                    # 567
                    xj = xs_root+NEIGHBOR_OFFSET_ARRAY[2*n_dir]
                    yj = ys_root+NEIGHBOR_OFFSET_ARRAY[2*n_dir+1]
                    if (xj < 0 or xj >= raster_x_size or
                            yj < 0 or yj >= raster_y_size):
                        continue
                    flow_dir_j = <int>flow_raster.get(xj, yj)
                    if (0xF & (flow_dir_j >> (
                            4 * FLOW_DIR_REVERSE_DIRECTION[n_dir]))):
                        # pixel flows inward, not a peak
                        peak_pixel = 0
                        break
                if peak_pixel:
                    work_queue.push(
                        pair[int, int](xs_root, ys_root))

                while work_queue.size() > 0:
                    xi = work_queue.front().first
                    yi = work_queue.front().second
                    work_queue.pop()

                    l_sum_avail_i = target_l_sum_avail_raster.get(xi, yi)
                    if not is_close(l_sum_avail_i, target_nodata):
                        # already defined
                        continue

                    # Equation 7, calculate L_sum_avail_i if possible, skip
                    # otherwise
                    upslope_defined = 1
                    # initialize to 0 so we indicate we haven't tracked any
                    # mfd values yet
                    j_neighbor_end_index = 0
                    mfd_dir_sum = 0
                    for n_dir in xrange(8):
                        if not upslope_defined:
                            break
                        # searching around the pattern:
                        # 321
                        # 4x0
                        # 567
                        xj = xi+NEIGHBOR_OFFSET_ARRAY[2*n_dir]
                        yj = yi+NEIGHBOR_OFFSET_ARRAY[2*n_dir+1]
                        if (xj < 0 or xj >= raster_x_size or
                                yj < 0 or yj >= raster_y_size):
                            continue
                        p_ij_base = (<int>flow_raster.get(xj, yj) >> (
                                4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])) & 0xF
                        if p_ij_base:
                            mfd_dir_sum += p_ij_base
                            # pixel flows inward, check upslope
                            l_sum_avail_j = target_l_sum_avail_raster.get(
                                xj, yj)
                            if is_close(l_sum_avail_j, target_nodata):
                                upslope_defined = 0
                                break
                            l_avail_j = target_li_avail_raster.get(
                                xj, yj)
                            # A step of Equation 7
                            mfd_direction_array[j_neighbor_end_index] = (
                                l_sum_avail_j + l_avail_j) * p_ij_base
                            j_neighbor_end_index += 1
                    # calculate l_sum_avail_i by summing all the valid
                    # directions then normalizing by the sum of the mfd
                    # direction weights (Equation 8)
                    if upslope_defined:
                        l_sum_avail_i = 0.0
                        # Equation 7
                        if j_neighbor_end_index > 0:
                            # we can have no upslope, and then why would we
                            # divide?
                            for index in range(j_neighbor_end_index):
                                l_sum_avail_i += mfd_direction_array[index]
                            l_sum_avail_i /= <float>mfd_dir_sum
                        target_l_sum_avail_raster.set(xi, yi, l_sum_avail_i)
                    else:
                        # if not defined, we'll get it on another pass
                        continue

                    aet_i = 0
                    p_i = 0
                    qf_i = 0

                    for m_index in range(12):
                        precip_m_raster = (
                            <_ManagedRaster?>precip_m_raster_list[m_index])
                        qf_m_raster = (
                            <_ManagedRaster?>qf_m_raster_list[m_index])
                        et0_m_raster = (
                            <_ManagedRaster?>et0_m_raster_list[m_index])
                        kc_m_raster = (
                            <_ManagedRaster?>kc_m_raster_list[m_index])

                        et0_nodata = et0_m_nodata_list[m_index]
                        precip_nodata = precip_m_nodata_list[m_index]
                        qf_nodata = qf_m_nodata_list[m_index]
                        kc_nodata = kc_m_nodata_list[m_index]

                        p_m = precip_m_raster.get(xi, yi)
                        if not is_close(p_m, precip_nodata):
                            p_i += p_m
                        else:
                            p_m = 0

                        qf_m = qf_m_raster.get(xi, yi)
                        if not is_close(qf_m, qf_nodata):
                            qf_i += qf_m
                        else:
                            qf_m = 0

                        kc_m = kc_m_raster.get(xi, yi)
                        pet_m = 0
                        et0_m = et0_m_raster.get(xi, yi)
                        if not (
                                is_close(kc_m, kc_nodata) or
                                is_close(et0_m, et0_nodata)):
                            # Equation 6
                            pet_m = kc_m * et0_m

                        # Equation 4/5
                        aet_i += min(
                            pet_m,
                            p_m - qf_m +
                            alpha_month_array[m_index]*beta_i*l_sum_avail_i)

                    target_pi_raster.set(xi, yi, p_i)

                    target_aet_raster.set(xi, yi, aet_i)
                    l_i = (p_i - qf_i - aet_i)

                    # Equation 8
                    l_avail_i = min(gamma*l_i, l_i)

                    target_li_raster.set(xi, yi, l_i)
                    target_li_avail_raster.set(xi, yi, l_avail_i)

                    flow_dir_mfd = <int>flow_raster.get(xi, yi)
                    for i_n in range(8):
                        if ((flow_dir_mfd >> (i_n * 4)) & 0xF) == 0:
                            # no flow in that direction
                            continue
                        xi_n = xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue
                        work_queue.push(pair[int, int](xi_n, yi_n))


def route_baseflow_sum(
        flow_dir_mfd_path, l_path, l_avail_path, l_sum_path,
        stream_path, target_b_path, target_b_sum_path):
    """Route Baseflow through MFD as described in Equation 11.

    Args:
        flow_dir_mfd_path (string): path to a pygeoprocessing multiple flow
            direction raster.
        l_path (string): path to local recharge raster.
        l_avail_path (string): path to local recharge raster that shows
            recharge available to the pixel.
        l_sum_path (string): path to upslope sum of l_path.
        stream_path (string): path to stream raster, 1 stream, 0 no stream,
            and nodata.
        target_b_path (string): path to created raster for per-pixel baseflow.
        target_b_sum_path (string): path to created raster for per-pixel
            upslope sum of baseflow.

    Returns:
        None.
    """

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    cdef float target_nodata = -1e32
    cdef int stream_val, outlet
    cdef float b_i, b_sum_i, l_j, l_avail_j, l_sum_j
    cdef int xi, yi, xj, yj, flow_dir_i, p_ij_base
    cdef int mfd_dir_sum, flow_dir_nodata
    cdef int raster_x_size, raster_y_size, xs_root, ys_root, xoff, yoff
    cdef int n_dir
    cdef int xs, ys, flow_dir_s, win_xsize, win_ysize
    cdef int stream_nodata
    cdef stack[pair[int, int]] work_stack

    # we know the PyGeoprocessing MFD raster flow dir type is a 32 bit int.
    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_mfd_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']

    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_sum_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])

    cdef _ManagedRaster target_b_sum_raster = _ManagedRaster(
        target_b_sum_path, 1, 1)
    cdef _ManagedRaster target_b_raster = _ManagedRaster(
        target_b_path, 1, 1)
    cdef _ManagedRaster l_raster = _ManagedRaster(l_path, 1, 0)
    cdef _ManagedRaster l_avail_raster = _ManagedRaster(l_avail_path, 1, 0)
    cdef _ManagedRaster l_sum_raster = _ManagedRaster(l_sum_path, 1, 0)
    cdef _ManagedRaster flow_dir_mfd_raster = _ManagedRaster(
        flow_dir_mfd_path, 1, 0)

    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, 0)

    current_pixel = 0
    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # search block for a peak pixel where no other pixel drains to it.
        for ys in xrange(win_ysize):
            ys_root = yoff+ys
            for xs in xrange(win_xsize):
                xs_root = xoff+xs
                flow_dir_s = <int>flow_dir_mfd_raster.get(xs_root, ys_root)
                if flow_dir_s == flow_dir_nodata:
                    current_pixel += 1
                    continue
                outlet = 1
                for n_dir in xrange(8):
                    if (flow_dir_s >> (n_dir * 4)) & 0xF:
                        # flows in this direction
                        xj = xs_root+NEIGHBOR_OFFSET_ARRAY[2*n_dir]
                        yj = ys_root+NEIGHBOR_OFFSET_ARRAY[2*n_dir+1]
                        if (xj < 0 or xj >= raster_x_size or
                                yj < 0 or yj >= raster_y_size):
                            continue
                        stream_val = <int>stream_raster.get(xj, yj)
                        if stream_val != stream_nodata:
                            outlet = 0
                            break
                if not outlet:
                    continue
                work_stack.push(pair[int, int](xs_root, ys_root))

                while work_stack.size() > 0:
                    xi = work_stack.top().first
                    yi = work_stack.top().second
                    work_stack.pop()
                    b_sum_i = target_b_sum_raster.get(xi, yi)
                    if not is_close(b_sum_i, target_nodata):
                        continue

                    if ctime(NULL) - last_log_time > 5.0:
                        last_log_time = ctime(NULL)
                        LOGGER.info(
                            'route base flow %.2f%% complete',
                            100.0 * current_pixel / <float>(
                                raster_x_size * raster_y_size))

                    b_sum_i = 0.0
                    mfd_dir_sum = 0
                    downslope_defined = 1
                    flow_dir_i = <int>flow_dir_mfd_raster.get(xi, yi)
                    if flow_dir_i == flow_dir_nodata:
                        LOGGER.error("flow dir nodata? this makes no sense")
                        continue
                    for n_dir in xrange(8):
                        if not downslope_defined:
                            break
                        # searching around the pattern:
                        # 321
                        # 4x0
                        # 567
                        p_ij_base = (flow_dir_i >> (4*n_dir)) & 0xF
                        if p_ij_base:
                            mfd_dir_sum += p_ij_base
                            xj = xi+NEIGHBOR_OFFSET_ARRAY[2*n_dir]
                            yj = yi+NEIGHBOR_OFFSET_ARRAY[2*n_dir+1]
                            if (xj < 0 or xj >= raster_x_size or
                                    yj < 0 or yj >= raster_y_size):
                                continue
                            stream_val = <int>stream_raster.get(xj, yj)

                            if stream_val:
                                b_sum_i += p_ij_base
                            else:
                                b_sum_j = target_b_sum_raster.get(xj, yj)
                                if is_close(b_sum_j, target_nodata):
                                    downslope_defined = 0
                                    break
                                l_j = l_raster.get(xj, yj)
                                l_avail_j = l_avail_raster.get(xj, yj)
                                l_sum_j = l_sum_raster.get(xj, yj)

                                if l_sum_j != 0 and (l_sum_j - l_j) != 0:
                                    b_sum_i += p_ij_base * (
                                        (1-l_avail_j / l_sum_j)*(
                                            b_sum_j / (l_sum_j - l_j)))
                                else:
                                    b_sum_i += p_ij_base

                    if not downslope_defined:
                        continue
                    l_sum_i = l_sum_raster.get(xi, yi)
                    if mfd_dir_sum > 0:
                        # normalize by mfd weight
                        b_sum_i = l_sum_i * b_sum_i / <float>mfd_dir_sum
                    target_b_sum_raster.set(xi, yi, b_sum_i)
                    l_i = l_raster.get(xi, yi)
                    if l_sum_i != 0:
                        b_i = max(b_sum_i * l_i / l_sum_i, 0.0)
                    else:
                        b_i = 0.0
                    target_b_raster.set(xi, yi, b_i)
                    current_pixel += 1

                    for n_dir in xrange(8):
                        # searching upslope for pixels that flow in
                        # 321
                        # 4x0
                        # 567
                        xj = xi+NEIGHBOR_OFFSET_ARRAY[2*n_dir]
                        yj = yi+NEIGHBOR_OFFSET_ARRAY[2*n_dir+1]
                        if (xj < 0 or xj >= raster_x_size or
                                yj < 0 or yj >= raster_y_size):
                            continue
                        flow_dir_j = <int>flow_dir_mfd_raster.get(xj, yj)
                        if (0xF & (flow_dir_j >> (
                                4 * FLOW_DIR_REVERSE_DIRECTION[n_dir]))):
                            # pixel flows here, push on queue
                            work_stack.push(pair[int, int](xj, yj))
