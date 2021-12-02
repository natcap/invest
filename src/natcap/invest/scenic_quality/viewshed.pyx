# coding=UTF-8
# cython: language_level=2
"""
Implements the Wang et al (2000) viewshed based on reference planes.

This algorithm was originally described in "Generating viewsheds without using
sightlines", authored by Jianjun Wang, Gary J. Robertson, and Kevin White,
published in Photogrammetric Engineering & Remote Sensing, Vol. 66, No. 1,
January 2000, pp. 87-90.

Calculations for adjusting the required height for curvature of the earth have
been adapted from the ESRI ArcGIS documentation:
http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/using-viewshed-and-observer-points-for-visibility.htm

Consistent with the routing functionality of pygeoprocessing, neighbor
directions follow the right-hand rule where neighbor indexes are interpreted
as:

    # 321
    # 4X0
    # 567
"""
import time
import os
import logging
import shutil
import tempfile

import numpy
import pygeoprocessing
from osgeo import gdal
from osgeo import osr
import shapely.geometry
from .. import utils
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time_t
from libc.time cimport time as ctime
from libcpp.list cimport list as clist
from libcpp.set cimport set as cset
from libcpp.deque cimport deque
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libc cimport math
cimport numpy
cimport cython


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
BYTE_GTIFF_CREATION_OPTIONS = (
    'GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
              'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
FLOAT_GTIFF_CREATION_OPTIONS = (
    'GTIFF', ('PREDICTOR=3',) + BYTE_GTIFF_CREATION_OPTIONS[1])

# Indexes for neighbors relative to the target pixel.
# Indexes in this array are stored in numpy order (row, col).
cdef int* NEIGHBORS_INDEXES = [
    0, 1,  # 0
    -1, 1,  # 1
    -1, 0,  # 2
    -1, -1,  # 3
    0, -1,  # 4
    1, -1,  # 5
    1, 0,  # 6
    1, 1  # 7
]

# The Wang et al algorithm defines two neighbors for the target viewpoint from
# which we construct the reference plane.  In this implementation, Neighbor 1
# is always the neighbor that is closest (in terms of euclidean distance) to
# the target.  Neighbor 2 is at a diagonal to the target.  The indexes of
# Neighbor 1 and Neighbor 2 relative to the target pixel vary by sector.
# This array maps the sector to the index of Neighbor 1, relative to the
# target.
cdef int* SECTOR_TO_NEIGHBOR_1_INDEX = [
    0, -1,
    1, 0,
    1, 0,
    0, 1,
    0, 1,
    -1, 0,
    -1, 0,
    0, -1
]

cdef int* SECTOR_TO_NEIGHBOR_2_INDEX = [
    1, -1,
    1, -1,
    1, 1,
    1, 1,
    -1, 1,
    -1, 1,
    -1, -1,
    -1, -1
]

# List out which targets (in terms of their neighbor index) to process per
# sector.  We can preemptively define which neighbors are to be visited after a
# given target by indexing into the arrays we've already created.  Each line in
# this array has four indexes.  The first two are for Neighbor1, the second two
# are for Neighbor2.
#
# Order of these is:
#    iy_next_neighbor1, ix_next_neighbor1, iy_next_neighbor1, ix_next_neighbor2
#
# When looking at the indexes, the number multiplied by 2 is the neighbor
# index.  So, sector 0 uses neighbors 0 and 1.  Sector 7 uses neighbors 7 and
# 0.
cdef int* SECTOR_NEXT_TARGET_INDEXES = [
    NEIGHBORS_INDEXES[2*0], NEIGHBORS_INDEXES[2*0+1], NEIGHBORS_INDEXES[2*1], NEIGHBORS_INDEXES[2*1+1],
    NEIGHBORS_INDEXES[2*1], NEIGHBORS_INDEXES[2*1+1], NEIGHBORS_INDEXES[2*2], NEIGHBORS_INDEXES[2*2+1],
    NEIGHBORS_INDEXES[2*2], NEIGHBORS_INDEXES[2*2+1], NEIGHBORS_INDEXES[2*3], NEIGHBORS_INDEXES[2*3+1],
    NEIGHBORS_INDEXES[2*3], NEIGHBORS_INDEXES[2*3+1], NEIGHBORS_INDEXES[2*4], NEIGHBORS_INDEXES[2*4+1],
    NEIGHBORS_INDEXES[2*4], NEIGHBORS_INDEXES[2*4+1], NEIGHBORS_INDEXES[2*5], NEIGHBORS_INDEXES[2*5+1],
    NEIGHBORS_INDEXES[2*5], NEIGHBORS_INDEXES[2*5+1], NEIGHBORS_INDEXES[2*6], NEIGHBORS_INDEXES[2*6+1],
    NEIGHBORS_INDEXES[2*6], NEIGHBORS_INDEXES[2*6+1], NEIGHBORS_INDEXES[2*7], NEIGHBORS_INDEXES[2*7+1],
    NEIGHBORS_INDEXES[2*7], NEIGHBORS_INDEXES[2*7+1], NEIGHBORS_INDEXES[2*0], NEIGHBORS_INDEXES[2*0+1]
]


# The processing queue must be seeded with a pixel to process.  From there, the
# processing loop will discover more neighbors to inspect until it runs out of
# neighbors in the sector.  These seed indexes are relative to the viewpoint
# index.
cdef int* SECTOR_SEEDS = [
    -1, 2,  # seed index for sector 0
    -2, 1,  # seed index for sector 1
    -2, -1,  # seed index for sector 2
    -1, -2,  # seed index for sector 3
    1, -2,  # seed index for sector 4
    2, -1,  # seed index for sector 5
    2, 1,  # seed index for sector 6
    1, 2,  # seed index for sector 7
]
cdef int AUX_NOT_VISITED = -9999
cdef int DIAM_EARTH = 12740000  # meters, from ArcGIS docs
cdef double DIAM_EARTH_INV = 1.0/DIAM_EARTH
cdef double IMPROBABLE_NODATA = -123457.12345

# This type represents an x,y coordinate pair.
ctypedef pair[long, long] CoordinatePair

# Defining a couple of operators for longs so I can minimize code to be
# executed based on `cython -a`.
cdef inline long labs(long a):
    if a > 0:
        return a
    return a*-1

cdef inline long lmax(long a, long b):
    if a > b:
        return a
    return b

cdef inline long lmin(long a, long b):
    if a < b:
        return a
    return b


# this is a least recently used cache written in C++ in an external file,
# exposing here so _ManagedRaster can use it.
# Copied from src/pygeoprocessing/routing/routing.pyx,
# revision 891288683889237cfd3a3d0a1f09483c23489fca.
cdef extern from "LRUCache.h":
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)


# exposing stl::priority_queue so we can have all 3 template arguments so
# we can pass a different Compare functor
cdef extern from "<queue>" namespace "std":
    cdef cppclass priority_queue[T, Container, Compare]:
        priority_queue() except +
        priority_queue(priority_queue&) except +
        priority_queue(Container&)
        bint empty()
        void pop()
        void push(T&)
        size_t size()
        T& top()


# functor for priority queue of pixels
cdef cppclass BlockwiseCloserTarget:
    bint get "operator()"(TargetPixel& lhs, TargetPixel& rhs):
        # Go with the lower ring ID if possible.
        if lhs.ring_id < rhs.ring_id:
            return 0

        # If we're in the same ring, go with the lowest sector
        # If we're in the same sector, go with the shortest distance.
        if lhs.ring_id == rhs.ring_id:
            if lhs.sector < rhs.sector:
                return 0
            if lhs.sector == rhs.sector:
                if lhs.distance_to_viewpoint < rhs.distance_to_viewpoint:
                    return 0
        return 1


# A priority queue of TargetPixels should should first minimize the block distance.
# If the block_distance is the same, the queue should minimize the target distance.
cdef struct TargetPixel:
    long ix
    long iy
    int ring_id
    int sector
    double distance_to_viewpoint # distance between viewpoint and target.


ctypedef priority_queue[
    TargetPixel, deque[TargetPixel], BlockwiseCloserTarget] TargetPixelPriorityQueue

# A function for wrapping up a call to get the length of the hypotenuse between two pixels.
cdef inline double pixel_dist(long ix_source, long ix_target, long iy_source, long iy_target):
        return math.hypot(
            lmax(ix_source, ix_target)-lmin(ix_source, ix_target),
            lmax(iy_source, iy_target)-lmin(iy_source, iy_target))


# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# The nodata value for visibility rasters
cdef int VISIBILITY_NODATA = 255

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


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def viewshed(dem_raster_path_band,
             viewpoint,
             visibility_filepath,
             viewpoint_height=0.0,
             curved_earth=True,
             refraction_coeff=0.13,
             max_distance=None,
             aux_filepath=None):
    """Compute the Wang et al. reference-plane based viewshed.

    Args:
        dem_raster_path_band (tuple): A tuple of (path, band_index) where
            ``path`` is a path to a GDAL-compatible raster on disk and
            ``band_index`` is the 1-based band index.  This DEM must be tiled
            with block sizes as a power of 2.  If the viewshed is being
            adjusted for curvature of the earth and/or refraction, the
            elevation units of the DEM must be in meters.  The DEM need not be
            projected in meters.
        viewpoint (tuple):  A tuple of 2 numbers in the order
            ``(east offset, north offset)``.  These units must be of the same
            units as the coordinate system of the DEM.  This index represents
            the viewpoint location.  The closest pixel to this viewpoint index
            will be used as the viewpoint.
        visibility_filepath (string): A filepath on disk to where the
            visibility raster will be written. If a raster exists in this
            location, it will be overwritten.
        viewpoint_height=0.0 (float):  The height (in the units of the DEM
            height) of the observer at the viewpoint.
        curved_earth=True (bool): Whether to adjust viewshed calculations for
            the curvature of the earth.  If False, the earth will be treated as
            though it is flat.
        refraction_coeff=0.13 (float):  The coefficient of atmospheric
            refraction that may be adjusted to accommodate varying atmospheric
            conditions.  Default is ``0.13``.  Set to ``0`` to ignore
            refraction calculations.
        max_distance=None (float):  If provided, visibility will not be
            calculated for DEM pixels that are more than this distance (in meters)
            from the viewpoint.
        aux_filepath=None (string): A path to a location on disk
            where the raster containing the auxiliary matrix will be written.
            The auxiliary matrix defines the height that a DEM must exceed in
            order to be visible from the viewpoint.  This matrix is very useful
            for debugging.  If a raster already exists at this location, it
            will be overwritten.  If this path is not provided by the user, the
            viewshed will create this file as a temporary file wherever the
            system keeps its temp files and remove it when the viewshed
            finishes.  See python's ``tempfile`` documentation for where this
            might be on your system.

    Raises:
        ValueError: When either the viewpoint does not overlap with the DEM or
            the DEM is not tiled appropriately.

        LookupError: When the ``viewpoint`` coordinate pair is over nodata.

        AssertionError: When pixel dimensions are not square.

    Returns:
        ``None``
    """
    start_time = time.time()

    # Check the bounding box to make sure that the viewpoint overlaps the DEM.
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    dem_gt = dem_raster_info['geotransform']
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = dem_raster_info['bounding_box']
    if (not bbox_minx <= viewpoint[0] <= bbox_maxx or
            not bbox_miny <= viewpoint[1] <= bbox_maxy):
        raise ValueError(('Viewpoint (%s, %s) does not overlap with DEM with '
                          'bounding box %s') % (viewpoint[0], viewpoint[1],
                                                dem_raster_info['bounding_box']))
    cdef long iy_viewpoint = int((viewpoint[1] - dem_gt[3]) / dem_gt[5])
    cdef long ix_viewpoint = int((viewpoint[0] - dem_gt[0]) / dem_gt[1])

    # Get the elevation of the pixel under the viewpoint and check to see if
    # it's nodata.
    raster = gdal.OpenEx(dem_raster_path_band[0])
    band = raster.GetRasterBand(dem_raster_path_band[1])
    viewpoint_elevation = band.ReadAsArray(ix_viewpoint, iy_viewpoint, 1, 1)
    band = None
    raster = None

    band_to_array_index = dem_raster_path_band[1] - 1
    nodata_value = dem_raster_info['nodata'][band_to_array_index]
    if viewpoint_elevation == nodata_value:
        raise LookupError('Viewpoint is over nodata')

    # Need to handle the case where the nodata value is not defined.
    if nodata_value is None:
        nodata_value = IMPROBABLE_NODATA
    cdef double nodata = nodata_value

    # Verify that pixels are very close to square.  The Wang et al algorithm
    # doesn't require that this be the case, but the math is simplified if it
    # is.
    pixel_xsize, pixel_ysize = dem_raster_info['pixel_size']
    if not (abs(abs(pixel_xsize) - abs(pixel_ysize)) < 0.5e-7):
        raise AssertionError(
            'Pixel dimensions must match:\n X size:%s\n Y size:%s' %
                             (pixel_xsize, pixel_ysize))

    # Verify that the block sizes are powers of 2 and are square.
    # This is needed for the _ManagedRaster classes.  If this is not asserted
    # here, the _ManagedRaster classes will crash with a segfault.
    block_xsize, block_ysize = dem_raster_info['block_size']
    if (block_xsize & (block_xsize - 1) != 0 or (
            block_ysize & (block_ysize - 1) != 0)) or (
                block_xsize != block_ysize):
        raise ValueError(
            'DEM must be tiled and tiles must be of equal sizes that are a '
            'power of 2.  Current block size is (%s, %s)' %
            (block_xsize, block_ysize))

    # Create the auxiliary raster for storing the calculated minimum height
    # for visibility at a given point.
    temp_dir = None
    if aux_filepath is None:
        temp_dir = tempfile.mkdtemp(
            prefix='viewshed_%s' % time.strftime(
                '%Y-%m-%d_%H_%M_%S', time.gmtime()))
        aux_filepath = os.path.join(temp_dir, 'auxiliary.tif')

    LOGGER.info("Creating auxiliary raster %s", aux_filepath)
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], aux_filepath, gdal.GDT_Float64, [AUX_NOT_VISITED],
        fill_value_list=[AUX_NOT_VISITED],
        raster_driver_creation_tuple=FLOAT_GTIFF_CREATION_OPTIONS)

    # Create the visibility raster for indicating whether a pixel is visible
    # based on the calculated minimum height.
    LOGGER.info('Creating visibility raster %s', visibility_filepath)
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], visibility_filepath, gdal.GDT_Byte,
        [VISIBILITY_NODATA], fill_value_list=[VISIBILITY_NODATA],
        raster_driver_creation_tuple=BYTE_GTIFF_CREATION_OPTIONS)

    # LRU-cached rasters for easier access to individual pixels.
    cdef _ManagedRaster dem_managed_raster = (
            _ManagedRaster(dem_raster_path_band[0], dem_raster_path_band[1], 0))
    cdef _ManagedRaster aux_managed_raster = (
            _ManagedRaster(aux_filepath, 1, 1))
    cdef _ManagedRaster visibility_managed_raster = (
            _ManagedRaster(visibility_filepath, 1, 1))

    # get the pixel size in terms of meters.
    dem_srs = osr.SpatialReference()
    dem_srs.ImportFromWkt(dem_raster_info['projection_wkt'])
    linear_units = dem_srs.GetLinearUnits()
    cdef double pixel_size = utils.mean_pixel_size_and_area(
        dem_raster_info['pixel_size'])[0]*linear_units
    cdef long raster_x_size = dem_raster_info['raster_size'][0]
    cdef long raster_y_size = dem_raster_info['raster_size'][1]
    cdef double max_visible_radius
    cdef long pixels_in_raster
    if max_distance is not None:
        # This is an estimate of the number of pixels in the raster when
        # viewshed is given a max_distance radius.  The estimate is based on
        # the area of the DEM bounding box that intersects with the circle
        # indicating the max_distance radius around the viewpoint.
        max_visible_radius = max_distance

        # prefer to overestimate.
        point = shapely.geometry.Point(viewpoint).buffer(max_distance+pixel_size)
        bbox_rectangle = shapely.geometry.box(*dem_raster_info['bounding_box'])
        pixels_in_raster = <long> int(
            point.intersection(bbox_rectangle).area/(pixel_size**2))
        LOGGER.debug('Approx. %s pixels in raster within max distance %s',
                     pixels_in_raster, max_distance)
    else:
        # max visible distance is the whole raster, which should be the hypotenuse
        # between two adjoining edges of the bounding box.
        max_visible_radius = math.hypot(raster_x_size, raster_y_size)*pixel_size
        pixels_in_raster = raster_x_size * raster_y_size

    cdef long m, n, xi, yi
    cdef int correct_for_curvature = curved_earth
    cdef int correct_for_refraction = math.fabs(math.ceil(refraction_coeff) - 1.0) < 0.5e-7
    cdef double target_height_adjustment = 0  # initializing for compiler
    cdef double adjustment = 0.0
    cdef float refract_coeff = refraction_coeff  # from the user
    cdef int block_x_size = dem_raster_info['block_size'][0]
    cdef int block_y_size = dem_raster_info['block_size'][1]
    cdef int block_bits = numpy.log2(block_x_size)  # for bit-shifting
    cdef long ix_viewpoint_block = ix_viewpoint >> block_x_size
    cdef long iy_viewpoint_block = iy_viewpoint >> block_y_size
    cdef long pixels_touched = 0

    # Following the Wang et al. terminology, it's helpful to think of blocks in
    # terms of rings around the block containing the viewpoint.  Ring 0 is the
    # 'ring' containing the viewpoint.  A block in ring 1 is one of the 8
    # blocks immediately adjacent to the viewpoint's block.
    cdef int ring_id

    LOGGER.info("Starting viewshed for viewpoint %s on DEM %s",
                viewpoint, dem_raster_path_band[0])

    # As defined by Wang et al, the viewpoint and the immediate neighbors are
    # all assumed to be visible.
    for yi in xrange(iy_viewpoint-1, iy_viewpoint+2):
        # Bounds check on the row we're initializing
        if not 0 <= yi < raster_y_size:
            continue

        for xi in xrange(ix_viewpoint-1, ix_viewpoint+2):
            # Bounds check on the column we're initializing.
            if not 0 <= xi < raster_x_size:
                continue

            aux_managed_raster.set(
                xi, yi, dem_managed_raster.get(xi, yi))
            visibility_managed_raster.set(xi, yi, 1)
            pixels_touched += 1

    # Save the viewpoint index for later.  These are the variable names used in
    # the reference-plane equation in Wang et al.
    # i is the row, j is the column.
    cdef long i = iy_viewpoint
    cdef long j = ix_viewpoint

    # If the user defined a viewpoint, we add it to the actual viewpoint height
    # in the DEM matrix.
    cdef double r_v = dem_managed_raster.get(ix_viewpoint, iy_viewpoint) + viewpoint_height

    # Defining cardinal and intercardinal directions is significantly simpler
    # than defining other pixels because the reference plane is constructed
    # from only 1 previous pixel (as opposed to 2 for the other pixels).
    # Rather than try to shoehorn these calculations into a single processing
    # loop, we can take care of this special case ahead of time.
    cdef long ix_target, iy_target
    cdef long ix_prev_target, iy_prev_target
    cdef long ix_cardinal_target, iy_cardinal_target
    cdef double target_dem_height, adjusted_dem_height
    cdef double target_distance
    cdef double slope_distance
    cdef double z = 0  # initializing for compiler
    cdef int multiplier
    cdef double sqrt2 = math.sqrt(2)
    cdef int i_n
    cdef time_t last_log_time = ctime(NULL)
    for i_n in xrange(8):
        ix_cardinal_target = NEIGHBORS_INDEXES[2*i_n]
        iy_cardinal_target = NEIGHBORS_INDEXES[2*i_n+1]
        multiplier = 2
        while True:
            iy_target = iy_viewpoint+iy_cardinal_target*multiplier
            if not 0 <= iy_target < raster_y_size:
                break

            ix_target = ix_viewpoint+ix_cardinal_target*multiplier
            if not 0 <= ix_target < raster_x_size:
                break

            ix_prev_target = ix_viewpoint+ix_cardinal_target*(multiplier-1)
            iy_prev_target = iy_viewpoint+iy_cardinal_target*(multiplier-1)
            previous_height = aux_managed_raster.get(ix_prev_target,
                                                     iy_prev_target)

            if lmax(ix_cardinal_target, iy_cardinal_target) == 0:
                slope_distance = labs(
                    lmin(ix_cardinal_target, iy_cardinal_target)*(multiplier-1))
                target_distance = labs(
                    lmin(ix_cardinal_target, iy_cardinal_target)*(multiplier))
            else:
                slope_distance = labs(
                    lmax(ix_cardinal_target, iy_cardinal_target)*(multiplier-1))
                target_distance = labs(
                    lmax(ix_cardinal_target, iy_cardinal_target)*(multiplier))

            # If we're on a diagonal, multiply by sqrt(2) to adjust the pixel
            # distance for the diagonal.
            if ix_cardinal_target != 0 and iy_cardinal_target != 0:
                slope_distance *= sqrt2
                target_distance *= sqrt2

            # adjust for pixel_size
            target_distance *= pixel_size
            slope_distance *= pixel_size

            if target_distance > max_visible_radius:
                break

            z = (((previous_height-r_v)/slope_distance) *
                 target_distance + r_v)

            # add on refractivity/curvature-of-earth calculations.
            adjustment = 0.0  # increase in required height due to curvature
            if correct_for_curvature or correct_for_refraction:
                # target_height_adjustment is the apparent height reduction of
                # the target due to the curvature of the earth.
                target_height_adjustment = (target_distance**2)*DIAM_EARTH_INV
                if correct_for_curvature:
                    adjustment += target_height_adjustment
                if correct_for_refraction:
                    adjustment -= refract_coeff*target_height_adjustment

            target_dem_height = dem_managed_raster.get(ix_target, iy_target)
            adjusted_dem_height = target_dem_height - adjustment
            if (adjusted_dem_height >= z and
                    target_distance < max_visible_radius and
                    target_dem_height != nodata):
                visibility_managed_raster.set(ix_target, iy_target, 1)
                aux_managed_raster.set(ix_target, iy_target, adjusted_dem_height)
            else:
                visibility_managed_raster.set(ix_target, iy_target, 0)
                aux_managed_raster.set(ix_target, iy_target, z)

            multiplier += 1
            pixels_touched += 1

    # Process subsequent land points.
    # The queue keeps track of the order, the set tracks what's in the queue.
    # The set allows us to do constant-time checking of what's in the queue so
    # we don't add more work than we have to.
    cdef TargetPixelPriorityQueue process_queue
    cdef cset[CoordinatePair] process_queue_set

    cdef int next_target_idx
    cdef long ix_seed, iy_seed
    cdef int sector
    cdef long ix_next_target, iy_next_target
    cdef CoordinatePair next_target_index
    cdef TargetPixel target_pixel
    cdef int pixels_touched_at_last_log = 0
    cdef int pixels_processed_since_last_log

    # Seed the sector with a pixel in the same direction as the sector.
    # We can only seed the pixel if it's within the raster.
    for sector in xrange(0, 8):
        iy_seed = iy_viewpoint + SECTOR_SEEDS[2*sector]
        if not 0 <= iy_seed < raster_y_size:
            continue

        ix_seed = ix_viewpoint + SECTOR_SEEDS[2*sector+1]
        if not 0 <= ix_seed < raster_x_size:
            continue

        target_distance = pixel_dist(ix_viewpoint, ix_seed,
                                     iy_viewpoint, iy_seed)*pixel_size
        ring_id = lmax(labs(ix_viewpoint_block - ix_seed>>block_bits),
                       labs(iy_viewpoint_block - iy_seed>>block_bits))

        process_queue.push(TargetPixel(
            ix_seed, iy_seed, ring_id, sector, target_distance))
        process_queue_set.insert(CoordinatePair(ix_seed, iy_seed))

    while not process_queue.empty():
        if ctime(NULL) - last_log_time > 5.0:
            pixels_processed_since_last_log = pixels_touched - pixels_touched_at_last_log
            time_since_last_log = ctime(NULL) - last_log_time
            last_log_time = ctime(NULL)
            LOGGER.info(
                ('Viewshed approx. %.2f%% complete. Remaining pixels: %i '
                 '(%6.2f pixels/sec)'),
                100.0*pixels_touched/<double>pixels_in_raster,
                pixels_in_raster-pixels_touched,
                pixels_processed_since_last_log/<double>time_since_last_log
            )
            pixels_touched_at_last_log = pixels_touched

        target_pixel = process_queue.top()
        process_queue_set.erase(
            CoordinatePair(target_pixel.ix, target_pixel.iy))
        process_queue.pop()

        # We have a target, determine visibility
        m = target_pixel.iy  # y index (row)
        n = target_pixel.ix   # x index (col)
        r_n1 = aux_managed_raster.get(
            n + SECTOR_TO_NEIGHBOR_1_INDEX[2*target_pixel.sector+1],
            m + SECTOR_TO_NEIGHBOR_1_INDEX[2*target_pixel.sector])
        r_n2 = aux_managed_raster.get(
            n + SECTOR_TO_NEIGHBOR_2_INDEX[2*target_pixel.sector+1],
            m + SECTOR_TO_NEIGHBOR_2_INDEX[2*target_pixel.sector])

        # These equations are taken directly from the Wang et al. paper.
        # Sector 3 is the sector that is explicitly referenced in the paper.
        # The others are adjusted based on the neighbors selected.
        # I could abstract this in such a way that the math is only written out
        # on one line, but that ends up being much longer than just writing out
        # the math here for each sector.
        #
        # Variable names in the math has been maintained where possible.
        # r_n1 is the equivalent of r_m,n+1 in the paper.
        # r_n2 is the equivalent of r+m+1,n+1 in the paper.
        #
        # The only modification I have made to the math is working in the
        # viewpoint height, r_v, which allows us to have a slope that is
        # positive or negative, and is relative to the viewpoint height.  This
        # modification is not in the paper, and is not in any citable resource
        # I have found.
        if target_pixel.sector == 0:
            z = -(m-i)*(r_n1-r_n2)+(j-n)*((m-i)*(r_n1-r_n2)-r_v+r_n1)/(j+1-n)+r_v
        elif target_pixel.sector == 1:
            z = -(j-n)*(r_n1-r_n2)+(m-i)*((j-n)*(r_n1-r_n2)-r_v+r_n1)/(m+1-i)+r_v
        elif target_pixel.sector == 2:
            z = -(n-j)*(r_n1-r_n2)+(m-i)*((n-j)*(r_n1-r_n2)-r_v+r_n1)/(m+1-i)+r_v
        elif target_pixel.sector == 3:
            z = -(m-i)*(r_n1-r_n2)+(n-j)*((m-i)*(r_n1-r_n2)-r_v+r_n1)/(n+1-j)+r_v
        elif target_pixel.sector == 4:
            z = -(i-m)*(r_n1-r_n2)+(n-j)*((i-m)*(r_n1-r_n2)-r_v+r_n1)/(n+1-j)+r_v
        elif target_pixel.sector == 5:
            z = -(n-j)*(r_n1-r_n2)+(i-m)*((n-j)*(r_n1-r_n2)-r_v+r_n1)/(i+1-m)+r_v
        elif target_pixel.sector == 6:
            z = -(j-n)*(r_n1-r_n2)+(i-m)*((j-n)*(r_n1-r_n2)-r_v+r_n1)/(i+1-m)+r_v
        elif target_pixel.sector == 7:
            z = -(i-m)*(r_n1-r_n2)+(j-n)*((i-m)*(r_n1-r_n2)-r_v+r_n1)/(j+1-n)+r_v

        # Refraction and curvature calculations are not in the Wang et al
        # paper.  I adapted these from the ESRI documentation of their
        # viewshed, which can be found at
        # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/using-viewshed-and-observer-points-for-visibility.htm
        adjustment = 0.0  # increase in required height due to curvature
        if correct_for_curvature or correct_for_refraction:
            # target_height_adjustment is the apparent height reduction of
            # the target due to the curvature of the earth.
            target_height_adjustment = (target_pixel.distance_to_viewpoint**2)*DIAM_EARTH_INV
            if correct_for_curvature:
                adjustment += target_height_adjustment
            if correct_for_refraction:
                adjustment -= refract_coeff*target_height_adjustment

        # Given the reference plane and any adjustments to the minimum required
        # height for visibility, the DEM pixel is only visible if it is greater
        # than or equal to the minimum-visible height AND is closer than the
        # maximum visible radius.
        target_dem_height = dem_managed_raster.get(n, m)
        adjusted_dem_height = dem_managed_raster.get(n, m) - adjustment
        if (adjusted_dem_height >= z and
                target_pixel.distance_to_viewpoint < max_visible_radius and
                target_dem_height != nodata):
            visibility_managed_raster.set(n, m, 1)
            aux_managed_raster.set(n, m, adjusted_dem_height)
        else:
            # If it's close enough to nodata to be interpreted as nodata,
            # consider it to be nodata.  Nodata implies that visibility is
            # undefined ... which it is, since there's no defined DEM value for
            # this pixel.
            if math.fabs(target_dem_height - nodata) <= 1.0e-7:
                visibility_managed_raster.set(n, m, VISIBILITY_NODATA)
            else:
                # If we're not over nodata, then the pixel isn't visible.
                visibility_managed_raster.set(n, m, 0)
            aux_managed_raster.set(n, m, z)
        pixels_touched += 1

        # Having determined the visibility for the target_pixel, we now need to
        # enqueue those pixels that depend on this pixel.  This is determined
        # by the sector that target_pixel is in.  Every target has two possible
        # neighbors that may be enqueued, but there are a variety of reasons
        # why a neighbor might not be a valid target.
        for next_target_idx in xrange(target_pixel.sector*4,
                                      target_pixel.sector*4+4, 2):
            iy_next_target = m + SECTOR_NEXT_TARGET_INDEXES[next_target_idx]
            ix_next_target = n + SECTOR_NEXT_TARGET_INDEXES[next_target_idx+1]

            # Skip the target if it's off the bounds of the raster.
            if not 0 <= iy_next_target < raster_y_size:
                continue
            if not 0 <= ix_next_target < raster_x_size:
                continue

            # Skip the target if it's already in the queue.
            next_target_index = CoordinatePair(ix_next_target, iy_next_target)
            if (process_queue_set.find(next_target_index)
                    != process_queue_set.end()):
                continue

            # Skip the target if it's too far away from the viewpoint.
            target_distance = pixel_dist(ix_next_target, ix_viewpoint,
                                         iy_next_target, iy_viewpoint)*pixel_size
            if target_distance > max_visible_radius:
                continue

            # Checks pass, add the next target to the queue.
            ring_id = lmax(labs(ix_viewpoint_block - ix_next_target>>block_bits),
                           labs(iy_viewpoint_block - iy_next_target>>block_bits))
            process_queue.push(
                TargetPixel(ix_next_target, iy_next_target,
                            ring_id, target_pixel.sector, target_distance))
            process_queue_set.insert(next_target_index)
    LOGGER.info('%6.2f%% complete after %.2fs', 100.0, time.time()-start_time)

    dem_managed_raster.close()
    aux_managed_raster.close()
    visibility_managed_raster.close()

    if temp_dir is not None:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            LOGGER.exception('Could not remove temporary folder %s', temp_dir)
