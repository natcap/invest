# cython: language_level=3
# distutils: language = c++
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libc.math cimport isnan

cdef struct s_neighborTuple:
    int direction
    int x
    int y
    float flow_proportion

ctypedef s_neighborTuple NeighborTuple

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

    cdef inline void set(_ManagedRaster self, long xi, long yi, double value)
    cdef inline double get(_ManagedRaster self, long xi, long yi)
    cdef void _load_block(_ManagedRaster self, int block_index) except *


cdef class ManagedFlowDirRaster(_ManagedRaster):

    cdef bint is_local_high_point(ManagedFlowDirRaster self, long xi, long yi)

    cdef NeighborTuple* get_upslope_neighbors(ManagedFlowDirRaster self, long xi, long yi)

    cdef NeighborTuple* get_downslope_neighbors(ManagedFlowDirRaster self, long xi, long yi, bint skip_oob=*)


# These offsets are for the neighbor rows and columns according to the
# ordering: 3 2 1
#           4 x 0
#           5 6 7
cdef int *ROW_OFFSETS
cdef int *COL_OFFSETS
cdef int *FLOW_DIR_REVERSE_DIRECTION
cdef int *INFLOW_OFFSETS

cdef inline int is_close(double x, double y):
    if isnan(x) and isnan(y):
        return 1
    return abs(x - y) <= (1e-8 + 1e-05 * abs(y))
