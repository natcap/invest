# cython: language_level=3
# distutils: language = c++
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.math cimport isnan

# this is a least recently used cache written in C++ in an external file,
# exposing here so ManagedRaster can use it
cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)

cdef extern from "ManagedRaster.h":
    cdef cppclass ManagedRaster:
        LRUCache[int, double*]* lru_cache
        cset[int] dirty_blocks
        int block_xsize
        int block_ysize
        int block_xmod
        int block_ymod
        int block_xbits
        int block_ybits
        long raster_x_size
        long raster_y_size
        int block_nx
        int block_ny
        int write_mode
        string raster_path
        int band_id
        int closed

        ManagedRaster() except +
        ManagedRaster(char*, int, bool) except +
        void set(long xi, long yi, double value)
        double get(long xi, long yi)
        void _load_block(int block_index) except *

    cdef cppclass ManagedFlowDirRaster:
        LRUCache[int, double*]* lru_cache
        cset[int] dirty_blocks
        int block_xsize
        int block_ysize
        int block_xmod
        int block_ymod
        int block_xbits
        int block_ybits
        long raster_x_size
        long raster_y_size
        int block_nx
        int block_ny
        int write_mode
        string raster_path
        int band_id
        int closed

        bint is_local_high_point(int xi, int yi)

        ManagedFlowDirRaster() except +
        ManagedFlowDirRaster(char*, int, bool) except +
        void set(long xi, long yi, double value)
        double get(long xi, long yi)

    cdef cppclass NeighborTuple:
        NeighborTuple() except +
        NeighborTuple(int, int, int, float) except +
        int direction, x, y
        float flow_proportion

    cdef cppclass DownslopeNeighborIterator:
        ManagedFlowDirRaster raster
        int col
        int row
        int n_dir
        int flow_dir
        int flow_dir_sum

        DownslopeNeighborIterator()
        DownslopeNeighborIterator(ManagedFlowDirRaster, int, int)
        NeighborTuple next()

    cdef cppclass UpslopeNeighborIterator:
        ManagedFlowDirRaster raster
        int col
        int row
        int n_dir
        int flow_dir

        UpslopeNeighborIterator()
        UpslopeNeighborIterator(ManagedFlowDirRaster, int, int)
        NeighborTuple next()
        NeighborTuple next_skip(int skip)





# cdef class UpslopeNeighborIterator:

#     cdef ManagedFlowDirRaster raster
#     cdef int col
#     cdef int row
#     cdef int n_dir
#     cdef int flow_dir

#     cdef NeighborTuple next(UpslopeNeighborIterator self)
#     cdef NeighborTuple next_skip(UpslopeNeighborIterator self, int skip)


# cdef class ManagedFlowDirRaster(PyManagedRaster):

#     cdef bint is_local_high_point(ManagedFlowDirRaster self, int xi, int yi)

#     cdef vector[NeighborTuple] get_upslope_neighbors(ManagedFlowDirRaster self, long xi, long yi)

#     cdef vector[NeighborTuple] get_upslope_neighbors_skip(ManagedFlowDirRaster self, long xi, long yi, int skip)

#     cdef vector[NeighborTuple] get_downslope_neighbors(ManagedFlowDirRaster self, long xi, long yi, bint skip_oob=*)


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
