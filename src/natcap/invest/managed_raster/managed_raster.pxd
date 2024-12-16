# cython: language_level=3
# distutils: language = c++
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp.stack cimport stack
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
        double nodata

        ManagedRaster() except +
        ManagedRaster(char*, int, bool) except +
        void set(long xi, long yi, double value)
        double get(long xi, long yi)
        void _load_block(int block_index) except *
        void close()

    cdef cppclass ManagedFlowDirRaster[T]:
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
        double nodata

        bint is_local_high_point(int xi, int yi)

        ManagedFlowDirRaster() except +
        ManagedFlowDirRaster(char*, int, bool) except +
        void set(long xi, long yi, double value)
        double get(long xi, long yi)
        void close()

    cdef cppclass D8
    cdef cppclass MFD

    cdef cppclass NeighborTuple:
        NeighborTuple() except +
        NeighborTuple(int, int, int, float) except +
        int direction, x, y
        float flow_proportion

    cdef cppclass DownslopeNeighborIterator[T]:
        ManagedFlowDirRaster[T] raster
        int col
        int row
        int n_dir
        int flow_dir
        int flow_dir_sum

        DownslopeNeighborIterator()
        DownslopeNeighborIterator(ManagedFlowDirRaster[T], int, int)
        void next()

    cdef cppclass DownslopeNeighborNoSkipIterator[T]:
        ManagedFlowDirRaster[T] raster
        int col
        int row
        int n_dir
        int flow_dir
        int flow_dir_sum

        DownslopeNeighborNoSkipIterator()
        DownslopeNeighborNoSkipIterator(ManagedFlowDirRaster[T], int, int)
        void next()

    cdef cppclass UpslopeNeighborIterator[T]:
        ManagedFlowDirRaster[T] raster
        int col
        int row
        int n_dir
        int flow_dir

        UpslopeNeighborIterator()
        UpslopeNeighborIterator(ManagedFlowDirRaster[T], int, int)
        void next()

    cdef cppclass UpslopeNeighborNoDivideIterator[T]:
        ManagedFlowDirRaster[T] raster
        int col
        int row
        int n_dir
        int flow_dir

        UpslopeNeighborNoDivideIterator()
        UpslopeNeighborNoDivideIterator(ManagedFlowDirRaster[T], int, int)
        void next()

    bint is_close(double, double)

    int[8] INFLOW_OFFSETS
    int[8] COL_OFFSETS
    int[8] ROW_OFFSETS
    int[8] FLOW_DIR_REVERSE_DIRECTION
