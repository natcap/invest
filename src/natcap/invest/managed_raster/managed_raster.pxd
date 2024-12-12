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
        void close()

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
        void close()

    cdef cppclass NeighborTuple:
        NeighborTuple() except +
        NeighborTuple(int, int, int, float) except +
        int direction, x, y
        float flow_proportion

    cdef cppclass UpslopeNeighborIteratorSkip:
        ManagedFlowDirRaster raster
        int col
        int row
        int n_dir
        int flow_dir

        UpslopeNeighborIteratorSkip()
        UpslopeNeighborIteratorSkip(ManagedFlowDirRaster, int, int, int)
        NeighborTuple next()

    cdef cppclass Pixel:
        ManagedFlowDirRaster raster
        int x
        int y
        int val

        Pixel()
        Pixel(ManagedFlowDirRaster, int, int)

    cdef cppclass NeighborIterator:
        NeighborIterator()
        NeighborIterator(NeighborTuple* n)
        NeighborIterator(Pixel)
        NeighborTuple operator*()
        NeighborIterator operator++()
        bint operator==(NeighborIterator)
        bint operator!=(NeighborIterator)

    cdef cppclass DownslopeNeighborIterator:
        DownslopeNeighborIterator()
        DownslopeNeighborIterator(NeighborTuple* n)
        DownslopeNeighborIterator(Pixel)
        NeighborTuple operator*()
        DownslopeNeighborIterator operator++()
        bint operator==(DownslopeNeighborIterator)
        bint operator!=(DownslopeNeighborIterator)

    cdef cppclass DownslopeNeighborNoSkipIterator:
        DownslopeNeighborNoSkipIterator()
        DownslopeNeighborNoSkipIterator(NeighborTuple* n)
        DownslopeNeighborNoSkipIterator(Pixel)
        NeighborTuple operator*()
        DownslopeNeighborNoSkipIterator operator++()
        bint operator==(DownslopeNeighborNoSkipIterator)
        bint operator!=(DownslopeNeighborNoSkipIterator)

    cdef cppclass UpslopeNeighborIterator:
        UpslopeNeighborIterator()
        UpslopeNeighborIterator(NeighborTuple* n)
        UpslopeNeighborIterator(Pixel)
        NeighborTuple operator*()
        UpslopeNeighborIterator operator++()
        bint operator==(UpslopeNeighborIterator)
        bint operator!=(UpslopeNeighborIterator)

    cdef cppclass UpslopeNeighborNoDivideIterator:
        UpslopeNeighborNoDivideIterator()
        UpslopeNeighborNoDivideIterator(NeighborTuple* n)
        UpslopeNeighborNoDivideIterator(Pixel)
        NeighborTuple operator*()
        UpslopeNeighborNoDivideIterator operator++()
        bint operator==(UpslopeNeighborNoDivideIterator)
        bint operator!=(UpslopeNeighborNoDivideIterator)

    cdef cppclass Neighbors:
        Neighbors()
        Neighbors(Pixel)
        NeighborIterator begin()
        NeighborIterator end()

    cdef cppclass DownslopeNeighbors:
        DownslopeNeighbors()
        DownslopeNeighbors(Pixel)
        DownslopeNeighborIterator begin()
        DownslopeNeighborIterator end()

    cdef cppclass DownslopeNeighborsNoSkip:
        DownslopeNeighborsNoSkip()
        DownslopeNeighborsNoSkip(Pixel)
        DownslopeNeighborNoSkipIterator begin()
        DownslopeNeighborNoSkipIterator end()

    cdef cppclass UpslopeNeighbors:
        UpslopeNeighbors()
        UpslopeNeighbors(Pixel)
        UpslopeNeighborIterator begin()
        UpslopeNeighborIterator end()

    cdef cppclass UpslopeNeighborsNoDivide:
        UpslopeNeighborsNoDivide()
        UpslopeNeighborsNoDivide(Pixel)
        UpslopeNeighborNoDivideIterator begin()
        UpslopeNeighborNoDivideIterator end()

    bint is_close(double, double)

    int[8] INFLOW_OFFSETS
    int[8] COL_OFFSETS
    int[8] ROW_OFFSETS
    int[8] FLOW_DIR_REVERSE_DIRECTION
