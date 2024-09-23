#ifndef NATCAP_INVEST_MANAGEDRASTER_H_
#define NATCAP_INVEST_MANAGEDRASTER_H_

#include <iostream>

#include "gdal.h"
#include "gdal_priv.h"
#include <stdint.h>

#include <errno.h>
#include <format>
#include <string>
#include <set>
#include <stack>
#include <cmath>
#include <list>
#include <utility>
#include <iterator>

#include "LRUCache.h"

int MANAGED_RASTER_N_BLOCKS = pow(2, 6);
// given the pixel neighbor numbering system
//  3 2 1
//  4 x 0
//  5 6 7
// These offsets are for the neighbor rows and columns
int ROW_OFFSETS[8] = {0, -1, -1, -1,  0,  1, 1, 1};
int COL_OFFSETS[8] = {1,  1,  0, -1, -1, -1, 0, 1};
int FLOW_DIR_REVERSE_DIRECTION[8] = {4, 5, 6, 7, 0, 1, 2, 3};

// if a pixel `x` has a neighbor `n` in position `i`,
// then `n`'s neighbor in position `inflow_offsets[i]`
// is the original pixel `x`
int INFLOW_OFFSETS[8] = {4, 5, 6, 7, 0, 1, 2, 3};

typedef std::pair<int, double*> BlockBufferPair;

class D8 {};
class MFD {};

class NeighborTuple {
public:
    int direction, x, y;
    float flow_proportion;

    NeighborTuple () {}

    NeighborTuple (int direction, int x, int y, float flow_proportion) {
        this->direction = direction;
        this->x = x;
        this->y = y;
        this->flow_proportion = flow_proportion;
    }

    ~NeighborTuple () {}
};


class ManagedRaster {
    public:
        LRUCache<int, double*>* lru_cache;
        std::set<int> dirty_blocks;
        int* actualBlockWidths;
        int block_xsize;
        int block_ysize;
        int block_xmod;
        int block_ymod;
        int block_xbits;
        int block_ybits;
        long raster_x_size;
        long raster_y_size;
        int block_nx;
        int block_ny;
        char* raster_path;
        int band_id;
        GDALDataset* dataset;
        GDALRasterBand* band;
        int write_mode;
        int closed;
        double nodata;
        double* geotransform;
        int hasNodata;

        ManagedRaster() { }

        ManagedRaster(char* raster_path, int band_id, bool write_mode)
            : raster_path { raster_path }
            , band_id { band_id }
            , write_mode { write_mode }
        {
            // """Create new instance of Managed Raster.

            // Args:
            //     raster_path (char*): path to raster that has block sizes that are
            //         powers of 2. If not, an exception is raised.
            //     band_id (int): which band in `raster_path` to index. Uses GDAL
            //         notation that starts at 1.
            //     write_mode (boolean): if true, this raster is writable and dirty
            //         memory blocks will be written back to the raster as blocks
            //         are swapped out of the cache or when the object deconstructs.

            // Returns:
            //     None.
            //         """
            GDALAllRegister();

            dataset = (GDALDataset *) GDALOpen( raster_path, GA_Update );

            raster_x_size = dataset->GetRasterXSize();
            raster_y_size = dataset->GetRasterYSize();

            if (band_id < 1 or band_id > dataset->GetRasterCount()) {
                throw std::invalid_argument(
                    "Error: band ID is not a valid band number. "
                    "This error is happening in the ManagedRaster.h extension.");
            }
            band = dataset->GetRasterBand(band_id);
            band->GetBlockSize( &block_xsize, &block_ysize );

            block_xmod = block_xsize - 1;
            block_ymod = block_ysize - 1;

            nodata = band->GetNoDataValue( &hasNodata );

            geotransform = (double *) CPLMalloc(sizeof(double) * 6);
            dataset->GetGeoTransform(geotransform);

            // if (self.block_xsize & (self.block_xsize - 1) != 0) or (
            //         self.block_ysize & (self.block_ysize - 1) != 0):
            //     # If inputs are not a power of two, this will at least print
            //     # an error message. Unfortunately with Cython, the exception will
            //     # present itself as a hard seg-fault, but I'm leaving the
            //     # ValueError in here at least for readability.
            //     err_msg = (
            //         "Error: Block size is not a power of two: "
            //         "block_xsize: %d, %d, %s. This exception is happening"
            //         "in Cython, so it will cause a hard seg-fault, but it's"
            //         "otherwise meant to be a ValueError." % (
            //             self.block_xsize, self.block_ysize, raster_path))
            //     print(err_msg)
            //     raise ValueError(err_msg)

            block_xbits = log2(block_xsize);
            block_ybits = log2(block_ysize);

            // integer floor division
            block_nx = (raster_x_size + block_xsize - 1) / block_xsize;
            block_ny = (raster_y_size + block_ysize - 1) / block_ysize;

            int actual_x = 0;
            int actual_y = 0;
            actualBlockWidths = (int *) CPLMalloc(sizeof(int) * block_nx * block_ny);

            for (int block_yi = 0; block_yi < block_ny; block_yi++) {
                for (int block_xi = 0; block_xi < block_nx; block_xi++) {
                    band->GetActualBlockSize(block_xi, block_yi, &actual_x, &actual_y);
                    actualBlockWidths[block_yi * block_nx + block_xi] = actual_x;
                }
            }

            this->lru_cache = new LRUCache<int, double*>(MANAGED_RASTER_N_BLOCKS);
            closed = 0;
        }

        void set(long xi, long yi, double value) {
            // Set the pixel at `xi,yi` to `value`
            int block_xi = xi / block_xsize;
            int block_yi = yi / block_ysize;

            // this is the flat index for the block
            int block_index = block_yi * block_nx + block_xi;

            if (not lru_cache->exist(block_index)) {
                _load_block(block_index);
            }

            int idx = ((yi & block_ymod) * actualBlockWidths[block_index]) + (xi & block_xmod);
            lru_cache->get(block_index)[idx] = value;
            if (write_mode) {

                std::set<int>::iterator dirty_itr = dirty_blocks.find(block_index);
                if (dirty_itr == dirty_blocks.end()) {
                    dirty_blocks.insert(block_index);
                }
            }
        }

        double get(long xi, long yi) {
            // Return the value of the pixel at `xi,yi`
            int block_xi = xi / block_xsize;
            int block_yi = yi / block_ysize;

            // this is the flat index for the block
            int block_index = block_yi * block_nx + block_xi;

            if (not lru_cache->exist(block_index)) {
                _load_block(block_index);
            }
            double* block = lru_cache->get(block_index);



            // Using the property n % 2^i = n & (2^i - 1)
            // to efficienty compute the modulo: yi % block_xsize
            int idx = ((yi & block_ymod) * actualBlockWidths[block_index]) + (xi & block_xmod);

            double value = block[idx];
            return value;
        }

        void _load_block(int block_index) {
            int block_xi = block_index % block_nx;
            int block_yi = block_index / block_nx;

            // we need the offsets to subtract from global indexes for cached array
            int xoff = block_xi << block_xbits;
            int yoff = block_yi << block_ybits;

            double *double_buffer;
            list<BlockBufferPair> removed_value_list;

            // determine the block aligned xoffset for read as array

            // initially the win size is the same as the block size unless
            // we're at the edge of a raster
            int win_xsize = block_xsize;
            int win_ysize = block_ysize;

            // load a new block
            if ((xoff + win_xsize) > raster_x_size) {
                win_xsize = win_xsize - (xoff + win_xsize - raster_x_size);
            }
            if ((yoff + win_ysize) > raster_y_size) {
                win_ysize = win_ysize - (yoff + win_ysize - raster_y_size);
            }

            double *pafScanline = (double *) CPLMalloc(sizeof(double) * win_xsize * win_ysize);
            CPLErr err = band->RasterIO(GF_Read, xoff, yoff, win_xsize, win_ysize,
                        pafScanline, win_xsize, win_ysize, GDT_Float64,
                        0, 0 );

            if (err != CE_None) {
                std::cout << "Error reading block\n";
            }
            lru_cache->put(block_index, pafScanline, removed_value_list);
            while (not removed_value_list.empty()) {
                // write the changed value back if desired
                double_buffer = removed_value_list.front().second;

                if (write_mode) {
                    block_index = removed_value_list.front().first;

                    // write back the block if it's dirty
                    std::set<int>::iterator dirty_itr = dirty_blocks.find(block_index);
                    if (dirty_itr != dirty_blocks.end()) {
                        dirty_blocks.erase(dirty_itr);

                        block_xi = block_index % block_nx;
                        block_yi = block_index / block_nx;

                        xoff = block_xi << block_xbits;
                        yoff = block_yi << block_ybits;

                        win_xsize = block_xsize;
                        win_ysize = block_ysize;

                        if (xoff + win_xsize > raster_x_size) {
                            win_xsize = win_xsize - (xoff + win_xsize - raster_x_size);
                        }
                        if (yoff + win_ysize > raster_y_size) {
                            win_ysize = win_ysize - (yoff + win_ysize - raster_y_size);
                        }
                        err = band->RasterIO( GF_Write, xoff, yoff, win_xsize, win_ysize,
                            double_buffer, win_xsize, win_ysize, GDT_Float64, 0, 0 );
                        if (err != CE_None) {
                            std::cout << "Error writing block\n";
                        }
                    }
                }

                CPLFree(double_buffer);
                removed_value_list.pop_front();
            }
        }

        void close() {
        // """Close the _ManagedRaster and free up resources.

        //     This call writes any dirty blocks to disk, frees up the memory
        //     allocated as part of the cache, and frees all GDAL references.

        //     Any subsequent calls to any other functions in _ManagedRaster will
        //     have undefined behavior.
        // """
            if (closed) {
                return;
            }
            closed = 1;

            double *double_buffer;
            int block_xi;
            int block_yi;
            int block_index;
            // initially the win size is the same as the block size unless
            // we're at the edge of a raster
            int win_xsize;
            int win_ysize;

            // we need the offsets to subtract from global indexes for cached array
            int xoff;
            int yoff;

            if (not write_mode) {
                for (auto it = lru_cache->begin(); it != lru_cache->end(); it++) {
                    // write the changed value back if desired
                    CPLFree(it->second);
                }
                GDALClose( (GDALDatasetH) dataset );
                return;
            }

            // if we get here, we're in write_mode
            std::set<int>::iterator dirty_itr;
            for (auto it = lru_cache->begin(); it != lru_cache->end(); it++) {
                double_buffer = it->second;
                block_index = it->first;

                // write to disk if block is dirty
                dirty_itr = dirty_blocks.find(block_index);
                if (dirty_itr != dirty_blocks.end()) {
                    dirty_blocks.erase(dirty_itr);
                    block_xi = block_index % block_nx;
                    block_yi = block_index / block_nx;

                    // we need the offsets to subtract from global indexes for
                    // cached array
                    xoff = block_xi << block_xbits;
                    yoff = block_yi << block_ybits;

                    win_xsize = block_xsize;
                    win_ysize = block_ysize;

                    // clip window sizes if necessary
                    if (xoff + win_xsize > raster_x_size) {
                        win_xsize = win_xsize - (xoff + win_xsize - raster_x_size);
                    }
                    if (yoff + win_ysize > raster_y_size) {
                        win_ysize = win_ysize - (yoff + win_ysize - raster_y_size);
                    }
                    CPLErr err = band->RasterIO( GF_Write, xoff, yoff, win_xsize, win_ysize,
                        double_buffer, win_xsize, win_ysize, GDT_Float64, 0, 0 );
                    if (err != CE_None) {
                        std::cout << "Error writing block\n";
                    }
                }
                CPLFree(double_buffer);
            }
            GDALClose( (GDALDatasetH) dataset );
        }
};


class D8;
class MFD;


template<class T>
class ManagedFlowDirRaster: public ManagedRaster {

public:

    ManagedFlowDirRaster<T>() {}

    ManagedFlowDirRaster<T>(char* raster_path, int band_id, bool write_mode)
        : ManagedRaster(raster_path, band_id, write_mode)   // Call the superclass constructor in the subclass' initialization list.
        {
            // do something with bar
        }

    bool is_local_high_point(int xi, int yi) {
        // """Check if a given pixel is a local high point.

        // Args:
        //     xi (int): x coord in pixel space of the pixel to consider
        //     yi (int): y coord in pixel space of the pixel to consider

        // Returns:
        //     True if the pixel is a local high point, i.e. it has no
        //     upslope neighbors; False otherwise.
        // """
        int flow_dir_j;
        long xj, yj;
        float flow_ji;

        for (int n_dir = 0; n_dir < 8; n_dir++) {
            xj = xi + COL_OFFSETS[n_dir];
            yj = yi + ROW_OFFSETS[n_dir];
            if (xj < 0 or xj >= raster_x_size or yj < 0 or yj >= raster_y_size) {
                continue;
            }
            flow_dir_j = get(xj, yj);
            flow_ji = (0xF & (flow_dir_j >> (4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])));

            if (flow_ji) {
                return false;
            }
        }
        return true;

    }
};





template<class T>
class DownslopeNeighborIterator {
public:

    ManagedFlowDirRaster<T> raster;
    int col;
    int row;
    int n_dir;
    int flow_dir;
    int flow_dir_sum;

    DownslopeNeighborIterator<T>() { }

    DownslopeNeighborIterator<T>(ManagedFlowDirRaster<T> managed_raster, int x, int y)
        : raster { managed_raster }
        , col { x }
        , row { y }
    {
        n_dir = 0;
        flow_dir = int(raster.get(col, row));
        flow_dir_sum = 0;
    }

    template<typename T_ = T, std::enable_if_t<std::is_same<T_, MFD>::value>* = nullptr>
    NeighborTuple next() {
        NeighborTuple n;
        long xj, yj;
        int flow_ij;

        if (n_dir == 8) {
            n = NeighborTuple(8, -1, -1, -1);
            return n;
        }

        xj = col + COL_OFFSETS[n_dir];
        yj = row + ROW_OFFSETS[n_dir];

        if (xj < 0 or xj >= raster.raster_x_size or
                yj < 0 or yj >= raster.raster_y_size) {
            n_dir += 1;
            return next();
        }
        flow_ij = (flow_dir >> (n_dir * 4)) & 0xF;
        if (flow_ij) {
            flow_dir_sum += flow_ij;
            n = NeighborTuple(n_dir, xj, yj, flow_ij);
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next();
        }
    }

    template<typename T_ = T, std::enable_if_t<std::is_same<T_, D8>::value>* = nullptr>
    NeighborTuple next() {
        if (n_dir == 8) {
            return NeighborTuple(8, -1, -1, -1);
        }
        long xj = col + COL_OFFSETS[flow_dir];
        long yj = row + ROW_OFFSETS[flow_dir];
        if (xj < 0 or xj >= raster.raster_x_size or
                yj < 0 or yj >= raster.raster_y_size) {
            n_dir = 8;
            return NeighborTuple(8, -1, -1, -1);
        }
        n_dir = 8;
        return NeighborTuple(flow_dir, xj, yj, 1);
    }

    template<typename T_ = T, std::enable_if_t<std::is_same<T_, MFD>::value>* = nullptr>
    NeighborTuple next_no_skip() {
        NeighborTuple n;
        long xj, yj;
        int flow_ij;

        if (n_dir == 8) {
            return NeighborTuple(8, -1, -1, -1);
        }

        xj = col + COL_OFFSETS[n_dir];
        yj = row + ROW_OFFSETS[n_dir];

        flow_ij = (flow_dir >> (n_dir * 4)) & 0xF;
        if (flow_ij) {
            flow_dir_sum += flow_ij;
            n = NeighborTuple(n_dir, xj, yj, flow_ij);
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next_no_skip();
        }
    }

    template<typename T_ = T, std::enable_if_t<std::is_same<T_, D8>::value>* = nullptr>
    NeighborTuple next_no_skip() {
        if (n_dir == 8) {
            return NeighborTuple(8, -1, -1, -1);
        }
        long xj = col + COL_OFFSETS[flow_dir];
        long yj = row + ROW_OFFSETS[flow_dir];

        n_dir = 8;
        return NeighborTuple(flow_dir, xj, yj, 1);
    }
};

template<class T>
class UpslopeNeighborIterator {
public:

    ManagedFlowDirRaster<T> raster;
    int col;
    int row;
    int n_dir;
    int flow_dir;

    UpslopeNeighborIterator<T>() { }

    UpslopeNeighborIterator<T>(ManagedFlowDirRaster<T> managed_raster, int x, int y)
        : raster { managed_raster}
        , col { x }
        , row { y }
    {
        n_dir = 0;
    }

    NeighborTuple next() {

        NeighborTuple n;
        long xj, yj;
        int flow_dir_j;
        int flow_ji;
        long flow_dir_j_sum;

        if (n_dir == 8) {
            n = NeighborTuple(8, -1, -1, -1);
            return n;
        }

        xj = col + COL_OFFSETS[n_dir];
        yj = row + ROW_OFFSETS[n_dir];

        if (xj < 0 or xj >= raster.raster_x_size or
                yj < 0 or yj >= raster.raster_y_size) {
            n_dir += 1;
            return next();
        }

        flow_dir_j = int(raster.get(xj, yj));
        flow_ji = (0xF & (flow_dir_j >> (4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])));

        if (flow_ji) {
            flow_dir_j_sum = 0;
            for (int idx = 0; idx < 8; idx++) {
                flow_dir_j_sum += (flow_dir_j >> (idx * 4)) & 0xF;
            }

            n = NeighborTuple(n_dir, xj, yj, static_cast<float>(flow_ji) / static_cast<float>(flow_dir_j_sum));
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next();
        }
    }

    NeighborTuple next_no_divide() {

        NeighborTuple n;
        long xj, yj;
        int flow_dir_j;
        int flow_ji;

        if (n_dir == 8) {
            n = NeighborTuple(8, -1, -1, -1);
            return n;
        }

        xj = col + COL_OFFSETS[n_dir];
        yj = row + ROW_OFFSETS[n_dir];

        if (xj < 0 or xj >= raster.raster_x_size or
                yj < 0 or yj >= raster.raster_y_size) {
            n_dir += 1;
            return next_no_divide();
        }

        flow_dir_j = int(raster.get(xj, yj));
        flow_ji = (0xF & (flow_dir_j >> (4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])));

        if (flow_ji) {
            n = NeighborTuple(n_dir, xj, yj, static_cast<float>(flow_ji));
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next_no_divide();
        }
    }

    NeighborTuple next_skip(int skip) {

        NeighborTuple n;
        long xj, yj;
        int flow_dir_j;
        int flow_ji;
        long flow_dir_j_sum;

        if (n_dir == 8) {
            n = NeighborTuple(8, -1, -1, -1);
            return n;
        }

        xj = col + COL_OFFSETS[n_dir];
        yj = row + ROW_OFFSETS[n_dir];

        if (xj < 0 or xj >= raster.raster_x_size or
                yj < 0 or yj >= raster.raster_y_size or
                INFLOW_OFFSETS[n_dir] == skip) {
            n_dir += 1;
            return next_skip(skip);
        }

        flow_dir_j = int(raster.get(xj, yj));
        flow_ji = (0xF & (flow_dir_j >> (4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])));

        if (flow_ji) {
            flow_dir_j_sum = 0;
            for (int idx = 0; idx < 8; idx++) {
                flow_dir_j_sum += (flow_dir_j >> (idx * 4)) & 0xF;
            }

            n = NeighborTuple(n_dir, xj, yj, static_cast<float>(flow_ji) / static_cast<float>(flow_dir_j_sum));
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next_skip(skip);
        }
    }
};


bool is_close(double x, double y) {
    if (isnan(x) and isnan(y)) {
        return true;
    }
    return abs(x - y) <= (pow(10, -8) + pow(10, -05) * abs(y));
}

#endif  // NATCAP_INVEST_MANAGEDRASTER_H_