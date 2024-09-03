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

            nodata = band->GetNoDataValue();

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
            std::cout << "get " << xi << " " << yi << " " << value << " " << idx << " " << block_index << std::endl;
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

    // UpslopeNeighborIterator<T> getUpslopeNeighborIterator(int xi, int yi) {
    //     return UpslopeNeighborIterator<T>(this, xi, yi);
    // }

    // DownslopeNeighborIterator<T> getDownslopeNeighborIterator(int xi, int yi) {
    //     return DownslopeNeighborIterator<T>(this, xi, yi);
    // }

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
        flow_dir = raster.get(col, row);
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

    // template<typename T_ = T, std::enable_if_t<std::is_same<T_, D8>::value>* = nullptr>
    // NeighborTuple next() {
    //     long xj, yj;
    //     xj = col + COL_OFFSETS[flow_dir];
    //     yj = row + ROW_OFFSETS[flow_dir];
    //     return NeighborTuple(flow_dir, xj, yj, 1);
    // }

    NeighborTuple next_no_skip() {
        NeighborTuple n;
        long xj, yj;
        int flow_ij;

        if (n_dir == 8) {
            n = NeighborTuple(8, -1, -1, -1);
            return n;
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

        flow_dir_j = raster.get(xj, yj);
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

        flow_dir_j = raster.get(xj, yj);
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

        flow_dir_j = raster.get(xj, yj);
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


template <class T>
void run_sediment_deposition(
        char* flow_direction_path,
        char* e_prime_path,
        char* f_path,
        char* sdr_path,
        char* sediment_deposition_path) {

    ManagedFlowDirRaster flow_dir_raster = ManagedFlowDirRaster<T>(
        flow_direction_path, 1, false);
    ManagedRaster e_prime_raster = ManagedRaster(e_prime_path, 2, false);
    ManagedRaster sdr_raster = ManagedRaster(sdr_path, 1, false);
    ManagedRaster f_raster = ManagedRaster(f_path, 1, true);
    ManagedRaster sediment_deposition_raster = ManagedRaster(
        sediment_deposition_path, 1, true);

    int mfd_nodata = 0;
    stack<long> processing_stack;
    float target_nodata = -1;
    long win_xsize, win_ysize, xoff, yoff;
    long global_col, global_row;
    int xs, ys;
    long flat_index;
    float downslope_sdr_weighted_sum, sdr_i, sdr_j;
    float f_j;
    // unsigned long n_pixels_processed = 0;
    bool upslope_neighbors_processed;
    // time_t last_log_time = ctime(NULL)
    float f_j_weighted_sum;
    NeighborTuple neighbor;
    NeighborTuple neighbor_of_neighbor;
    float e_prime_i, dr_i, t_i, f_i;

    UpslopeNeighborIterator<T> up_iterator;
    DownslopeNeighborIterator<T> dn_iterator;

    // efficient way to calculate ceiling division:
    // a divided by b rounded up = (a + (b - 1)) / b
    // note that / represents integer floor division
    // https://stackoverflow.com/a/62032709/14451410
    int n_col_blocks = (flow_dir_raster.raster_x_size + (flow_dir_raster.block_xsize - 1)) / flow_dir_raster.block_xsize;
    int n_row_blocks = (flow_dir_raster.raster_y_size + (flow_dir_raster.block_ysize - 1)) / flow_dir_raster.block_ysize;

    for (int row_block_index = 0; row_block_index < n_row_blocks; row_block_index++) {
        yoff = row_block_index * flow_dir_raster.block_ysize;
        win_ysize = flow_dir_raster.raster_y_size - yoff;
        if (win_ysize > flow_dir_raster.block_ysize) {
            win_ysize = flow_dir_raster.block_ysize;
        }
        for (int col_block_index = 0; col_block_index < n_col_blocks; col_block_index++) {
            xoff = col_block_index * flow_dir_raster.block_xsize;
            win_xsize = flow_dir_raster.raster_x_size - xoff;
            if (win_xsize > flow_dir_raster.block_xsize) {
                win_xsize = flow_dir_raster.block_xsize;
            }

            std::cout << xoff << " " << yoff << " " << win_xsize << " " << win_ysize << std::endl;

            // if ctime(NULL) - last_log_time > 5.0:
            //     last_log_time = ctime(NULL)
            //     LOGGER.info('Sediment deposition %.2f%% complete', 100 * (
            //         n_pixels_processed / float(flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size)))

            for (int row_index = 0; row_index < win_ysize; row_index++) {
                ys = yoff + row_index;
                for (int col_index = 0; col_index < win_xsize; col_index++) {
                    xs = xoff + col_index;

                    if (flow_dir_raster.get(xs, ys) == mfd_nodata) {
                        std::cout << "continue" << std::endl;
                        continue;
                    }

                    // if this can be a seed pixel and hasn't already been
                    // calculated, put it on the stack
                    if (flow_dir_raster.is_local_high_point(xs, ys) and
                            is_close(sediment_deposition_raster.get(xs, ys), target_nodata)) {
                        std::cout << "push " << xs << " " << ys << std::endl;
                        processing_stack.push(ys * flow_dir_raster.raster_x_size + xs);
                    }

                    while (processing_stack.size() > 0) {
                        // # loop invariant: cell has all upslope neighbors
                        // # processed. this is true for seed pixels because they
                        // # have no upslope neighbors.
                        flat_index = processing_stack.top();
                        processing_stack.pop();
                        global_row = flat_index / flow_dir_raster.raster_x_size;
                        global_col = flat_index % flow_dir_raster.raster_x_size;
                        std::cout << "processing " << global_col << " " << global_row << std::endl;

                        // # (sum over j ∈ J of f_j * p(i,j) in the equation for t_i)
                        // # calculate the upslope f_j contribution to this pixel,
                        // # the weighted sum of flux flowing onto this pixel from
                        // # all neighbors
                        f_j_weighted_sum = 0;
                        up_iterator = UpslopeNeighborIterator(
                            flow_dir_raster, global_col, global_row);
                        neighbor = up_iterator.next();
                        while (neighbor.direction < 8) {

                            f_j = f_raster.get(neighbor.x, neighbor.y);
                            if (is_close(f_j, target_nodata)) {
                                neighbor = up_iterator.next();
                                std::cout << "continue a" << std::endl;
                                continue;
                            }

                            // add the neighbor's flux value, weighted by the
                            // flow proportion
                            f_j_weighted_sum += neighbor.flow_proportion * f_j;
                            neighbor = up_iterator.next();
                        }

                        // # calculate sum of SDR values of immediate downslope
                        // # neighbors, weighted by proportion of flow into each
                        // # neighbor
                        // # (sum over k ∈ K of SDR_k * p(i,k) in the equation above)
                        downslope_sdr_weighted_sum = 0;
                        dn_iterator = DownslopeNeighborIterator<T>(
                            flow_dir_raster, global_col, global_row);

                        std::cout << "b" << std::endl;
                        neighbor = dn_iterator.next();
                        while (neighbor.direction < 8) {

                            sdr_j = sdr_raster.get(neighbor.x, neighbor.y);
                            if (is_close(sdr_j, sdr_raster.nodata)) {
                                neighbor = dn_iterator.next();
                                continue;
                            }
                            if (sdr_j == 0) {
                                // # this means it's a stream, for SDR deposition
                                // # purposes, we set sdr to 1 to indicate this
                                // # is the last step on which to retain sediment
                                sdr_j = 1;
                            }

                            downslope_sdr_weighted_sum += (
                                sdr_j * neighbor.flow_proportion);
                            // # check if we can add neighbor j to the stack yet
                            // #
                            // # if there is a downslope neighbor it
                            // # couldn't have been pushed on the processing
                            // # stack yet, because the upslope was just
                            // # completed
                            upslope_neighbors_processed = true;
                            // # iterate over each neighbor-of-neighbor
                            up_iterator = UpslopeNeighborIterator<T>(
                                flow_dir_raster, neighbor.x, neighbor.y);
                            neighbor_of_neighbor = up_iterator.next_skip(neighbor.direction);
                            while (neighbor_of_neighbor.direction < 8) {
                                if (is_close(sediment_deposition_raster.get(
                                    neighbor_of_neighbor.x, neighbor_of_neighbor.y
                                ), target_nodata)) {
                                    upslope_neighbors_processed = false;
                                    break;
                                }
                                neighbor_of_neighbor = up_iterator.next_skip(neighbor.direction);
                            }
                            // # if all upslope neighbors of neighbor j are
                            // # processed, we can push j onto the stack.
                            if (upslope_neighbors_processed) {
                                processing_stack.push(
                                    neighbor.y * flow_dir_raster.raster_x_size + neighbor.x);
                            }

                            neighbor = dn_iterator.next();
                        }

                        std::cout << "c" << std::endl;
                        // # nodata pixels should propagate to the results
                        sdr_i = sdr_raster.get(global_col, global_row);
                        std::cout << sdr_i << " " << sdr_raster.nodata << std::endl;
                        if (is_close(sdr_i, sdr_raster.nodata)) {
                            std::cout << "d " << std::endl;
                            continue;
                        }
                        e_prime_i = e_prime_raster.get(global_col, global_row);
                        if (is_close(e_prime_i, e_prime_raster.nodata)) {
                            std::cout << "e" << std::endl;
                            continue;
                        }

                        if (dn_iterator.flow_dir_sum) {
                            downslope_sdr_weighted_sum /= dn_iterator.flow_dir_sum;
                        }

                        // # This condition reflects property A in the user's guide.
                        if (downslope_sdr_weighted_sum < sdr_i) {
                            // # i think this happens because of our low resolution
                            // # flow direction, it's okay to zero out.
                            downslope_sdr_weighted_sum = sdr_i;
                        }

                        // # these correspond to the full equations for
                        // # dr_i, t_i, and f_i given in the docstring
                        if (sdr_i == 1) {
                            // # This reflects property B in the user's guide and is
                            // # an edge case to avoid division-by-zero.
                            dr_i = 1;
                        } else {
                            dr_i = (downslope_sdr_weighted_sum - sdr_i) / (1 - sdr_i);
                        }

                        // # Lisa's modified equations
                        t_i = dr_i * f_j_weighted_sum;  // deposition, a.k.a trapped sediment
                        f_i = (1 - dr_i) * f_j_weighted_sum + e_prime_i; // flux

                        // # On large flow paths, it's possible for dr_i, f_i and t_i
                        // # to have very small negative values that are numerically
                        // # equivalent to 0. These negative values were raising
                        // # questions on the forums and it's easier to clamp the
                        // # values here than to explain IEEE 754.
                        if (dr_i < 0) {
                            dr_i = 0;
                        }
                        if (t_i < 0) {
                            t_i = 0;
                        }
                        if (f_i < 0) {
                            f_i = 0;
                        }
                        std::cout << "set" << " " << global_col << " " << global_row << std::endl;
                        sediment_deposition_raster.set(global_col, global_row, t_i);
                        f_raster.set(global_col, global_row, f_i);
                    }
                }
            }
            // n_pixels_processed += win_xsize * win_ysize;
        }
    }
    sediment_deposition_raster.close();
    flow_dir_raster.close();
    e_prime_raster.close();
    sdr_raster.close();
    f_raster.close();

    // LOGGER.info('Sediment deposition 100% complete')

}
