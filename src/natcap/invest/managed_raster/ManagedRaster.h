#include <iostream>

#include "gdal.h"
#include "gdal_priv.h"

#include <errno.h>
#include <string>
#include <set>
#include <cmath>
#include <list>
#include <utility>

int main() {
  cout << "Hello World" << endl;
  return 0;
}

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

typedef pair<int, double*> BlockBufferPair;

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

// int main(int argc, const char* argv[])
// {
//     if (argc != 2) {
//         return EINVAL;
//     }
//     const char* pszFilename = argv[1];

//     GDALDatasetUniquePtr poDataset;

//     const GDALAccess eAccess = GA_ReadOnly;
//     poDataset = GDALDatasetUniquePtr(GDALDataset::FromHandle(GDALOpen( pszFilename, eAccess )));
//     if( !poDataset )
//     {
//         ...; // handle error
//     }
//     return 0;
// }

class ManagedRaster {
    public:
        LRUCache<int, double*>* lru_cache;
        std::set<int> dirty_blocks;
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
        int write_mode;
        int closed;

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
            // raster_info = pygeoprocessing.get_raster_info(raster_path)
            GDALAllRegister();
            GDALDatasetUniquePtr poDataset;
            GDALRasterBand* band;

            cout << "path1: ";
            cout << raster_path;
            cout << "\n";

            GDALDataset *poSrcDS;

            cout << "h";

            poSrcDS = (GDALDataset *) GDALOpen( raster_path, GA_ReadOnly );
            cout << "a";
            poDataset = GDALDatasetUniquePtr(GDALDataset::FromHandle(poSrcDS));

            cout << "aaa";
            raster_x_size = poDataset->GetRasterXSize();
            raster_y_size = poDataset->GetRasterYSize();

            band = poDataset->GetRasterBand(band_id);

            band->GetBlockSize( &block_xsize, &block_ysize );

            block_xmod = block_xsize - 1;
            block_ymod = block_ysize - 1;

            // if not (1 <= band_id <= raster_info['n_bands']):
            //     err_msg = (
            //         "Error: band ID (%s) is not a valid band number. "
            //         "This exception is happening in Cython, so it will cause a "
            //         "hard seg-fault, but it's otherwise meant to be a "
            //         "ValueError." % (band_id))
            //     print(err_msg)
            //     raise ValueError(err_msg)
            // self.band_id = band_id

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

            cout << "bbb";
            // integer floor division
            block_nx = raster_x_size + (block_xsize) - 1 / block_xsize;
            block_ny = raster_y_size + (block_ysize) - 1 / block_ysize;

            cout << "ccc";
            lru_cache = new LRUCache<int, double*>(MANAGED_RASTER_N_BLOCKS);
            cout << "ddd";
            raster_path = raster_path;
            write_mode = write_mode;
            closed = 0;
            cout << "done";
            cout << "\n";
        }

        void set(long xi, long yi, double value) {
            // Set the pixel at `xi,yi` to `value`
            int block_xi = xi >> block_xbits;
            int block_yi = yi >> block_ybits;
            // this is the flat index for the block
            int block_index = block_yi * block_nx + block_xi;
            if (not lru_cache->exist(block_index)) {
                _load_block(block_index);
            }
            lru_cache->get(block_index)[
                    ((yi & (block_ymod)) << block_xbits) +
                    (xi & (block_xmod))] = value;
            if (write_mode) {

                std::set<int>::iterator dirty_itr = dirty_blocks.find(block_index);
                if (dirty_itr == dirty_blocks.end()) {
                    dirty_blocks.insert(block_index);
                }
            }
        }

        double get(long xi, long yi) {
            cout << "get\n";
            // Return the value of the pixel at `xi,yi`
            int block_xi = xi >> block_xbits;
            int block_yi = yi >> block_ybits;
            // this is the flat index for the block
            int block_index = block_yi * block_nx + block_xi;
            if (not lru_cache->exist(block_index)) {
                cout << "load block\n";
                _load_block(block_index);
                cout << "done loading block\n";
            }
            cout << "lru cache get\n";
            return lru_cache->get(block_index)[
                ((yi & (block_ymod)) << block_xbits) +
                (xi & (block_xmod))];
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
            if (xoff + win_xsize > raster_x_size) {
                win_xsize = win_xsize - (xoff + win_xsize - raster_x_size);
            }
            if (yoff + win_ysize > raster_y_size) {
                win_ysize = win_ysize - (yoff + win_ysize - raster_y_size);
            }

            GDALDatasetUniquePtr poDataset;
            GDALDataset *poSrcDS;

            poSrcDS = (GDALDataset *) GDALOpen( raster_path, GA_ReadOnly );
            poDataset = GDALDatasetUniquePtr(GDALDataset::FromHandle(poSrcDS));
            GDALRasterBand* band = poDataset->GetRasterBand(band_id);

            double *pafScanline;
            pafScanline = (double *) CPLMalloc(sizeof(double)*win_xsize * win_ysize);
            CPLErr err = band->RasterIO(GF_Read, xoff, yoff, win_xsize, win_ysize,
                        pafScanline, win_xsize, win_ysize, GDT_Float32,
                        0, 0 );
            if (not err == CE_None) {
                cout << "Error reading block";
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
                            double_buffer, win_xsize, win_ysize, GDT_Byte, 0, 0 );
                        if (not err == CE_None) {
                            cout << "Error reading block";
                        }
                    }
                }

                CPLFree(double_buffer);
                removed_value_list.pop_front();
            }

            cout << "done\n";
            if (write_mode) {
                GDALClose( (GDALDatasetH) poSrcDS );
            }
            cout << "end\n";
        }

};

class ManagedFlowDirRaster: public ManagedRaster {

public:

    ManagedFlowDirRaster() {}

    ManagedFlowDirRaster(char* raster_path, int band_id, bool write_mode)
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


class DownslopeNeighborIterator {
public:

    ManagedFlowDirRaster raster;
    int col;
    int row;
    int n_dir;
    int flow_dir;
    int flow_dir_sum;

    DownslopeNeighborIterator() { }

    DownslopeNeighborIterator(ManagedFlowDirRaster managed_raster, int x, int y)
        : raster { managed_raster }
        , col { x }
        , row { y }
    {
        n_dir = 0;
        flow_dir = raster.get(col, row);
        flow_dir_sum = 0;
    }


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
};

class UpslopeNeighborIterator {
public:

    ManagedFlowDirRaster raster;
    int col;
    int row;
    int n_dir;
    int flow_dir;

    UpslopeNeighborIterator() { }

    UpslopeNeighborIterator(ManagedFlowDirRaster managed_raster, int x, int y)
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

            n = NeighborTuple(n_dir, xj, yj, flow_ji / flow_dir_j_sum);
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next();
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

            n = NeighborTuple(n_dir, xj, yj, flow_ji / flow_dir_j_sum);
            n_dir += 1;
            return n;
        } else {
            n_dir += 1;
            return next_skip(skip);
        }
    }
};



