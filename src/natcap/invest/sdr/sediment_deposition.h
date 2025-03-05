#include "ManagedRaster.h"
#include <ctime>

// Calculate sediment deposition layer.
//
// This algorithm outputs both sediment deposition (t_i) and flux (f_i)::
//
//   t_i  = dt_i  * (sum over j ∈ J of f_j * p(j,i))
//
//   f_i  = (1 - dt_i) * (sum over j ∈ J of f_j * p(j,i)) + E'_i
//
//
//           (sum over k ∈ K of SDR_k * p(i,k)) - SDR_i
//   dt_i = --------------------------------------------
//               (1 - SDR_i)
//
// where:
//
// - ``p(i,j)`` is the proportion of flow from pixel ``i`` into pixel ``j``
// - ``J`` is the set of pixels that are immediate upslope neighbors of
//   pixel ``i``
// - ``K`` is the set of pixels that are immediate downslope neighbors of
//   pixel ``i``
// - ``E'`` is ``USLE * (1 - SDR)``, the amount of sediment loss from pixel
//   ``i`` that doesn't reach a stream (``e_prime_path``)
// - ``SDR`` is the sediment delivery ratio (``sdr_path``)
//
// ``f_i`` is recursively defined in terms of ``i``'s upslope neighbors.
// The algorithm begins from seed pixels that are local high points and so
// have no upslope neighbors. It works downslope from each seed pixel,
// only adding a pixel to the stack when all its upslope neighbors are
// already calculated.
//
// Note that this function is designed to be used in the context of the SDR
// model. Because the algorithm is recursive upslope and downslope of each
// pixel, nodata values in the SDR input would propagate along the flow path.
// This case is not handled because we assume the SDR and flow dir inputs
// will come from the SDR model and have nodata in the same places.
//
// Args:
//   flow_direction_path: a path to a flow direction raster,
//     in either MFD or D8 format. Specify with the ``algorithm`` arg.
//   e_prime_path: path to a raster that shows sources of
//     sediment that wash off a pixel but do not reach the stream.
//   f_path: path to a raster that shows the sediment flux
//     on a pixel for sediment that does not reach the stream.
//   sdr_path: path to Sediment Delivery Ratio raster.
//   target_sediment_deposition_path: path to created that
//     shows where the E' sources end up across the landscape.
template<class T>
void run_sediment_deposition(
  char* flow_direction_path,
  char* e_prime_path,
  char* f_path,
  char* sdr_path,
  char* sediment_deposition_path) {

  ManagedFlowDirRaster flow_dir_raster = ManagedFlowDirRaster<T>(
  flow_direction_path, 1, false);

  ManagedRaster e_prime_raster = ManagedRaster(e_prime_path, 1, false);
  ManagedRaster sdr_raster = ManagedRaster(sdr_path, 1, false);
  ManagedRaster f_raster = ManagedRaster(f_path, 1, true);
  ManagedRaster sediment_deposition_raster = ManagedRaster(
  sediment_deposition_path, 1, true);

  stack<long> processing_stack;
  float target_nodata = -1;
  long win_xsize, win_ysize, xoff, yoff;
  long global_col, global_row;
  int xs, ys;
  long flat_index;
  double downslope_sdr_weighted_sum;
  double sdr_i, e_prime_i, sdr_j, f_j;
  long flow_dir_sum;
  time_t last_log_time = time(NULL);
  unsigned long n_pixels_processed = 0;
  bool upslope_neighbors_processed;
  double f_j_weighted_sum;
  NeighborTuple neighbor;
  NeighborTuple neighbor_of_neighbor;
  double dr_i, t_i, f_i;
  UpslopeNeighbors<T> up_neighbors;
  DownslopeNeighbors<T> dn_neighbors;
  float total_n_pixels = flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size;

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

      if (time(NULL) - last_log_time > 5) {
        last_log_time = time(NULL);
        log_msg(
          LogLevel::info,
          "Sediment deposition "  + std::to_string(
            100 * n_pixels_processed / total_n_pixels
          ) + " complete"
        );
      }

      for (int row_index = 0; row_index < win_ysize; row_index++) {
        ys = yoff + row_index;
        for (int col_index = 0; col_index < win_xsize; col_index++) {
          xs = xoff + col_index;

          if (flow_dir_raster.get(xs, ys) == flow_dir_raster.nodata) {
            continue;
          }

          // if this can be a seed pixel and hasn't already been
          // calculated, put it on the stack
          if (flow_dir_raster.is_local_high_point(xs, ys) and
              is_close(sediment_deposition_raster.get(xs, ys), target_nodata)) {
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

            // # (sum over j ∈ J of f_j * p(i,j) in the equation for t_i)
            // # calculate the upslope f_j contribution to this pixel,
            // # the weighted sum of flux flowing onto this pixel from
            // # all neighbors
            f_j_weighted_sum = 0;
            up_neighbors = UpslopeNeighbors<T>(
              Pixel<T>(flow_dir_raster, global_col, global_row));
            for (auto neighbor: up_neighbors) {
              f_j = f_raster.get(neighbor.x, neighbor.y);
              if (is_close(f_j, target_nodata)) {
                continue;
              }
              // add the neighbor's flux value, weighted by the
              // flow proportion
              f_j_weighted_sum += neighbor.flow_proportion * f_j;
            }

            // # calculate sum of SDR values of immediate downslope
            // # neighbors, weighted by proportion of flow into each
            // # neighbor
            // # (sum over k ∈ K of SDR_k * p(i,k) in the equation above)
            downslope_sdr_weighted_sum = 0;
            dn_neighbors = DownslopeNeighbors<T>(
              Pixel<T>(flow_dir_raster, global_col, global_row));
            flow_dir_sum = 0;
            for (auto neighbor: dn_neighbors) {
              flow_dir_sum += static_cast<long>(neighbor.flow_proportion);
              sdr_j = sdr_raster.get(neighbor.x, neighbor.y);
              if (is_close(sdr_j, sdr_raster.nodata)) {
                continue;
              }
              if (sdr_j == 0) {
                // # this means it's a stream, for SDR deposition
                // # purposes, we set sdr to 1 to indicate this
                // # is the last step on which to retain sediment
                sdr_j = 1;
              }

              downslope_sdr_weighted_sum += (sdr_j * neighbor.flow_proportion);
              // # check if we can add neighbor j to the stack yet
              // #
              // # if there is a downslope neighbor it
              // # couldn't have been pushed on the processing
              // # stack yet, because the upslope was just
              // # completed
              upslope_neighbors_processed = true;
              // # iterate over each neighbor-of-neighbor
              up_neighbors = UpslopeNeighbors<T>(
                Pixel<T>(flow_dir_raster, neighbor.x, neighbor.y));
              for (auto neighbor_of_neighbor: up_neighbors) {
                if (INFLOW_OFFSETS[neighbor_of_neighbor.direction] == neighbor.direction) {
                  continue;
                }
                if (is_close(sediment_deposition_raster.get(
                  neighbor_of_neighbor.x, neighbor_of_neighbor.y
                ), target_nodata)) {
                  upslope_neighbors_processed = false;
                  break;
                }
              }
              // # if all upslope neighbors of neighbor j are
              // # processed, we can push j onto the stack.
              if (upslope_neighbors_processed) {
                processing_stack.push(
                  neighbor.y * flow_dir_raster.raster_x_size + neighbor.x);
              }
            }

            // # nodata pixels should propagate to the results
            sdr_i = sdr_raster.get(global_col, global_row);
            if (is_close(sdr_i, sdr_raster.nodata)) {
              continue;
            }
            e_prime_i = e_prime_raster.get(global_col, global_row);
            if (is_close(e_prime_i, e_prime_raster.nodata)) {
              continue;
            }

            if (flow_dir_sum) {
              downslope_sdr_weighted_sum /= flow_dir_sum;
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
            sediment_deposition_raster.set(global_col, global_row, t_i);
            f_raster.set(global_col, global_row, f_i);
          }
        }
      }
      n_pixels_processed += win_xsize * win_ysize;
    }
  }
  sediment_deposition_raster.close();
  flow_dir_raster.close();
  e_prime_raster.close();
  sdr_raster.close();
  f_raster.close();
  log_msg(LogLevel::info, "Sediment deposition 100% complete");
}
