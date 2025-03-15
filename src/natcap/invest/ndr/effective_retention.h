#include "ManagedRaster.h"
#include <cmath>
#include <stack>
#include <ctime>

// Calculate flow downhill effective_retention to the channel.
// Args:
//   flow_direction_path: a path to a flow direction raster (MFD or D8)
//   stream_path: a path to a raster where 1 indicates a
//     stream all other values ignored must be same dimensions and
//     projection as flow_direction_path.
//   retention_eff_lulc_path: a path to a raster indicating
//     the maximum retention efficiency that the landcover on that
//     pixel can accumulate.
//   crit_len_path: a path to a raster indicating the critical
//     length of the retention efficiency that the landcover on this
//     pixel.
//   effective_retention_path: path to a raster that is
//     created by this call that contains a per-pixel effective
//     sediment retention to the stream.
template<class T>
void run_effective_retention(
    char* flow_direction_path,
    char* stream_path,
    char* retention_eff_lulc_path,
    char* crit_len_path,
    char* to_process_flow_directions_path,
    char* effective_retention_path) {
  // Within a stream, the effective retention is 0
  int STREAM_EFFECTIVE_RETENTION = 0;
  float effective_retention_nodata = -1;
  stack<long> processing_stack;

  ManagedFlowDirRaster flow_dir_raster = ManagedFlowDirRaster<T>(
    flow_direction_path, 1, false);
  ManagedRaster stream_raster = ManagedRaster(stream_path, 1, false);
  ManagedRaster retention_eff_lulc_raster = ManagedRaster(
    retention_eff_lulc_path, 1, false);
  ManagedRaster crit_len_raster = ManagedRaster(crit_len_path, 1, false);
  ManagedRaster to_process_flow_directions_raster = ManagedRaster(
    to_process_flow_directions_path, 1, true);
  ManagedRaster effective_retention_raster = ManagedRaster(
    effective_retention_path, 1, true);

  long n_cols = flow_dir_raster.raster_x_size;
  long n_rows = flow_dir_raster.raster_y_size;
  // cell sizes must be square, so no reason to test at this point.
  double cell_size = stream_raster.geotransform[1];

  double crit_len_nodata = crit_len_raster.nodata;
  double retention_eff_nodata = retention_eff_lulc_raster.nodata;

  long win_xsize, win_ysize, xoff, yoff;
  long global_col, global_row;
  unsigned long flat_index;
  long flow_dir, neighbor_flow_dirs;
  double current_step_factor, step_size, crit_len, retention_eff_lulc;
  long neighbor_row, neighbor_col;
  int neighbor_outflow_dir, neighbor_outflow_dir_mask, neighbor_process_flow_dir;
  int outflow_dirs, dir_mask;
  NeighborTuple neighbor;
  bool should_seed;
  double working_retention_eff;
  DownslopeNeighborsNoSkip<T> dn_neighbors;
  UpslopeNeighbors<T> up_neighbors;
  bool has_outflow;
  double neighbor_effective_retention;
  double intermediate_retention;
  string s;
  long flow_dir_sum;
  time_t last_log_time = time(NULL);
  unsigned long n_pixels_processed = 0;
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
          "Effective retention " + std::to_string(
            100 * n_pixels_processed / total_n_pixels
          ) + " complete"
        );
      }

      for (int row_index = 0; row_index < win_ysize; row_index++) {
        global_row = yoff + row_index;
        for (int col_index = 0; col_index < win_xsize; col_index++) {
          global_col = xoff + col_index;
          outflow_dirs = int(to_process_flow_directions_raster.get(
            global_col, global_row));
          should_seed = false;
          // # see if this pixel drains to nodata or the edge, if so it's
          // # a drain
          for (int i = 0; i < 8; i++) {
            dir_mask = 1 << i;
            if ((outflow_dirs & dir_mask) > 0) {
              neighbor_col = COL_OFFSETS[i] + global_col;
              neighbor_row = ROW_OFFSETS[i] + global_row;
              if (neighbor_col < 0 or neighbor_col >= n_cols or
                neighbor_row < 0 or neighbor_row >= n_rows) {
                should_seed = true;
                outflow_dirs &= ~dir_mask;
              } else {
                // Only consider neighbor flow directions if the
                // neighbor index is within the raster.
                neighbor_flow_dirs = long(
                  to_process_flow_directions_raster.get(
                    neighbor_col, neighbor_row));
                if (neighbor_flow_dirs == 0) {
                  should_seed = true;
                  outflow_dirs &= ~dir_mask;
                }
              }
            }
          }

          if (should_seed) {
            // mark all outflow directions processed
            to_process_flow_directions_raster.set(
              global_col, global_row, outflow_dirs);
            processing_stack.push(global_row * n_cols + global_col);
          }
        }
      }

      while (processing_stack.size() > 0) {
        // loop invariant, we don't push a cell on the stack that
        // hasn't already been set for processing.
        flat_index = processing_stack.top();
        processing_stack.pop();
        global_row = flat_index / n_cols;  // integer floor division
        global_col = flat_index % n_cols;

        crit_len = crit_len_raster.get(global_col, global_row);
        retention_eff_lulc = retention_eff_lulc_raster.get(global_col, global_row);
        flow_dir = int(flow_dir_raster.get(global_col, global_row));
        if (stream_raster.get(global_col, global_row) == 1) {
          // if it's a stream, effective retention is 0.
          effective_retention_raster.set(global_col, global_row, STREAM_EFFECTIVE_RETENTION);
        } else if (is_close(crit_len, crit_len_nodata) or
            is_close(retention_eff_lulc, retention_eff_nodata) or
            flow_dir == 0) {
          // if it's nodata, effective retention is nodata.
          effective_retention_raster.set(
            global_col, global_row, effective_retention_nodata);
        } else {
          working_retention_eff = 0;

          dn_neighbors = DownslopeNeighborsNoSkip<T>(
            Pixel<T>(flow_dir_raster, global_col, global_row));
          has_outflow = false;
          flow_dir_sum = 0;
          for (auto neighbor: dn_neighbors) {
            has_outflow = true;
            flow_dir_sum += static_cast<long>(neighbor.flow_proportion);
            if (neighbor.x < 0 or neighbor.x >= n_cols or
              neighbor.y < 0 or neighbor.y >= n_rows) {
              continue;
            }
            if (neighbor.direction % 2 == 1) {
              step_size = cell_size * 1.41421356237;
            } else {
              step_size = cell_size;
            }
            // guard against a critical length factor that's 0
            if (crit_len > 0) {
              current_step_factor = exp(-5 * step_size / crit_len);
            } else {
              current_step_factor = 0;
            }

            neighbor_effective_retention = (
              effective_retention_raster.get(
                neighbor.x, neighbor.y));

            // Case 1: downslope neighbor is a stream pixel
            if (neighbor_effective_retention == STREAM_EFFECTIVE_RETENTION) {
              intermediate_retention = (
                retention_eff_lulc * (1 - current_step_factor));
             // Case 2: the current LULC's retention exceeds the neighbor's retention.
            } else if (retention_eff_lulc > neighbor_effective_retention) {
              intermediate_retention = (
                (neighbor_effective_retention * current_step_factor) +
                (retention_eff_lulc * (1 - current_step_factor)));
            // Case 3: the other 2 cases have not been hit.
            } else {
              intermediate_retention = neighbor_effective_retention;
            }

            working_retention_eff += (
              intermediate_retention * neighbor.flow_proportion);
          }

          if (has_outflow) {
            double v = working_retention_eff / flow_dir_sum;
            effective_retention_raster.set(
              global_col, global_row, v);
          } else {
            throw std::logic_error(
              "got to a cell that has no outflow! This error is happening"
              "in effective_retention.h");
          }
        }
        // search upslope to see if we need to push a cell on the stack
        // for i in range(8):
        up_neighbors = UpslopeNeighbors<T>(Pixel<T>(flow_dir_raster, global_col, global_row));
        for (auto neighbor: up_neighbors) {
          neighbor_outflow_dir = INFLOW_OFFSETS[neighbor.direction];
          neighbor_outflow_dir_mask = 1 << neighbor_outflow_dir;
          neighbor_process_flow_dir = int(
            to_process_flow_directions_raster.get(
              neighbor.x, neighbor.y));
          if (neighbor_process_flow_dir == 0) {
            // skip, due to loop invariant this must be a nodata pixel
            continue;
          }
          if ((neighbor_process_flow_dir & neighbor_outflow_dir_mask )== 0) {
            // no outflow
            continue;
          }
          // mask out the outflow dir that this iteration processed
          neighbor_process_flow_dir &= ~neighbor_outflow_dir_mask;
          to_process_flow_directions_raster.set(
            neighbor.x, neighbor.y, neighbor_process_flow_dir);
          if (neighbor_process_flow_dir == 0) {
            // if 0 then all downslope have been processed,
            // push on stack, otherwise another downslope pixel will
            // pick it up
            processing_stack.push(neighbor.y * n_cols + neighbor.x);
          }
        }
      }
      n_pixels_processed += win_xsize * win_ysize;
    }
  }
  stream_raster.close();
  crit_len_raster.close();
  retention_eff_lulc_raster.close();
  effective_retention_raster.close();
  flow_dir_raster.close();
  to_process_flow_directions_raster.close();
  log_msg(LogLevel::info, "Effective retention 100% complete");
}
