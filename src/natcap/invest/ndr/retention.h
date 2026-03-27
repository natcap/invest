#include "ManagedRaster.h"
#include <cmath>
#include <stack>
#include <ctime>

// Calculate flow downhill retention to the channel.
// Args:
//   flow_direction_path: a path to a flow direction raster (MFD or D8)
//   stream_path: a path to a raster where 1 indicates a
//     stream all other values ignored must be same dimensions and
//     projection as flow_direction_path.
//   retention_efficiency_path: a path to a raster indicating
//     the maximum retention efficiency that the landcover on that
//     pixel can accumulate.
//   critical_length_path: a path to a raster indicating the critical
//     length of the retention efficiency that the landcover on this
//     pixel.
//   retention_path: path to a raster that is
//     created by this call that contains a per-pixel effective
//     sediment retention to the stream.
template<class T>
void calculate_retention(
    char* flow_direction_path,
    char* stream_path,
    char* retention_efficiency_path,
    char* critical_length_path,
    char* to_process_flow_directions_path,
    char* retention_path) {
  // Within a stream, the retention is 0
  int STREAM_RETENTION = 0;
  float retention_nodata = -1;
  stack<long> processing_stack;

  ManagedFlowDirRaster flow_dir_raster = ManagedFlowDirRaster<T>(
    flow_direction_path, 1, false);
  ManagedRaster stream_raster = ManagedRaster(stream_path, 1, false);
  ManagedRaster retention_efficiency_raster = ManagedRaster(
    retention_efficiency_path, 1, false);
  ManagedRaster critical_length_raster = ManagedRaster(critical_length_path, 1, false);
  ManagedRaster to_process_flow_directions_raster = ManagedRaster(
    to_process_flow_directions_path, 1, true);
  ManagedRaster retention_raster = ManagedRaster(retention_path, 1, true);

  long n_cols = flow_dir_raster.raster_x_size;
  long n_rows = flow_dir_raster.raster_y_size;
  // cell sizes must be square, so no reason to test at this point.
  double cell_size = stream_raster.geotransform[1];

  long win_xsize, win_ysize, xoff, yoff;
  long x_i, y_i;
  unsigned long flat_index;
  long flow_dir_i, neighbor_flow_dirs;
  double step_factor, step_length, critical_length_i, retention_efficiency_i;
  long neighbor_row, neighbor_col;
  int outflow_dir, outflow_dir_mask, directions_to_process;
  int outflow_dirs, dir_mask;
  bool should_seed;
  double retention_i;
  DownslopeNeighborsNoSkip<T> downslope_neighbors;
  UpslopeNeighbors<T> upslope_neighbors;
  bool has_outflow;
  double retention_j;
  double intermediate_retention;
  string s;
  long flow_dir_sum;
  time_t last_log_time = time(NULL);
  unsigned long n_pixels_processed = 0;
  float total_n_pixels = flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size;
  double sqrt_2 = sqrt(2);

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
          "Retention " + std::to_string(
            100 * n_pixels_processed / total_n_pixels
          ) + " complete"
        );
      }

      for (int row_index = 0; row_index < win_ysize; row_index++) {
        y_i = yoff + row_index;
        for (int col_index = 0; col_index < win_xsize; col_index++) {
          x_i = xoff + col_index;
          outflow_dirs = int(to_process_flow_directions_raster.get(
            x_i, y_i));
          should_seed = false;
          // # see if this pixel drains to nodata or the edge, if so it's
          // # a drain
          for (int i = 0; i < 8; i++) {
            dir_mask = 1 << i;
            if ((outflow_dirs & dir_mask) > 0) {
              neighbor_col = COL_OFFSETS[i] + x_i;
              neighbor_row = ROW_OFFSETS[i] + y_i;
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
              x_i, y_i, outflow_dirs);
            processing_stack.push(y_i * n_cols + x_i);
          }
        }
      }

      while (processing_stack.size() > 0) {
        // loop invariant, we don't push a cell on the stack that
        // hasn't already been set for processing.
        flat_index = processing_stack.top();
        processing_stack.pop();
        y_i = flat_index / n_cols;  // integer floor division
        x_i = flat_index % n_cols;

        critical_length_i = critical_length_raster.get(x_i, y_i);
        retention_efficiency_i = retention_efficiency_raster.get(x_i, y_i);
        flow_dir_i = int(flow_dir_raster.get(x_i, y_i));
        if (stream_raster.get(x_i, y_i) == 1) {
          // if pixel i is a stream, retention is 0.
          retention_raster.set(x_i, y_i, STREAM_RETENTION);
        } else if (
            is_close(critical_length_i, critical_length_raster.nodata) or
            is_close(retention_efficiency_i, retention_efficiency_raster.nodata) or
            is_close(flow_dir_i, flow_dir_raster.nodata)
          ) {
          // if inputs are nodata, retention is undefined.
          retention_raster.set(x_i, y_i, retention_nodata);
        } else {
          retention_i = 0;

          downslope_neighbors = DownslopeNeighborsNoSkip<T>(
            Pixel<T>(flow_dir_raster, x_i, y_i));
          has_outflow = false;
          flow_dir_sum = 0;
          // For each pixel j, a downslope neighbor of i
          for (auto j: downslope_neighbors) {
            has_outflow = true;
            flow_dir_sum += static_cast<long>(j.flow_proportion);
            if (j.x < 0 or j.x >= n_cols or j.y < 0 or j.y >= n_rows) {
              continue;
            }
            retention_j = retention_raster.get(j.x, j.y);
            if (is_close(retention_j, retention_nodata)) {
              continue;
            }

            // step length:
            // the distance between the centerpoints of pixel i and pixel j
            if (j.direction % 2 == 1) {
              step_length = cell_size * sqrt_2;
            } else {
              step_length = cell_size;
            }
            // guard against a critical length factor that's 0
            if (critical_length_i > 0) {
              step_factor = exp(-5 * step_length / critical_length_i);
            } else {
              step_factor = 0;
            }

            // Case 1: downslope neighbor is a stream pixel
            if (retention_j == STREAM_RETENTION) {
              intermediate_retention = retention_efficiency_i * (1 - step_factor);
            // Case 2: the current LULC's retention exceeds the neighbor's retention.
            } else if (retention_efficiency_i > retention_j) {
              intermediate_retention = (
                (retention_j * step_factor) +
                (retention_efficiency_i * (1 - step_factor)));
            // Case 3: the other 2 cases have not been hit.
            } else {
              intermediate_retention = retention_j;
            }

            retention_i += intermediate_retention * j.flow_proportion;
          }

          if (has_outflow) {
            retention_i = retention_i / flow_dir_sum;
            retention_raster.set(x_i, y_i, retention_i);
          } else {
            throw std::logic_error(
              "got to a cell that has no outflow! This error is happening"
              "in retention.h");
          }
        }
        // for each pixel k that is an upslope neighbor of i,
        // check if we can push k onto the stack yet
        upslope_neighbors = UpslopeNeighbors<T>(Pixel<T>(flow_dir_raster, x_i, y_i));
        for (auto k: upslope_neighbors) {
          outflow_dir = INFLOW_OFFSETS[k.direction];
          outflow_dir_mask = 1 << outflow_dir;
          directions_to_process = int(
            to_process_flow_directions_raster.get(k.x, k.y));
          if (directions_to_process == 0) {
            // skip, due to loop invariant this must be a nodata pixel
            continue;
          }
          if ((directions_to_process & outflow_dir_mask) == 0) {
            // no outflow
            continue;
          }
          // mask out the outflow dir that this iteration processed
          directions_to_process &= ~outflow_dir_mask;
          to_process_flow_directions_raster.set(k.x, k.y, directions_to_process);
          if (directions_to_process == 0) {
            // if 0 then all downslope have been processed,
            // push on stack, otherwise another downslope pixel will
            // pick it up
            processing_stack.push(k.y * n_cols + k.x);
          }
        }
      }
      n_pixels_processed += win_xsize * win_ysize;
    }
  }
  stream_raster.close();
  critical_length_raster.close();
  retention_efficiency_raster.close();
  retention_raster.close();
  flow_dir_raster.close();
  to_process_flow_directions_raster.close();
  log_msg(LogLevel::info, "Retention 100% complete");
}
