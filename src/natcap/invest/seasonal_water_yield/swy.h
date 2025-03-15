#include <algorithm>
#include <stack>
#include <queue>
#include <ctime>

#include "ManagedRaster.h"

// Calculate the rasters defined by equations [3]-[7].
//
// Note all input rasters must be in the same coordinate system and
// have the same dimensions.
//
// Args:
//   precip_paths: paths to monthly precipitation rasters. (model input)
//   et0_paths: paths to monthly ET0 rasters. (model input)
//   qf_m_paths: paths to monthly quickflow rasters calculated by
//     Equation [1].
//   flow_dir_path: path to a flow direction raster (MFD or D8). Indicate MFD
//    or D8 with the template argument.
//   kc_paths: list of rasters of the monthly crop factor for the pixel.
//   alpha_values: list of monthly alpha values (fraction of upslope annual
//     available recharge that is available in each month)
//   beta_i:  fraction of the upgradient subsidy that is available
//     for downgradient evapotranspiration.
//   gamma: the fraction of pixel recharge that is available to
//     downgradient pixels.
//   stream_path: path to the stream raster where 1 is a stream,
//     0 is not, and nodata is outside of the DEM.
//   target_li_path: created by this call, path to local recharge
//     derived from the annual water budget. (Equation 3).
//   target_li_avail_path: created by this call, path to raster
//     indicating available recharge to a pixel.
//   target_l_sum_avail_path: created by this call, the recursive
//     upslope accumulation of target_li_avail_path.
//   target_aet_path: created by this call, the annual actual
//     evapotranspiration.
//   target_pi_path: created by this call, the annual precipitation on
//     a pixel.
template<class T>
void run_calculate_local_recharge(
    vector<char*> precip_paths,
    vector<char*> et0_paths,
    vector<char*> qf_m_paths,
    char* flow_dir_path,
    vector<char*> kc_paths,
    vector<float> alpha_values,
    float beta_i,
    float gamma,
    char* stream_path,
    char* target_li_path,
    char* target_li_avail_path,
    char* target_l_sum_avail_path,
    char* target_aet_path,
    char* target_pi_path) {
  long xs_root, ys_root, xoff, yoff;
  long xi, yi, mfd_dir_sum;
  long win_xsize, win_ysize;
  double kc_m, pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i;
  double l_avail_i, l_avail_j, l_sum_avail_i, l_sum_avail_j;
  bool upslope_defined;

  queue<pair<long, long>> work_queue;

  UpslopeNeighborsNoDivide<T> up_neighbors;
  DownslopeNeighbors<T> dn_neighbors;

  ManagedFlowDirRaster<T> flow_dir_raster = ManagedFlowDirRaster<T>(
    flow_dir_path, 1, 0);
  NeighborTuple neighbor;

  time_t last_log_time = time(NULL);
  unsigned long n_pixels_processed = 0;
  float total_n_pixels = flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size;

  // make sure that user input nodata values are defined
  // set to -1 if not defined
  // precipitation and evapotranspiration data should
  // always be non-negative
  vector<ManagedRaster> et0_m_rasters;
  vector<double> et0_m_nodata_list;
  for (auto et0_m_path: et0_paths) {
    ManagedRaster et0_raster = ManagedRaster(et0_m_path, 1, 0);
    et0_m_rasters.push_back(et0_raster);
    if (et0_raster.hasNodata) {
      et0_m_nodata_list.push_back(et0_raster.nodata);
    } else {
      et0_m_nodata_list.push_back(-1);
    }
  }

  vector<ManagedRaster> precip_m_rasters;
  vector<double> precip_m_nodata_list;
  for (auto precip_m_path: precip_paths) {
    ManagedRaster precip_raster = ManagedRaster(precip_m_path, 1, 0);
    precip_m_rasters.push_back(precip_raster);
    if (precip_raster.hasNodata) {
      precip_m_nodata_list.push_back(precip_raster.nodata);
    } else {
      precip_m_nodata_list.push_back(-1);
    }
  }

  vector<ManagedRaster> qf_m_rasters;
  for (auto qf_m_path: qf_m_paths) {
    qf_m_rasters.push_back(ManagedRaster(qf_m_path, 1, 0));
  }

  vector<ManagedRaster> kc_m_rasters;
  for (auto kc_m_path: kc_paths) {
    kc_m_rasters.push_back(ManagedRaster(kc_m_path, 1, 0));
  }

  ManagedRaster target_li_raster = ManagedRaster(target_li_path, 1, 1);
  ManagedRaster target_li_avail_raster = ManagedRaster(target_li_avail_path, 1, 1);
  ManagedRaster target_l_sum_avail_raster = ManagedRaster(target_l_sum_avail_path, 1, 1);
  ManagedRaster target_aet_raster = ManagedRaster(target_aet_path, 1, 1);
  ManagedRaster target_pi_raster = ManagedRaster(target_pi_path, 1, 1);

  double target_nodata = -1e32;

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
          "Local recharge " + std::to_string(
            100 * n_pixels_processed / total_n_pixels
          ) + " complete"
        );
      }

      for (int row_index = 0; row_index < win_ysize; row_index++) {
        ys_root = yoff + row_index;
        for (int col_index = 0; col_index < win_xsize; col_index++) {
          xs_root = xoff + col_index;

          if (flow_dir_raster.get(xs_root, ys_root) == flow_dir_raster.nodata) {
            continue;
          }

          if (flow_dir_raster.is_local_high_point(xs_root, ys_root)) {
            work_queue.push(pair<long, long>(xs_root, ys_root));
          }

          while (work_queue.size() > 0) {
            xi = work_queue.front().first;
            yi = work_queue.front().second;
            work_queue.pop();

            l_sum_avail_i = target_l_sum_avail_raster.get(xi, yi);
            if (not is_close(l_sum_avail_i, target_nodata)) {
              // already defined
              continue;
            }

            // Equation 7, calculate L_sum_avail_i if possible, skip
            // otherwise
            upslope_defined = true;
            // initialize to 0 so we indicate we haven't tracked any
            // mfd values yet
            l_sum_avail_i = 0.0;
            mfd_dir_sum = 0;
            up_neighbors = UpslopeNeighborsNoDivide<T>(Pixel<T>(flow_dir_raster, xi, yi));
            for (auto neighbor: up_neighbors) {
              // pixel flows inward, check upslope
              l_sum_avail_j = target_l_sum_avail_raster.get(
                neighbor.x, neighbor.y);
              if (is_close(l_sum_avail_j, target_nodata)) {
                upslope_defined = false;
                break;
              }
              l_avail_j = target_li_avail_raster.get(
                neighbor.x, neighbor.y);
              // A step of Equation 7
              l_sum_avail_i += (
                l_sum_avail_j + l_avail_j) * neighbor.flow_proportion;
              mfd_dir_sum += static_cast<int>(neighbor.flow_proportion);
            }
            // calculate l_sum_avail_i by summing all the valid
            // directions then normalizing by the sum of the mfd
            // direction weights (Equation 8)
            if (upslope_defined) {
              // Equation 7
              if (mfd_dir_sum > 0) {
                l_sum_avail_i /= static_cast<float>(mfd_dir_sum);
              }
              target_l_sum_avail_raster.set(xi, yi, l_sum_avail_i);
            } else {
              // if not defined, we'll get it on another pass
              continue;
            }

            aet_i = 0;
            p_i = 0;
            qf_i = 0;

            for (int m_index = 0; m_index < 12; m_index++) {
              p_m = precip_m_rasters[m_index].get(xi, yi);
              if (not is_close(p_m, precip_m_rasters[m_index].nodata)) {
                p_i += p_m;
              } else {
                p_m = 0;
              }

              qf_m = qf_m_rasters[m_index].get(xi, yi);
              if (not is_close(qf_m, qf_m_rasters[m_index].nodata)) {
                qf_i += qf_m;
              } else {
                qf_m = 0;
              }

              kc_m = kc_m_rasters[m_index].get(xi, yi);
              pet_m = 0;
              et0_m = et0_m_rasters[m_index].get(xi, yi);
              if (not (
                  is_close(kc_m, kc_m_rasters[m_index].nodata) or
                  is_close(et0_m, et0_m_rasters[m_index].nodata))) {
                // Equation 6
                pet_m = kc_m * et0_m;
              }

              // Equation 4/5
              aet_i += min(
                pet_m,
                p_m - qf_m +
                alpha_values[m_index]*beta_i*l_sum_avail_i);
            }
            l_i = (p_i - qf_i - aet_i);
            l_avail_i = min(gamma * l_i, l_i);

            target_pi_raster.set(xi, yi, p_i);
            target_aet_raster.set(xi, yi, aet_i);
            target_li_raster.set(xi, yi, l_i);
            target_li_avail_raster.set(xi, yi, l_avail_i);

            dn_neighbors = DownslopeNeighbors<T>(Pixel<T>(flow_dir_raster, xi, yi));
            for (auto neighbor: dn_neighbors) {
              work_queue.push(pair<long, long>(neighbor.x, neighbor.y));
            }
          }
        }
      }
      n_pixels_processed += win_xsize * win_ysize;
    }
  }
  flow_dir_raster.close();
  target_li_raster.close();
  target_li_avail_raster.close();
  target_l_sum_avail_raster.close();
  target_aet_raster.close();
  target_pi_raster.close();
  for (int i = 0; i < 12; i++) {
    et0_m_rasters[i].close();
    precip_m_rasters[i].close();
    qf_m_rasters[i].close();
    kc_m_rasters[i].close();
  }
  log_msg(LogLevel::info, "Local recharge 100% complete");
}

// Route Baseflow as described in Equation 11.
// Args:
//   flow_dir_path: path to a MFD or D8 flow direction raster.
//   l_path: path to local recharge raster.
//   l_avail_path: path to local recharge raster that shows
//     recharge available to the pixel.
//   l_sum_path: path to upslope sum of l_path.
//   stream_path: path to stream raster, 1 stream, 0 no stream,
//     and nodata.
//   target_b_path: path to created raster for per-pixel baseflow.
//   target_b_sum_path: path to created raster for per-pixel
//     upslope sum of baseflow.
template<class T>
void run_route_baseflow_sum(
    char* flow_dir_path,
    char* l_path,
    char* l_avail_path,
    char* l_sum_path,
    char* stream_path,
    char* target_b_path,
    char* target_b_sum_path) {

  float target_nodata = static_cast<float>(-1e32);
  double b_i, b_sum_i, b_sum_j, l_j, l_avail_j, l_sum_j;
  double l_i, l_sum_i;
  long xi, yi, flow_dir_sum;
  long xs_root, ys_root, xoff, yoff;
  int win_xsize, win_ysize;
  stack<pair<long, long>> work_stack;
  bool outlet, downslope_defined;

  ManagedRaster target_b_sum_raster = ManagedRaster(target_b_sum_path, 1, 1);
  ManagedRaster target_b_raster = ManagedRaster(target_b_path, 1, 1);
  ManagedRaster l_raster = ManagedRaster(l_path, 1, 0);
  ManagedRaster l_avail_raster = ManagedRaster(l_avail_path, 1, 0);
  ManagedRaster l_sum_raster = ManagedRaster(l_sum_path, 1, 0);
  ManagedFlowDirRaster<T> flow_dir_raster = ManagedFlowDirRaster<T>(flow_dir_path, 1, 0);
  ManagedRaster stream_raster = ManagedRaster(stream_path, 1, 0);

  UpslopeNeighbors<T> up_neighbors;
  DownslopeNeighbors<T> dn_neighbors;
  DownslopeNeighborsNoSkip<T> dn_neighbors_no_skip;
  NeighborTuple neighbor;

  time_t last_log_time = time(NULL);
  unsigned long current_pixel = 0;
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

      for (int row_index = 0; row_index < win_ysize; row_index++) {
        ys_root = yoff + row_index;
        for (int col_index = 0; col_index < win_xsize; col_index++) {
          xs_root = xoff + col_index;

          if (static_cast<int>(flow_dir_raster.get(xs_root, ys_root)) ==
              static_cast<int>(flow_dir_raster.nodata)) {
            current_pixel += 1;
            continue;
          }

          // search for a pixel that has no downslope neighbors,
          // or whose downslope neighbors all have nodata in the stream raster (?)
          outlet = true;
          dn_neighbors = DownslopeNeighbors<T>(Pixel<T>(flow_dir_raster, xs_root, ys_root));
          for (auto neighbor: dn_neighbors) {
            if (static_cast<int>(stream_raster.get(neighbor.x, neighbor.y)) !=
                static_cast<int>(stream_raster.nodata)) {
              outlet = 0;
              break;
            }
          }
          if (not outlet) {
            continue;
          }
          work_stack.push(pair<long, long>(xs_root, ys_root));

          while (work_stack.size() > 0) {
            xi = work_stack.top().first;
            yi = work_stack.top().second;
            work_stack.pop();
            b_sum_i = target_b_sum_raster.get(xi, yi);
            if (not is_close(b_sum_i, target_nodata)) {
              continue;
            }

            if (time(NULL) - last_log_time > 5) {
              last_log_time = time(NULL);
              log_msg(
                LogLevel::info,
                "Baseflow " + std::to_string(
                  100 * current_pixel / total_n_pixels
                ) + " complete"
              );
            }

            b_sum_i = 0;
            downslope_defined = true;
            dn_neighbors_no_skip = DownslopeNeighborsNoSkip<T>(Pixel<T>(flow_dir_raster, xi, yi));
            flow_dir_sum = 0;
            for (auto neighbor: dn_neighbors_no_skip) {
              flow_dir_sum += static_cast<long>(neighbor.flow_proportion);

              if (neighbor.x < 0 or neighbor.x >= flow_dir_raster.raster_x_size or
                neighbor.y < 0 or neighbor.y >= flow_dir_raster.raster_y_size) {
                continue;
              }

              if (static_cast<int>(stream_raster.get(neighbor.x, neighbor.y))) {
                b_sum_i += neighbor.flow_proportion;
              } else {
                b_sum_j = target_b_sum_raster.get(neighbor.x, neighbor.y);
                if (is_close(b_sum_j, target_nodata)) {
                  downslope_defined = false;
                  break;
                }
                l_j = l_raster.get(neighbor.x, neighbor.y);
                l_avail_j = l_avail_raster.get(neighbor.x, neighbor.y);
                l_sum_j = l_sum_raster.get(neighbor.x, neighbor.y);

                if (l_sum_j != 0 and (l_sum_j - l_j) != 0) {
                  b_sum_i += neighbor.flow_proportion * (
                    (1 - l_avail_j / l_sum_j) * (
                      b_sum_j / (l_sum_j - l_j)));
                } else {
                  b_sum_i += neighbor.flow_proportion;
                }
              }
            }

            if (not downslope_defined) {
              continue;
            }
            l_i = l_raster.get(xi, yi);
            l_sum_i = l_sum_raster.get(xi, yi);

            if (flow_dir_sum > 0) {
              b_sum_i = l_sum_i * b_sum_i / flow_dir_sum;
            }


            if (l_sum_i != 0) {
              b_i = max(b_sum_i * l_i / l_sum_i, 0.0);
            } else {
              b_i = 0;
            }

            target_b_raster.set(xi, yi, b_i);
            target_b_sum_raster.set(xi, yi, b_sum_i);

            current_pixel += 1;
            up_neighbors = UpslopeNeighbors<T>(Pixel<T>(flow_dir_raster, xi, yi));
            for (auto neighbor: up_neighbors) {
              work_stack.push(pair<long, long>(neighbor.x, neighbor.y));
            }
          }
        }
      }
    }
  }
  target_b_sum_raster.close();
  target_b_raster.close();
  l_raster.close();
  l_avail_raster.close();
  l_sum_raster.close();
  flow_dir_raster.close();
  stream_raster.close();
  log_msg(LogLevel::info, "Baseflow 100% complete");
}
