#include <algorithm>
#include <stack>
#include <queue>

#include "ManagedRaster.h"

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
    // """
    // Calculate the rasters defined by equations [3]-[7].

    // Note all input rasters must be in the same coordinate system and
    // have the same dimensions.

    // Args:
    //     precip_path_list (list): list of paths to monthly precipitation
    //         rasters. (model input)
    //     et0_path_list (list): path to monthly ET0 rasters. (model input)
    //     qf_m_path_list (list): path to monthly quickflow rasters calculated by
    //         Equation [1].
    //     flow_dir_path (str): path to a PyGeoprocessing Multiple Flow
    //         Direction raster indicating flow directions for this analysis.
    //     alpha_month_map (dict): fraction of upslope annual available recharge
    //         that is available in month m (indexed from 1).
    //     beta_i (float):  fraction of the upgradient subsidy that is available
    //         for downgradient evapotranspiration.
    //     gamma (float): the fraction of pixel recharge that is available to
    //         downgradient pixels.
    //     stream_path (str): path to the stream raster where 1 is a stream,
    //         0 is not, and nodata is outside of the DEM.
    //     kc_path_list (str): list of rasters of the monthly crop factor for the
    //         pixel.
    //     target_li_path (str): created by this call, path to local recharge
    //         derived from the annual water budget. (Equation 3).
    //     target_li_avail_path (str): created by this call, path to raster
    //         indicating available recharge to a pixel.
    //     target_l_sum_avail_path (str): created by this call, the recursive
    //         upslope accumulation of target_li_avail_path.
    //     target_aet_path (str): created by this call, the annual actual
    //         evapotranspiration.
    //     target_pi_path (str): created by this call, the annual precipitation on
    //         a pixel.

    //     Returns:
    //         None.

    // """
    long xs_root, ys_root, xoff, yoff;
    long xi, yi, mfd_dir_sum;
    long win_xsize, win_ysize;
    double kc_m, pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i;
    double l_avail_i, l_avail_j, l_sum_avail_i, l_sum_avail_j;
    bool upslope_defined;

    queue<pair<long, long>> work_queue;

    UpslopeNeighborIterator<T> up_iterator;
    DownslopeNeighborIterator<T> dn_iterator;

    // # used for time-delayed logging
    // cdef time_t last_log_time
    // last_log_time = ctime(NULL)

    ManagedFlowDirRaster<T> flow_dir_raster = ManagedFlowDirRaster<T>(
        flow_dir_path, 1, 0);
    NeighborTuple neighbor;

    // make sure that user input nodata values are defined
    // set to -1 if not defined
    // precipitation and evapotranspiration data should
    // always be non-negative
    vector<ManagedRaster> et0_m_rasters;
    vector<double> et0_m_nodata_list;
    for (int i = 0; i < et0_paths.size(); i++) {
        et0_m_rasters.push_back(ManagedRaster(et0_paths[i], 1, 0));
        if (et0_m_rasters[i].hasNodata) {
            et0_m_nodata_list.push_back(et0_m_rasters[i].nodata);
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

            // if ctime(NULL) - last_log_time > 5.0:
            //     last_log_time = ctime(NULL)
            //     LOGGER.info('Sediment deposition %.2f%% complete', 100 * (
            //         n_pixels_processed / float(flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size)))

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
                        up_iterator = UpslopeNeighborIterator<T>(flow_dir_raster, xi, yi);
                        neighbor = up_iterator.next_no_divide();
                        mfd_dir_sum = 0;
                        while (neighbor.direction < 8) {
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
                            neighbor = up_iterator.next_no_divide();
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

                        dn_iterator = DownslopeNeighborIterator<T>(flow_dir_raster, xi, yi);
                        neighbor = dn_iterator.next();
                        while (neighbor.direction < 8) {
                            work_queue.push(pair<long, long>(neighbor.x, neighbor.y));
                            neighbor = dn_iterator.next();
                        }
                    }
                }
            }
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
}

template<class T>
void run_route_baseflow_sum(
        char* flow_dir_path,
        char* l_path,
        char* l_avail_path,
        char* l_sum_path,
        char* stream_path,
        char* target_b_path,
        char* target_b_sum_path) {
    // """Route Baseflow through MFD as described in Equation 11.

    // Args:
    //     flow_dir_path (string): path to a pygeoprocessing multiple flow
    //         direction raster.
    //     l_path (string): path to local recharge raster.
    //     l_avail_path (string): path to local recharge raster that shows
    //         recharge available to the pixel.
    //     l_sum_path (string): path to upslope sum of l_path.
    //     stream_path (string): path to stream raster, 1 stream, 0 no stream,
    //         and nodata.
    //     target_b_path (string): path to created raster for per-pixel baseflow.
    //     target_b_sum_path (string): path to created raster for per-pixel
    //         upslope sum of baseflow.

    // Returns:
    //     None.
    // """

    // used for time-delayed logging
    // cdef time_t last_log_time
    // last_log_time = ctime(NULL)

    float target_nodata = -1e32;
    float b_i, b_sum_i, b_sum_j, l_j, l_avail_j, l_sum_j;
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

    UpslopeNeighborIterator<T> up_iterator;
    DownslopeNeighborIterator<T> dn_iterator;
    NeighborTuple neighbor;

    // int current_pixel = 0;

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

            // if ctime(NULL) - last_log_time > 5.0:
            //     last_log_time = ctime(NULL)
            //     LOGGER.info('Sediment deposition %.2f%% complete', 100 * (
            //         n_pixels_processed / float(flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size)))

            for (int row_index = 0; row_index < win_ysize; row_index++) {
                ys_root = yoff + row_index;
                for (int col_index = 0; col_index < win_xsize; col_index++) {
                    xs_root = xoff + col_index;

                    if (static_cast<int>(flow_dir_raster.get(xs_root, ys_root)) ==
                            static_cast<int>(flow_dir_raster.nodata)) {
                        // current_pixel += 1;
                        continue;
                    }

                    // search for a pixel that has no downslope neighbors,
                    // or whose downslope neighbors all have nodata in the stream raster (?)
                    outlet = true;
                    dn_iterator = DownslopeNeighborIterator(flow_dir_raster, xs_root, ys_root);
                    neighbor = dn_iterator.next();
                    while (neighbor.direction < 8) {
                        if (static_cast<int>(stream_raster.get(neighbor.x, neighbor.y)) !=
                                static_cast<int>(stream_raster.nodata)) {
                            outlet = 0;
                            break;
                        }
                        neighbor = dn_iterator.next();
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

                        // if ctime(NULL) - last_log_time > 5.0:
                        //     last_log_time = ctime(NULL)
                        //     LOGGER.info(
                        //         'route base flow %.2f%% complete',
                        //         100.0 * current_pixel / <float>(
                        //             flow_dir_raster.raster_x_size * flow_dir_raster.raster_y_size))

                        b_sum_i = 0;
                        downslope_defined = true;
                        dn_iterator = DownslopeNeighborIterator(flow_dir_raster, xi, yi);
                        neighbor = dn_iterator.next_no_skip();
                        flow_dir_sum = 0;
                        while (neighbor.direction < 8) {
                            flow_dir_sum += neighbor.flow_proportion;

                            if (neighbor.x < 0 or neighbor.x >= flow_dir_raster.raster_x_size or
                                neighbor.y < 0 or neighbor.y >= flow_dir_raster.raster_y_size) {
                                neighbor = dn_iterator.next_no_skip();
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
                            neighbor = dn_iterator.next_no_skip();
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

                        // current_pixel += 1;
                        up_iterator = UpslopeNeighborIterator<T>(flow_dir_raster, xi, yi);
                        neighbor = up_iterator.next();
                        while (neighbor.direction < 8) {
                            work_stack.push(pair<long, long>(neighbor.x, neighbor.y));
                            neighbor = up_iterator.next();
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
}
