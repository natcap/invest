# cython: profile=False
# cython: language_level=2
import logging
import os
import collections
import sys
import gc
import pygeoprocessing

import numpy
cimport numpy
cimport cython
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from libcpp.pair cimport pair
from libcpp.stack cimport stack
from libcpp.queue cimport queue
from libc.time cimport time as ctime
from ..managed_raster.managed_raster cimport _ManagedRaster

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

cdef int is_close(double x, double y):
    return abs(x-y) <= (1e-8+1e-05*abs(y))

LOGGER = logging.getLogger(__name__)

cdef int N_MONTHS = 12

# used to loop over neighbors and offset the x/y values as defined below
#  321
#  4x0
#  567
cdef int* NEIGHBOR_OFFSET_ARRAY = [
    1, 0,  # 0
    1, -1,  # 1
    0, -1,  # 2
    -1, -1,  # 3
    -1, 0,  # 4
    -1, 1,  # 5
    0, 1,  # 6
    1, 1  # 7
    ]

# index into this array with a direction and get the index for the reverse
# direction. Useful for determining the direction a neighbor flows into a
# cell.
cdef int* FLOW_DIR_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]


def is_local_high_point(int xi, int yi, _ManagedRaster flow_dir_raster):
    ns = list(yield_upslope_neighbors(xi, yi, flow_dir_raster))
    if ns:
        return False
    return True


def yield_upslope_neighbors(int xi, int yi, _ManagedRaster flow_dir_raster):

    upslope_neighbor_tuples = []
    flow_sum = 0.0
    for n_dir in xrange(8):
        xj = xi + NEIGHBOR_OFFSET_ARRAY[2 * n_dir]
        yj = yi + NEIGHBOR_OFFSET_ARRAY[2 * n_dir + 1]
        if (xj < 0 or xj >= flow_dir_raster.raster_x_size or
                yj < 0 or yj >= flow_dir_raster.raster_y_size):
            continue
        flow_dir_j = <int>flow_dir_raster.get(xj, yj)
        flow_ji = (0xF & (flow_dir_j >> (4 * FLOW_DIR_REVERSE_DIRECTION[n_dir])))
        if flow_ji:
            upslope_neighbor_tuples.append((xj, yj, flow_ji))
            flow_sum += flow_ji

    for xj, yj, flow_ji in upslope_neighbor_tuples:
        p_ji = float(flow_ji) / flow_sum
        yield xj, yj, p_ji


def yield_downslope_neighbors(int xi, int yi, _ManagedRaster flow_dir_raster):
    flow_dir = <int>flow_dir_raster.get(xi, yi)
    flow_sum = 0.0
    downslope_neighbor_tuples = []
    for n_dir in xrange(8):
        flow_ij = (flow_dir >> (n_dir * 4)) & 0xF
        flow_sum += flow_ij
        if flow_ij:
            # flows in this direction
            xj = xi + NEIGHBOR_OFFSET_ARRAY[2 * n_dir]
            yj = yi + NEIGHBOR_OFFSET_ARRAY[2 * n_dir + 1]
            if (xj < 0 or xj >= flow_dir_raster.raster_x_size or
                    yj < 0 or yj >= flow_dir_raster.raster_y_size):
                continue
            downslope_neighbor_tuples.append((xj, yj, flow_ij))

    for xj, yj, flow_ij in downslope_neighbor_tuples:
        p_ij = float(flow_ij) / flow_sum
        yield xj, yj, p_ij


cpdef calculate_local_recharge(
        precip_path_list, et0_path_list, qf_m_path_list, flow_dir_mfd_path,
        kc_path_list, alpha_month_map, float beta_i, float gamma, stream_path,
        target_li_path, target_li_avail_path, target_l_sum_avail_path,
        target_aet_path, target_pi_path):
    """
    Calculate the rasters defined by equations [3]-[7].

    Note all input rasters must be in the same coordinate system and
    have the same dimensions.

    Args:
        precip_path_list (list): list of paths to monthly precipitation
            rasters. (model input)
        et0_path_list (list): path to monthly ET0 rasters. (model input)
        qf_m_path_list (list): path to monthly quickflow rasters calculated by
            Equation [1].
        flow_dir_mfd_path (str): path to a PyGeoprocessing Multiple Flow
            Direction raster indicating flow directions for this analysis.
        alpha_month_map (dict): fraction of upslope annual available recharge
            that is available in month m (indexed from 1).
        beta_i (float):  fraction of the upgradient subsidy that is available
            for downgradient evapotranspiration.
        gamma (float): the fraction of pixel recharge that is available to
            downgradient pixels.
        stream_path (str): path to the stream raster where 1 is a stream,
            0 is not, and nodata is outside of the DEM.
        kc_path_list (str): list of rasters of the monthly crop factor for the
            pixel.
        target_li_path (str): created by this call, path to local recharge
            derived from the annual water budget. (Equation 3).
        target_li_avail_path (str): created by this call, path to raster
            indicating available recharge to a pixel.
        target_l_sum_avail_path (str): created by this call, the recursive
            upslope accumulation of target_li_avail_path.
        target_aet_path (str): created by this call, the annual actual
            evapotranspiration.
        target_pi_path (str): created by this call, the annual precipitation on
            a pixel.

        Returns:
            None.

    """
    cdef int i_n, flow_dir_nodata, flow_dir_mfd
    cdef int peak_pixel
    cdef long xs, ys, xs_root, ys_root
    cdef int flow_dir_s
    cdef long xi, yi, xj, yj
    cdef int flow_dir_j, p_ij_base
    cdef int n_dir
    cdef long raster_x_size, raster_y_size
    cdef double pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i, l_avail_i
    cdef float qf_nodata, kc_nodata

    cdef int j_neighbor_end_index
    cdef float mfd_direction_array[8]

    cdef queue[pair[long, long]] work_queue
    cdef _ManagedRaster et0_m_raster, qf_m_raster, kc_m_raster

    cdef numpy.ndarray[numpy.npy_float32, ndim=1] alpha_month_array = (
        numpy.array(
            [x[1] for x in sorted(alpha_month_map.items())],
            dtype=numpy.float32))

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # we know the PyGeoprocessing MFD raster flow dir type is a 32 bit int.
    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_mfd_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']
    cdef _ManagedRaster flow_raster = _ManagedRaster(flow_dir_mfd_path, 1, 0)

    # make sure that user input nodata values are defined
    # set to -1 if not defined
    # precipitation and evapotranspiration data should
    # always be non-negative
    et0_m_raster_list = []
    et0_m_nodata_list = []
    for et0_path in et0_path_list:
        et0_m_raster_list.append(_ManagedRaster(et0_path, 1, 0))
        nodata = pygeoprocessing.get_raster_info(et0_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        et0_m_nodata_list.append(nodata)

    precip_m_raster_list = []
    precip_m_nodata_list = []
    for precip_m_path in precip_path_list:
        precip_m_raster_list.append(_ManagedRaster(precip_m_path, 1, 0))
        nodata = pygeoprocessing.get_raster_info(precip_m_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        precip_m_nodata_list.append(nodata)

    qf_m_raster_list = []
    qf_m_nodata_list = []
    for qf_m_path in qf_m_path_list:
        qf_m_raster_list.append(_ManagedRaster(qf_m_path, 1, 0))
        qf_m_nodata_list.append(
            pygeoprocessing.get_raster_info(qf_m_path)['nodata'][0])

    kc_m_raster_list = []
    kc_m_nodata_list = []
    for kc_m_path in kc_path_list:
        kc_m_raster_list.append(_ManagedRaster(kc_m_path, 1, 0))
        kc_m_nodata_list.append(
            pygeoprocessing.get_raster_info(kc_m_path)['nodata'][0])

    target_nodata = -1e32
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_li_raster = _ManagedRaster(
        target_li_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef _ManagedRaster target_li_avail_raster = _ManagedRaster(
        target_li_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_l_sum_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef _ManagedRaster target_l_sum_avail_raster = _ManagedRaster(
        target_l_sum_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_aet_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_aet_raster = _ManagedRaster(
        target_aet_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_pi_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_pi_raster = _ManagedRaster(
        target_pi_path, 1, 1)


    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = offset_dict['xoff'] + offset_dict['yoff'] * raster_x_size
            LOGGER.info(
                'peak point detection %.2f%% complete',
                100.0 * current_pixel / <float>(
                    raster_x_size * raster_y_size))

        # search block for a peak pixel where no other pixel drains to it.
        for ys in xrange(offset_dict['win_ysize']):
            ys_root = offset_dict['yoff'] + ys
            for xs in xrange(offset_dict['win_xsize']):
                xs_root = offset_dict['xoff'] + xs

                if is_local_high_point(xs_root, ys_root, flow_raster):
                    work_queue.push(
                        pair[long, long](xs_root, ys_root))

                while work_queue.size() > 0:
                    xi = work_queue.front().first
                    yi = work_queue.front().second
                    work_queue.pop()

                    l_sum_avail_i = target_l_sum_avail_raster.get(xi, yi)
                    if not is_close(l_sum_avail_i, target_nodata):
                        # already defined
                        continue

                    # Equation 7, calculate L_sum_avail_i if possible, skip
                    # otherwise
                    upslope_defined = 1
                    # initialize to 0 so we indicate we haven't tracked any
                    # mfd values yet
                    j_neighbor_end_index = 0
                    l_sum_avail_i = 0
                    for xj, yj, p_ij in yield_upslope_neighbors(xi, yi, flow_raster):
                        # pixel flows inward, check upslope
                        l_sum_avail_j = target_l_sum_avail_raster.get(xj, yj)
                        if is_close(l_sum_avail_j, target_nodata):
                            upslope_defined = 0
                            break
                        l_avail_j = target_li_avail_raster.get(xj, yj)
                        # A step of Equation 7
                        l_sum_avail_i += (l_sum_avail_j + l_avail_j) * p_ij
                        j_neighbor_end_index += 1
                    # calculate l_sum_avail_i by summing all the valid
                    # directions then normalizing by the sum of the mfd
                    # direction weights (Equation 8)
                    if upslope_defined:
                        # Equation 7
                        target_l_sum_avail_raster.set(xi, yi, l_sum_avail_i)
                    else:
                        # if not defined, we'll get it on another pass
                        continue

                    aet_i = 0
                    p_i = 0
                    qf_i = 0

                    for m_index in range(12):
                        precip_m_raster = (
                            <_ManagedRaster?>precip_m_raster_list[m_index])
                        qf_m_raster = (
                            <_ManagedRaster?>qf_m_raster_list[m_index])
                        et0_m_raster = (
                            <_ManagedRaster?>et0_m_raster_list[m_index])
                        kc_m_raster = (
                            <_ManagedRaster?>kc_m_raster_list[m_index])

                        et0_nodata = et0_m_nodata_list[m_index]
                        precip_nodata = precip_m_nodata_list[m_index]
                        qf_nodata = qf_m_nodata_list[m_index]
                        kc_nodata = kc_m_nodata_list[m_index]

                        p_m = precip_m_raster.get(xi, yi)
                        if not is_close(p_m, precip_nodata):
                            p_i += p_m
                        else:
                            p_m = 0

                        qf_m = qf_m_raster.get(xi, yi)
                        if not is_close(qf_m, qf_nodata):
                            qf_i += qf_m
                        else:
                            qf_m = 0

                        kc_m = kc_m_raster.get(xi, yi)
                        pet_m = 0
                        et0_m = et0_m_raster.get(xi, yi)
                        if not (
                                is_close(kc_m, kc_nodata) or
                                is_close(et0_m, et0_nodata)):
                            # Equation 6
                            pet_m = kc_m * et0_m

                        # Equation 4/5
                        aet_i += min(
                            pet_m,
                            p_m - qf_m +
                            alpha_month_array[m_index]*beta_i*l_sum_avail_i)

                    target_pi_raster.set(xi, yi, p_i)

                    target_aet_raster.set(xi, yi, aet_i)
                    l_i = (p_i - qf_i - aet_i)

                    # Equation 8
                    l_avail_i = min(gamma*l_i, l_i)

                    target_li_raster.set(xi, yi, l_i)
                    target_li_avail_raster.set(xi, yi, l_avail_i)

                    for xi_n, yi_n, _ in yield_downslope_neighbors(
                            xi, yi, flow_raster):
                        work_queue.push(pair[long, long](xi_n, yi_n))




def route_baseflow_sum(
        flow_dir_mfd_path, l_path, l_avail_path, l_sum_path,
        stream_path, target_b_path, target_b_sum_path):
    """Route Baseflow through MFD as described in Equation 11.

    Args:
        flow_dir_mfd_path (string): path to a pygeoprocessing multiple flow
            direction raster.
        l_path (string): path to local recharge raster.
        l_avail_path (string): path to local recharge raster that shows
            recharge available to the pixel.
        l_sum_path (string): path to upslope sum of l_path.
        stream_path (string): path to stream raster, 1 stream, 0 no stream,
            and nodata.
        target_b_path (string): path to created raster for per-pixel baseflow.
        target_b_sum_path (string): path to created raster for per-pixel
            upslope sum of baseflow.

    Returns:
        None.
    """

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    cdef float target_nodata = -1e32
    cdef int stream_val, outlet
    cdef float b_i, b_sum_i, l_j, l_avail_j, l_sum_j
    cdef long xi, yi, xj, yj
    cdef float p_ij
    cdef int flow_dir_i, p_ij_base
    cdef int flow_dir_nodata
    cdef long raster_x_size, raster_y_size, xs_root, ys_root
    cdef int n_dir
    cdef int xs, ys, flow_dir_s
    cdef int stream_nodata
    cdef stack[pair[long, long]] work_stack

    # we know the PyGeoprocessing MFD raster flow dir type is a 32 bit int.
    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_mfd_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']

    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_sum_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])

    cdef _ManagedRaster target_b_sum_raster = _ManagedRaster(
        target_b_sum_path, 1, 1)
    cdef _ManagedRaster target_b_raster = _ManagedRaster(
        target_b_path, 1, 1)
    cdef _ManagedRaster l_raster = _ManagedRaster(l_path, 1, 0)
    cdef _ManagedRaster l_avail_raster = _ManagedRaster(l_avail_path, 1, 0)
    cdef _ManagedRaster l_sum_raster = _ManagedRaster(l_sum_path, 1, 0)
    cdef _ManagedRaster flow_dir_mfd_raster = _ManagedRaster(
        flow_dir_mfd_path, 1, 0)

    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, 0)

    current_pixel = 0
    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):

        # search block for a peak pixel where no other pixel drains to it.
        for ys in xrange(offset_dict['win_ysize']):
            ys_root = offset_dict['yoff'] + ys
            for xs in xrange(offset_dict['win_xsize']):
                xs_root = offset_dict['xoff'] + xs
                flow_dir_s = <int>flow_dir_mfd_raster.get(xs_root, ys_root)
                if flow_dir_s == flow_dir_nodata:
                    current_pixel += 1
                    continue
                outlet = 1
                for xj, yj, p_ij in yield_downslope_neighbors(
                        xs_root, ys_root, flow_dir_mfd_raster):
                    stream_val = <int>stream_raster.get(xj, yj)
                    if stream_val != stream_nodata:
                        outlet = 0
                        break
                if not outlet:
                    continue
                work_stack.push(pair[long, long](xs_root, ys_root))

                while work_stack.size() > 0:
                    xi = work_stack.top().first
                    yi = work_stack.top().second
                    work_stack.pop()
                    b_sum_i = target_b_sum_raster.get(xi, yi)
                    if not is_close(b_sum_i, target_nodata):
                        continue

                    if ctime(NULL) - last_log_time > 5.0:
                        last_log_time = ctime(NULL)
                        LOGGER.info(
                            'route base flow %.2f%% complete',
                            100.0 * current_pixel / <float>(
                                raster_x_size * raster_y_size))

                    b_sum_i = 0.0
                    downslope_defined = 1

                    for xj, yj, p_ij in yield_downslope_neighbors(xi, yi, flow_dir_mfd_raster):
                        stream_val = <int>stream_raster.get(xj, yj)

                        if stream_val:
                            b_sum_i += p_ij
                        else:
                            b_sum_j = target_b_sum_raster.get(xj, yj)
                            if is_close(b_sum_j, target_nodata):
                                downslope_defined = 0
                                break
                            l_j = l_raster.get(xj, yj)
                            l_avail_j = l_avail_raster.get(xj, yj)
                            l_sum_j = l_sum_raster.get(xj, yj)

                            if l_sum_j != 0 and (l_sum_j - l_j) != 0:
                                b_sum_i += p_ij * (
                                    (1 - l_avail_j / l_sum_j) * (
                                        b_sum_j / (l_sum_j - l_j)))
                            else:
                                b_sum_i += p_ij

                    if not downslope_defined:
                        continue
                    l_sum_i = l_sum_raster.get(xi, yi)
                    b_sum_i = l_sum_i * b_sum_i
                    target_b_sum_raster.set(xi, yi, b_sum_i)
                    l_i = l_raster.get(xi, yi)
                    if l_sum_i != 0:
                        b_i = max(b_sum_i * l_i / l_sum_i, 0.0)
                    else:
                        b_i = 0.0
                    target_b_raster.set(xi, yi, b_i)
                    current_pixel += 1

                    for xj, yj, _ in yield_upslope_neighbors(xi, yi, flow_dir_mfd_raster):
                        work_stack.push(pair[long, long](xj, yj))
