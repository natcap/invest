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
from libcpp.vector cimport vector
from libc.time cimport time as ctime
from ..managed_raster.managed_raster cimport ManagedRaster
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport Pixel
from ..managed_raster.managed_raster cimport DownslopeNeighbors
from ..managed_raster.managed_raster cimport UpslopeNeighborIterator
from ..managed_raster.managed_raster cimport UpslopeNeighborIteratorNoDivide
from ..managed_raster.managed_raster cimport NeighborTuple
from ..managed_raster.managed_raster cimport is_close

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef int N_MONTHS = 12

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
    cdef long xs, ys, xs_root, ys_root, xoff, yoff
    cdef int flow_dir_s
    cdef long xi, yi, xj, yj
    cdef int flow_dir_j
    cdef int n_dir
    cdef long raster_x_size, raster_y_size, win_xsize, win_ysize
    cdef double pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i, l_avail_i
    cdef float qf_nodata, kc_nodata

    cdef float mfd_direction_array[8]

    cdef queue[pair[long, long]] work_queue
    cdef ManagedRaster et0_m_raster, qf_m_raster, kc_m_raster

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

    cdef ManagedFlowDirRaster flow_raster = ManagedFlowDirRaster(
        flow_dir_mfd_path.encode('utf-8'), 1, 0)
    cdef NeighborTuple neighbor

    # make sure that user input nodata values are defined
    # set to -1 if not defined
    # precipitation and evapotranspiration data should
    # always be non-negative
    cdef vector[ManagedRaster] et0_m_rasters
    et0_m_nodata_list = []
    for et0_path in et0_path_list:
        et0_m_rasters.push_back(ManagedRaster(et0_path.encode('utf-8'), 1, 0))
        nodata = pygeoprocessing.get_raster_info(et0_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        et0_m_nodata_list.append(nodata)

    cdef vector[ManagedRaster] precip_m_rasters
    precip_m_nodata_list = []
    for precip_m_path in precip_path_list:
        precip_m_rasters.push_back(ManagedRaster(precip_m_path.encode('utf-8'), 1, 0))
        nodata = pygeoprocessing.get_raster_info(precip_m_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        precip_m_nodata_list.append(nodata)

    cdef vector[ManagedRaster] qf_m_rasters
    qf_m_nodata_list = []
    for qf_m_path in qf_m_path_list:
        qf_m_rasters.push_back(ManagedRaster(qf_m_path.encode('utf-8'), 1, 0))
        qf_m_nodata_list.append(
            pygeoprocessing.get_raster_info(qf_m_path)['nodata'][0])

    cdef vector[ManagedRaster] kc_m_rasters
    kc_m_nodata_list = []
    for kc_m_path in kc_path_list:
        kc_m_rasters.push_back(ManagedRaster(kc_m_path.encode('utf-8'), 1, 0))
        kc_m_nodata_list.append(
            pygeoprocessing.get_raster_info(kc_m_path)['nodata'][0])

    target_nodata = -1e32
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef ManagedRaster target_li_raster = ManagedRaster(
        target_li_path.encode('utf-8'), 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef ManagedRaster target_li_avail_raster = ManagedRaster(
        target_li_avail_path.encode('utf-8'), 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_l_sum_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef ManagedRaster target_l_sum_avail_raster = ManagedRaster(
        target_l_sum_avail_path.encode('utf-8'), 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_aet_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef ManagedRaster target_aet_raster = ManagedRaster(
        target_aet_path.encode('utf-8'), 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_pi_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef ManagedRaster target_pi_raster = ManagedRaster(
        target_pi_path.encode('utf-8'), 1, 1)

    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info(
                'peak point detection %.2f%% complete',
                100.0 * current_pixel / <float>(
                    raster_x_size * raster_y_size))

        # search block for a peak pixel where no other pixel drains to it.
        for ys in xrange(win_ysize):
            ys_root = yoff + ys
            for xs in xrange(win_xsize):
                xs_root = xoff + xs

                if flow_raster.is_local_high_point(xs_root, ys_root):
                    work_queue.push(pair[long, long](xs_root, ys_root))

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
                    l_sum_avail_i = 0.0
                    up_iterator = UpslopeNeighborIteratorNoDivide(flow_raster, xi, yi)
                    neighbor = up_iterator.next()
                    mfd_dir_sum = 0
                    while neighbor.direction < 8:
                        # pixel flows inward, check upslope
                        l_sum_avail_j = target_l_sum_avail_raster.get(
                            neighbor.x, neighbor.y)
                        if is_close(l_sum_avail_j, target_nodata):
                            upslope_defined = 0
                            break
                        l_avail_j = target_li_avail_raster.get(
                            neighbor.x, neighbor.y)
                        # A step of Equation 7
                        l_sum_avail_i += (
                            l_sum_avail_j + l_avail_j) * neighbor.flow_proportion
                        mfd_dir_sum += <int>neighbor.flow_proportion
                        neighbor = up_iterator.next()
                    # calculate l_sum_avail_i by summing all the valid
                    # directions then normalizing by the sum of the mfd
                    # direction weights (Equation 8)
                    if upslope_defined:
                        # Equation 7
                        if mfd_dir_sum > 0:
                            l_sum_avail_i /= <float>mfd_dir_sum
                        target_l_sum_avail_raster.set(xi, yi, l_sum_avail_i)
                    else:
                        # if not defined, we'll get it on another pass
                        continue

                    aet_i = 0
                    p_i = 0
                    qf_i = 0

                    for m_index in range(12):
                        precip_m_raster = (
                            <ManagedRaster?>precip_m_rasters[m_index])
                        qf_m_raster = (
                            <ManagedRaster?>qf_m_rasters[m_index])
                        et0_m_raster = (
                            <ManagedRaster?>et0_m_rasters[m_index])
                        kc_m_raster = (
                            <ManagedRaster?>kc_m_rasters[m_index])

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

                    l_i = (p_i - qf_i - aet_i)
                    l_avail_i = min(gamma * l_i, l_i)

                    target_pi_raster.set(xi, yi, p_i)
                    target_aet_raster.set(xi, yi, aet_i)
                    target_li_raster.set(xi, yi, l_i)
                    target_li_avail_raster.set(xi, yi, l_avail_i)

                    dn_neighbors = DownslopeNeighbors(Pixel(flow_raster, xi, yi))
                    for neighbor in dn_neighbors:
                        work_queue.push(pair[long, long](neighbor.x, neighbor.y))

    flow_raster.close()
    target_li_raster.close()
    target_li_avail_raster.close()
    target_l_sum_avail_raster.close()
    target_aet_raster.close()
    target_pi_raster.close()
    for raster in et0_m_rasters:
        raster.close()
    for raster in precip_m_rasters:
        raster.close()
    for raster in qf_m_rasters:
        raster.close()
    for raster in kc_m_rasters:
        raster.close()


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
    cdef int flow_dir_i
    cdef int flow_dir_nodata
    cdef long raster_x_size, raster_y_size, xs_root, ys_root, xoff, yoff
    cdef int n_dir
    cdef int xs, ys, flow_dir_s, win_xsize, win_ysize
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

    cdef ManagedRaster target_b_sum_raster = ManagedRaster(
        target_b_sum_path.encode('utf-8'), 1, 1)
    cdef ManagedRaster target_b_raster = ManagedRaster(
        target_b_path.encode('utf-8'), 1, 1)
    cdef ManagedRaster l_raster = ManagedRaster(l_path.encode('utf-8'), 1, 0)
    cdef ManagedRaster l_avail_raster = ManagedRaster(l_avail_path.encode('utf-8'), 1, 0)
    cdef ManagedRaster l_sum_raster = ManagedRaster(l_sum_path.encode('utf-8'), 1, 0)
    cdef ManagedFlowDirRaster flow_dir_mfd_raster = ManagedFlowDirRaster(
        flow_dir_mfd_path.encode('utf-8'), 1, 0)
    cdef ManagedRaster stream_raster = ManagedRaster(stream_path.encode('utf-8'), 1, 0)

    current_pixel = 0
    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        for ys in xrange(win_ysize):
            ys_root = yoff + ys
            for xs in xrange(win_xsize):
                xs_root = xoff + xs
                flow_dir_s = <int>flow_dir_mfd_raster.get(xs_root, ys_root)
                if is_close(flow_dir_s, flow_dir_nodata):
                    current_pixel += 1
                    continue

                # search for a pixel that has no downslope neighbors,
                # or whose downslope neighbors all have nodata in the stream raster (?)
                outlet = 1
                dn_neighbors = DownslopeNeighbors(
                    Pixel(flow_dir_mfd_raster, xs_root, ys_root))
                for neighbor in dn_neighbors:
                    stream_val = <int>stream_raster.get(neighbor.x, neighbor.y)
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
                    dn_neighbors = DownslopeNeighbors(Pixel(flow_dir_mfd_raster, xi, yi))
                    for neighbor in dn_neighbors:
                        stream_val = <int>stream_raster.get(neighbor.x, neighbor.y)
                        if stream_val:
                            b_sum_i += neighbor.flow_proportion
                        else:
                            b_sum_j = target_b_sum_raster.get(neighbor.x, neighbor.y)
                            if is_close(b_sum_j, target_nodata):
                                downslope_defined = 0
                                break
                            l_j = l_raster.get(neighbor.x, neighbor.y)
                            l_avail_j = l_avail_raster.get(neighbor.x, neighbor.y)
                            l_sum_j = l_sum_raster.get(neighbor.x, neighbor.y)

                            if l_sum_j != 0 and (l_sum_j - l_j) != 0:
                                b_sum_i += neighbor.flow_proportion * (
                                    (1 - l_avail_j / l_sum_j) * (
                                        b_sum_j / (l_sum_j - l_j)))
                            else:
                                b_sum_i += neighbor.flow_proportion

                    if not downslope_defined:
                        continue

                    l_i = l_raster.get(xi, yi)
                    l_sum_i = l_sum_raster.get(xi, yi)
                    b_sum_i = l_sum_i * b_sum_i

                    if l_sum_i != 0:
                        b_i = max(b_sum_i * l_i / l_sum_i, 0.0)
                    else:
                        b_i = 0.0

                    target_b_raster.set(xi, yi, b_i)
                    target_b_sum_raster.set(xi, yi, b_sum_i)

                    current_pixel += 1
                    up_iterator = UpslopeNeighborIterator(flow_dir_mfd_raster, xi, yi)
                    neighbor = up_iterator.next()
                    while neighbor.direction < 8:
                        work_stack.push(pair[long, long](neighbor.x, neighbor.y))
                        neighbor = up_iterator.next()

    target_b_sum_raster.close()
    target_b_raster.close()
    l_raster.close()
    l_avail_raster.close()
    l_sum_raster.close()
    flow_dir_mfd_raster.close()
    stream_raster.close()
