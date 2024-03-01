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
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport route

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef int N_MONTHS = 12
cdef float TARGET_NODATA = -1e32


def local_recharge_seed_fn(x, y, ManagedFlowDirRaster flow_raster):
    return flow_raster.is_local_high_point(x, y)


def local_recharge_route_fn(x, y, ManagedFlowDirRaster flow_raster, alpha_month_map,
        beta_i, gamma,
        et0_m_raster_list, precip_m_raster_list, qf_m_raster_list, kc_m_raster_list,
        _ManagedRaster target_li_raster, _ManagedRaster target_li_avail_raster,
        _ManagedRaster target_l_sum_avail_raster, _ManagedRaster target_aet_raster,
        _ManagedRaster target_pi_raster):
    cdef double pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i, l_avail_i
    cdef _ManagedRaster et0_m_raster, qf_m_raster, kc_m_raster
    cdef numpy.ndarray[numpy.npy_float32, ndim=1] alpha_month_array = (
        numpy.array(
            [alpha[1] for alpha in sorted(alpha_month_map.items())],
            dtype=numpy.float32))

    cdef bint upslope_defined = True
    cdef float l_sum_avail_i = target_l_sum_avail_raster.get(x, y)

    if not is_close(l_sum_avail_i, target_l_sum_avail_raster.nodata):
        # already defined
        return []

    # Equation 7, calculate L_sum_avail_i if possible, skip
    # otherwise
    # initialize to 0 so we indicate we haven't tracked any
    # mfd values yet
    l_sum_avail_i = 0
    for neighbor in flow_raster.get_upslope_neighbors(x, y):
        # pixel flows inward, check upslope
        l_sum_avail_j = target_l_sum_avail_raster.get(
            neighbor.x, neighbor.y)
        if is_close(l_sum_avail_j, target_l_sum_avail_raster.nodata):
            upslope_defined = False
            break
        l_avail_j = target_li_avail_raster.get(
            neighbor.x, neighbor.y)
        # A step of Equation 7
        l_sum_avail_i += (
            l_sum_avail_j + l_avail_j) * neighbor.flow_proportion
    # calculate l_sum_avail_i by summing all the valid
    # directions then normalizing by the sum of the mfd
    # direction weights (Equation 8)
    if upslope_defined:
        # Equation 7
        target_l_sum_avail_raster.set(x, y, l_sum_avail_i)
    else:
        # if not defined, we'll get it on another pass
        return []

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

        p_m = precip_m_raster.get(x, y)
        if not is_close(p_m, precip_m_raster.nodata):
            p_i += p_m
        else:
            p_m = 0

        qf_m = qf_m_raster.get(x, y)
        if not is_close(qf_m, qf_m_raster.nodata):
            qf_i += qf_m
        else:
            qf_m = 0

        kc_m = kc_m_raster.get(x, y)
        pet_m = 0
        et0_m = et0_m_raster.get(x, y)
        if not (
                is_close(kc_m, kc_m_raster.nodata) or
                is_close(et0_m, et0_m_raster.nodata)):
            # Equation 6
            pet_m = kc_m * et0_m

        # Equation 4/5
        aet_i += min(
            pet_m,
            p_m - qf_m +
            alpha_month_array[m_index]*beta_i*l_sum_avail_i)

    l_i = (p_i - qf_i - aet_i)
    l_avail_i = min(gamma * l_i, l_i)

    target_pi_raster.set(x, y, p_i)
    target_aet_raster.set(x, y, aet_i)
    target_li_raster.set(x, y, l_i)
    target_li_avail_raster.set(x, y, l_avail_i)

    next_pixels = []
    for neighbor in flow_raster.get_downslope_neighbors(x, y):
        next_pixels.append(neighbor.y * flow_raster.raster_x_size + neighbor.x)
    return next_pixels


def calculate_local_recharge(
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
    cdef ManagedFlowDirRaster flow_raster = ManagedFlowDirRaster(
        flow_dir_mfd_path, 1, 0)

    # make sure that user input nodata values are defined
    # set to -1 if not defined
    # precipitation and evapotranspiration data should
    # always be non-negative
    et0_m_raster_list = []
    for et0_path in et0_path_list:
        et0_m_raster = _ManagedRaster(et0_path, 1, 0)
        et0_m_raster_list.append(et0_m_raster)
        if et0_m_raster.nodata is None:
            et0_m_raster.nodata = -1

    precip_m_raster_list = []
    for precip_m_path in precip_path_list:
        precip_m_raster = _ManagedRaster(precip_m_path, 1, 0)
        precip_m_raster_list.append(precip_m_raster)
        if precip_m_raster.nodata is None:
            precip_m_raster.nodata = -1

    qf_m_raster_list = []
    for qf_m_path in qf_m_path_list:
        qf_m_raster_list.append(_ManagedRaster(qf_m_path, 1, 0))

    kc_m_raster_list = []
    for kc_m_path in kc_path_list:
        kc_m_raster_list.append(_ManagedRaster(kc_m_path, 1, 0))

    TARGET_NODATA = -1e32
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_path, gdal.GDT_Float32, [TARGET_NODATA],
        fill_value_list=[TARGET_NODATA])
    cdef _ManagedRaster target_li_raster = _ManagedRaster(
        target_li_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_avail_path, gdal.GDT_Float32,
        [TARGET_NODATA], fill_value_list=[TARGET_NODATA])
    cdef _ManagedRaster target_li_avail_raster = _ManagedRaster(
        target_li_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_l_sum_avail_path, gdal.GDT_Float32,
        [TARGET_NODATA], fill_value_list=[TARGET_NODATA])
    cdef _ManagedRaster target_l_sum_avail_raster = _ManagedRaster(
        target_l_sum_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_aet_path, gdal.GDT_Float32, [TARGET_NODATA],
        fill_value_list=[TARGET_NODATA])
    cdef _ManagedRaster target_aet_raster = _ManagedRaster(
        target_aet_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_pi_path, gdal.GDT_Float32, [TARGET_NODATA],
        fill_value_list=[TARGET_NODATA])
    cdef _ManagedRaster target_pi_raster = _ManagedRaster(
        target_pi_path, 1, 1)

    route(
        flow_dir_path=flow_dir_mfd_path,
        seed_fn=local_recharge_seed_fn,
        route_fn=local_recharge_route_fn,
        seed_fn_args=[flow_raster],
        route_fn_args=[flow_raster, alpha_month_map, beta_i, gamma,
            et0_m_raster_list, precip_m_raster_list, qf_m_raster_list, kc_m_raster_list,
            target_li_raster, target_li_avail_raster,
            target_l_sum_avail_raster, target_aet_raster,
            target_pi_raster])


def baseflow_seed_fn(int x, int y, ManagedFlowDirRaster flow_dir_mfd_raster,
        _ManagedRaster stream_raster):
    """Determine whether the pixel at (x, y) is a seed pixel.

    Args:
        x (int): x coordinate of the pixel
        y (int): y coordinate of the pixel

    Returns:
        True if the pixel qualifies as a seed pixel, False otherwise
    """
    cdef int stream_val
    cdef int flow_dir_s = <int>flow_dir_mfd_raster.get(x, y)
    if is_close(flow_dir_s, flow_dir_mfd_raster.nodata):
        return False

    # search for a pixel that has no downslope neighbors,
    # or whose downslope neighbors all have nodata in the stream raster (?)
    for neighbor in flow_dir_mfd_raster.get_downslope_neighbors(x, y):
        stream_val = <int>stream_raster.get(neighbor.x, neighbor.y)
        if stream_val != stream_raster.nodata:
            return False
    return True


def baseflow_route_fn(x, y, ManagedFlowDirRaster flow_dir_mfd_raster,
        _ManagedRaster l_raster, _ManagedRaster l_avail_raster,
        _ManagedRaster l_sum_raster, _ManagedRaster stream_raster,
        _ManagedRaster target_b_raster, _ManagedRaster target_b_sum_raster):
    cdef int stream_val
    cdef float b_i, l_j, l_avail_j, l_sum_j
    cdef float b_sum_i = target_b_sum_raster.get(x, y)
    cdef bint downslope_defined = True

    next_pixels = []
    if not is_close(b_sum_i, TARGET_NODATA):
        return []

    b_sum_i = 0.0
    for neighbor in flow_dir_mfd_raster.get_downslope_neighbors(x, y):
        stream_val = <int>stream_raster.get(neighbor.x, neighbor.y)
        if stream_val:
            b_sum_i += neighbor.flow_proportion
        else:
            b_sum_j = target_b_sum_raster.get(neighbor.x, neighbor.y)
            if is_close(b_sum_j, TARGET_NODATA):
                downslope_defined = False
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
        return []

    l_i = l_raster.get(x, y)
    l_sum_i = l_sum_raster.get(x, y)
    b_sum_i = l_sum_i * b_sum_i

    if l_sum_i != 0:
        b_i = max(b_sum_i * l_i / l_sum_i, 0.0)
    else:
        b_i = 0.0

    target_b_raster.set(x, y, b_i)
    target_b_sum_raster.set(x, y, b_sum_i)

    for neighbor in flow_dir_mfd_raster.get_upslope_neighbors(x, y):
        next_pixels.append(neighbor.y * flow_dir_mfd_raster.raster_x_size + neighbor.x)
    return next_pixels

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
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_sum_path, gdal.GDT_Float32,
        [TARGET_NODATA], fill_value_list=[TARGET_NODATA])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_path, gdal.GDT_Float32,
        [TARGET_NODATA], fill_value_list=[TARGET_NODATA])

    cdef _ManagedRaster target_b_sum_raster = _ManagedRaster(
        target_b_sum_path, 1, 1)
    cdef _ManagedRaster target_b_raster = _ManagedRaster(
        target_b_path, 1, 1)
    cdef _ManagedRaster l_raster = _ManagedRaster(l_path, 1, 0)
    cdef _ManagedRaster l_avail_raster = _ManagedRaster(l_avail_path, 1, 0)
    cdef _ManagedRaster l_sum_raster = _ManagedRaster(l_sum_path, 1, 0)
    cdef ManagedFlowDirRaster flow_dir_mfd_raster = ManagedFlowDirRaster(
        flow_dir_mfd_path, 1, 0)
    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, 0)

    route(
        flow_dir_path=flow_dir_mfd_path,
        seed_fn=baseflow_seed_fn,
        route_fn=baseflow_route_fn,
        seed_fn_args=[flow_dir_mfd_raster, stream_raster],
        route_fn_args=[
            flow_dir_mfd_raster, l_raster, l_avail_raster, l_sum_raster,
            stream_raster, target_b_raster, target_b_sum_raster])

