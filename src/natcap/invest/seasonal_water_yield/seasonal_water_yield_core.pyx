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

from libcpp.vector cimport vector

from ..managed_raster.managed_raster cimport D8, MFD
from .swy cimport run_route_baseflow_sum, run_calculate_local_recharge

LOGGER = logging.getLogger(__name__)

cpdef calculate_local_recharge(
        precip_path_list, et0_path_list, qf_m_path_list, flow_dir_mfd_path,
        kc_path_list, alpha_month_map, float beta_i, float gamma, stream_path,
        target_li_path, target_li_avail_path, target_l_sum_avail_path,
        target_aet_path, target_pi_path, algorithm):
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
    cdef vector[float] alpha_values
    cdef vector[char*] et0_paths
    cdef vector[char*] precip_paths
    cdef vector[char*] qf_paths
    cdef vector[char*] kc_paths
    encoded_et0_paths = [p.encode('utf-8') for p in et0_path_list]
    encoded_precip_paths = [p.encode('utf-8') for p in precip_path_list]
    encoded_qf_paths = [p.encode('utf-8') for p in qf_m_path_list]
    encoded_kc_paths = [p.encode('utf-8') for p in kc_path_list]
    for i in range(12):
        et0_paths.push_back(encoded_et0_paths[i])
        precip_paths.push_back(encoded_precip_paths[i])
        qf_paths.push_back(encoded_qf_paths[i])
        kc_paths.push_back(encoded_kc_paths[i])
        alpha_values.push_back(alpha_month_map[i + 1])

    target_nodata = -1e32
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_l_sum_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_aet_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_pi_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    args = [
        precip_paths,
        et0_paths,
        qf_paths,
        flow_dir_mfd_path.encode('utf-8'),
        kc_paths,
        alpha_values,
        beta_i,
        gamma,
        stream_path.encode('utf-8'),
        target_li_path.encode('utf-8'),
        target_li_avail_path.encode('utf-8'),
        target_l_sum_avail_path.encode('utf-8'),
        target_aet_path.encode('utf-8'),
        target_pi_path.encode('utf-8')]

    if algorithm.lower() == 'mfd':
        run_calculate_local_recharge[MFD](*args)
    else:  # D8
        run_calculate_local_recharge[D8](*args)


def route_baseflow_sum(
        flow_dir_path, l_path, l_avail_path, l_sum_path,
        stream_path, target_b_path, target_b_sum_path, algorithm):
    """Route Baseflow through MFD as described in Equation 11.

    Args:
        flow_dir_path (string): path to a pygeoprocessing multiple flow
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
    cdef float target_nodata = -1e32

    pygeoprocessing.new_raster_from_base(
        flow_dir_path, target_b_sum_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_path, target_b_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])

    if algorithm.lower() == 'mfd':
        run_route_baseflow_sum[MFD](
            flow_dir_path.encode('utf-8'),
            l_path.encode('utf-8'),
            l_avail_path.encode('utf-8'),
            l_sum_path.encode('utf-8'),
            stream_path.encode('utf-8'),
            target_b_path.encode('utf-8'),
            target_b_sum_path.encode('utf-8'))
    else:  # D8
        run_route_baseflow_sum[D8](
            flow_dir_path.encode('utf-8'),
            l_path.encode('utf-8'),
            l_avail_path.encode('utf-8'),
            l_sum_path.encode('utf-8'),
            stream_path.encode('utf-8'),
            target_b_path.encode('utf-8'),
            target_b_sum_path.encode('utf-8'))
