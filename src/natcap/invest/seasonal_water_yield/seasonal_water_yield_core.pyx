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
from libc.time cimport time as ctime
from ..managed_raster.managed_raster cimport ManagedRaster
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport DownslopeNeighborIterator
from ..managed_raster.managed_raster cimport UpslopeNeighborIterator
from ..managed_raster.managed_raster cimport NeighborTuple
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport D8, MFD

from .swy cimport run_route_baseflow_sum, run_calculate_local_recharge

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

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
    cdef vector[float] alpha_values;
    for x in sorted(alpha_month_map.items()):
        alpha_values.push_back(x[1])

    # make sure that user input nodata values are defined
    # set to -1 if not defined
    # precipitation and evapotranspiration data should
    # always be non-negative
    cdef vector[char*] et0_paths
    for et0_path in et0_path_list:
        et0_paths.push_back(et0_path.encode('utf-8'))

    cdef vector[char*] precip_paths
    for precip_path in precip_path_list:
        precip_paths.push_back(precip_path.encode('utf-8'))

    cdef vector[char*] qf_paths
    for qf_path in qf_m_path_list:
        qf_paths.push_back(qf_path.encode('utf-8'))

    cdef vector[char*] kc_paths
    for kc_path in kc_path_list:
        kc_paths.push_back(kc_path.encode('utf-8'))

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

    run_calculate_local_recharge(
        precip_paths,
        et0_paths,
        qf_paths,
        flow_dir_mfd_path,
        kc_paths,
        alpha_values,
        beta_i,
        gamma,
        stream_path.encode('utf-8'),
        target_li_path.encode('utf-8'),
        target_li_avail_path.encode('utf-8'),
        target_l_sum_avail_path.encode('utf-8'),
        target_aet_path.encode('utf-8'),
        target_pi_path.encode('utf-8'))


def route_baseflow_sum(
        flow_dir_path, l_path, l_avail_path, l_sum_path,
        stream_path, target_b_path, target_b_sum_path):
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

    run_route_baseflow_sum[MFD](
        flow_dir_path.encode('utf-8'),
        l_path.encode('utf-8'),
        l_avail_path.encode('utf-8'),
        l_sum_path.encode('utf-8'),
        stream_path.encode('utf-8'),
        target_b_path.encode('utf-8'),
        target_b_sum_path.encode('utf-8'))
