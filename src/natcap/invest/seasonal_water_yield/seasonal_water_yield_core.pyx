# cython: profile=False

import logging
import os
import collections
import sys
import gc

import numpy
cimport numpy
cimport cython
import osgeo
from osgeo import gdal
from cython.operator cimport dereference as deref
import natcap.invest.scenic_quality.viewshed

from libcpp.set cimport set as c_set
from libcpp.deque cimport deque
from libcpp.map cimport map
from libcpp.stack cimport stack
from libc.math cimport atan
from libc.math cimport atan2
from libc.math cimport tan
from libc.math cimport sqrt
from libc.math cimport ceil

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef int N_MONTHS = 12

cdef double PI = 3.141592653589793238462643383279502884
cdef double INF = numpy.inf
cdef int N_BLOCK_ROWS = 4
cdef int N_BLOCK_COLS = 4


def calculate_local_recharge(
        precip_path_list, et0_path_list, qfm_path_list, flow_dir_path,
        dem_path, lulc_path, alpha_month, beta_i, gamma, stream_path, li_path,
        kc_path_list, li_avail_path, l_sum_avail_path, aet_path):
    """
        # call through to a cython function that does the necessary routing
        # between AET and L.sum.avail in equation [7], [4], and [3]

target_path_list=[
                file_registry['l_path'],
                file_registry['l_avail_path'],
                file_registry['l_sum_avail_path'],
                file_registry['aet_path']],
    """
    wrapper for local recharge so we can statically type outlet lists"""
    LOGGER.error('implement calculate_local_recharge')
    flow_raster = natcap.invest.scenic_quality.viewshed._ManagedRaster(
        flow_dir_path, 1, 0)
    dem_raster = natcap.invest.scenic_quality.viewshed._ManagedRaster(
        dem_path, 1, 0)

    LOGGER.debug(flow_raster)
    """
    cdef deque[int] outlet_cell_deque
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.find_outlets(
        dem_path, flow_dir_path, outlet_cell_deque)
    # convert a dictionary of month -> alpha to an array of alphas in sorted
    # order by month
    cdef numpy.ndarray alpha_month_array = numpy.array(
        [x[1] for x in sorted(alpha_month.iteritems())])
    route_local_recharge(
        precip_path_list, et0_path_list, kc_path_list, li_path,
        li_avail_path, l_sum_avail_path, aet_path, alpha_month_array, beta_i,
        gamma, qfm_path_list, stream_path, outlet_cell_deque)
    """


def route_baseflow_sum(
        dem_path, l_path, l_avail_path, l_sum_path,
        stream_path, b_sum_path):
    LOGGER.error('implement route_baseflow_sum')
    dem_raster = natcap.invest.scenic_quality.viewshed._ManagedRaster(
        dem_path, 1, 0)

    cdef time_t start
    time(&start)
