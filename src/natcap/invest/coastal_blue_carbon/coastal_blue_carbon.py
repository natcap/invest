"""Coastal Blue Carbon Model."""

import logging
import os
import collections
import pprint as pp

import gdal
import pygeoprocessing as pygeo
import numpy as np

from natcap.invest.coastal_blue_carbon.utilities import io
from natcap.invest.coastal_blue_carbon.utilities.cbc_model_classes import \
    CBCModelRun

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.coastal_blue_carbon.coastal_blue_carbon')

# Global Variables
NODATA_FLOAT = -16777216
NODATA_INT = -9999
HA_PER_M2 = 0.0001


def execute(args):
    """Entry point for Coastal Blue Carbon model.

    :param str args['workspace']: location into which all intermediate
        and output files should be placed.

    :param str args['results_suffix']: a string to append to output filenames.

    :param str args['lulc_lookup_uri']: filepath to a CSV table used to convert
        the lulc code to a name. Also used to determine if a given lulc type is
        a coastal blue carbon habitat.

    :param str args['lulc_transition_uri']:

    :param str args['lulc_snapshot_list']:

    :param str args['lulc_snapshot_years_list']:

    :param int args['analysis_year']:

    :param str args['carbon_pool_initial_uri']:

    :param str args['carbon_pool_transient_uri']:

    Example Args::

        args = {
            'workspace': 'path/to/workspace',
            'results_suffix': '',
            'lulc_lookup_uri': 'path/to/lulc_lookup_uri',
            'lulc_transition_uri': 'path/to/lulc_transition_uri',
            'lulc_snapshot_list': [raster1_uri, raster2_uri, ...],
            'lulc_snapshot_years_list': [2000, 2005, ...],
            'analysis_year': 2100,
            'carbon_pool_initial_uri': 'path/to/carbon_pool_initial_uri',
            'carbon_pool_transient_uri': 'path/to/carbon_pool_transient_uri'
        }
    """
    # Get Inputs
    vars_dict = io.get_inputs(args)

    # Run Model
    r = CBCModelRun(vars_dict)
    r.run()
