"""Habitat risk assessment (HRA) model for InVEST."""
# -*- coding: UTF-8 -*-
import json
import logging
import math
import os
import pickle
import shutil
import tempfile

import numpy
import pandas
import pygeoprocessing
import shapely.ops
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import MODEL_METADATA
from . import spec_utils
from . import utils
from . import validation
from .spec_utils import u

LOGGER = logging.getLogger('natcap.invest.hra')

# Parameters from the user-provided criteria and info tables
_BUFFER_HEADER = 'STRESSOR BUFFER (METERS)'
_CRITERIA_TYPE_HEADER = 'CRITERIA TYPE'
_HABITAT_NAME_HEADER = 'HABITAT NAME'
_HABITAT_RESILIENCE_HEADER = 'HABITAT RESILIENCE ATTRIBUTES'
_HABITAT_STRESSOR_OVERLAP_HEADER = 'HABITAT STRESSOR OVERLAP PROPERTIES'
_SPATIAL_CRITERIA_TYPE = 'spatial_criteria'
_HABITAT_TYPE = 'habitat'
_STRESSOR_TYPE = 'stressor'
_SUBREGION_FIELD_NAME = 'name'
_WEIGHT_KEY = 'Weight'
_DQ_KEY = 'DQ'

# Parameters to be used in dataframe and output stats CSV
_HABITAT_HEADER = 'HABITAT'
_STRESSOR_HEADER = 'STRESSOR'
_TOTAL_REGION_NAME = 'Total Region'

# Parameters for the spatially explicit criteria shapefiles
_RATING_FIELD = 'rating'

# A cutoff for the decay amount after which we will say scores are equivalent
# to 0, since we don't want to have values outside the buffer zone.
_EXP_DEDAY_CUTOFF = 1E-6

# Target cell type or values for raster files.
_TARGET_PIXEL_FLT = gdal.GDT_Float32
_TARGET_PIXEL_INT = gdal.GDT_Byte
_TARGET_NODATA_FLT = float(numpy.finfo(numpy.float32).min)
_TARGET_NODATA_INT = 255  # for unsigned 8-bit int

# ESPG code for warping rasters to WGS84 coordinate system.
_WGS84_ESPG_CODE = 4326

# Resampling method for rasters.
_RESAMPLE_METHOD = 'near'

# An argument list that will be passed to the GTiff driver. Useful for
# blocksizes, compression, and more.
_DEFAULT_GTIFF_CREATION_OPTIONS = (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256')

ARGS_SPEC = {
    "model_name": MODEL_METADATA["habitat_risk_assessment"].model_title,
    "pyname": MODEL_METADATA["habitat_risk_assessment"].pyname,
    "userguide_html": MODEL_METADATA["habitat_risk_assessment"].userguide,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "info_table_path": {
            "name": _("habitat stressor table"),
            "about": _("A table describing each habitat and stressor."),
            "type": "csv",
            "columns": {
                "name": {
                    "type": "freestyle_string",
                    "about": _(
                        "A unique name for each habitat or stressor. These "
                        "names must match the habitat and stressor names in "
                        "the Criteria Scores Table.")},
                "path": {
                    "type": {"vector", "raster"},
                    "bands": {1: {
                        "type": "number",
                        "units": u.none,
                        "about": _(
                            "Pixel values are 1, indicating presence of the "
                            "habitat/stressor, or 0 indicating absence. Any "
                            "values besides 0 or 1 will be treated as 0.")
                    }},
                    "fields": {},
                    "geometries": spec_utils.POLYGONS,
                    "about": _(
                        "Map of where the habitat or stressor exists. For "
                        "rasters, a pixel value of 1 indicates presence of "
                        "the habitat or stressor. 0 (or any other value) "
                        "indicates absence of the habitat or stressor. For "
                        "vectors, a polygon indicates an area where the "
                        "habitat or stressor is present.")
                },
                "type": {
                    "type": "option_string",
                    "options": {
                        "habitat": {"description": _("habitat")},
                        "stressor": {"description": _("stressor")}
                    },
                    "about": _(
                        "Whether this row is for a habitat or a stressor.")
                },
                "stressor buffer (meters)": {
                    "type": "number",
                    "units": u.meter,
                    "about": _(
                        "The desired buffer distance used to expand a given "
                        "stressor’s influence or footprint. This should be "
                        "left blank for habitats, but must be filled in for "
                        "stressors. Enter 0 if no buffering is desired for a "
                        "given stressor. The model will round down this "
                        "buffer distance to the nearest cell unit. e.g., a "
                        "buffer distance of 600m will buffer a stressor’s "
                        "footprint by two grid cells if the resolution of "
                        "analysis is 250m.")
                }
            },
            "excel_ok": True
        },
        "criteria_table_path": {
            "name": _("criteria scores table"),
            "about": _(
                "A table of criteria scores for all habitats and stressors."),
            "type": "csv",
            "excel_ok": True,
        },
        "resolution": {
            "name": _("resolution of analysis"),
            "about": _(
                "The resolution at which to run the analysis. The model "
                "outputs will have this resolution."),
            "type": "number",
            "units": u.meter,
            "expression": "value > 0",
        },
        "max_rating": {
            "name": _("maximum criteria score"),
            "about": _(
                "The highest possible criteria score in the scoring system."),
            "type": "number",
            "units": u.none,
            "expression": "value > 0"
        },
        "risk_eq": {
            "name": _("risk equation"),
            "about": _(
                "The equation to use to calculate risk from exposure and "
                "consequence."),
            "type": "option_string",
            "options": {
                "Multiplicative": {"display_name": _("multiplicative")},
                "Euclidean": {"display_name": _("Euclidean")}
            }
        },
        "decay_eq": {
            "name": _("decay equation"),
            "about": _(
                "The equation to model effects of stressors in buffer areas."),
            "type": "option_string",
            "options": {
                "None": {
                    "display_name": _("none"),
                    "description": _(
                        "No decay. Stressor has full effect in the buffer "
                        "area.")},
                "Linear": {
                    "display_name": _("linear"),
                    "description": _(
                        "Stressor effects in the buffer area decay linearly "
                        "with distance from the stressor.")},
                "Exponential": {
                    "display_name": _("exponential"),
                    "description": _(
                        "Stressor effects in the buffer area decay "
                        "exponentially with distance from the stressor.")}
            }
        },
        "aoi_vector_path": {
            **spec_utils.AOI,
            "projected": True,
            "projection_units": u.meter,
            "fields": {
                "name": {
                    "required": False,
                    "type": "freestyle_string",
                    "about": _(
                        "Uniquely identifies each feature. Required if "
                        "the vector contains more than one feature.")
                }
            },
            "about": _(
                "A GDAL-supported vector file containing feature containing "
                "one or more planning regions or subregions."),
        },
        "override_max_overlapping_stressors": {
            "name": _("Override Max Number of Overlapping Stressors"),
            "type": "number",
            "required": False,
            "about": _(
                "If provided, this number will be used in risk "
                "reclassification instead of the calculated number of "
                "stressor layers that overlap."),
            "units": u.none,
            "expression": "value > 0",
        },
        "visualize_outputs": {
            "name": _("Generate GeoJSONs"),
            "about": _("Generate GeoJSON outputs for web visualization."),
            "type": "boolean"
        }
    }
}

_VALID_RISK_EQS = set(ARGS_SPEC['args']['risk_eq']['options'].keys())


def execute(args):
    """Habitat Risk Assessment.

    Args:
        args['workspace_dir'] (str): a path to the output workspace folder.
            It will overwrite any files that exist if the path already exists.
        args['results_suffix'] (str): a string appended to each output file
            path. (optional)
        args['info_table_path'] (str): a path to the CSV or Excel file that
            contains the name of the habitat (H) or stressor (s) on the
            ``NAME`` column that matches the names in criteria_table_path.
            Each H/S has its corresponding vector or raster path on the
            ``PATH`` column. The ``STRESSOR BUFFER (meters)`` column should
            have a buffer value if the ``TYPE`` column is a stressor.
        args['criteria_table_path'] (str): a path to the CSV or Excel file that
            contains the set of criteria ranking of each stressor on each
            habitat.
        args['resolution'] (int): a number representing the desired pixel
            dimensions of output rasters in meters.
        args['max_rating'] (str, int or float): a number representing the
            highest potential value that should be represented in rating in the
            criteria scores table.
        args['risk_eq'] (str): a string identifying the equation that should be
            used in calculating risk scores for each H-S overlap cell. This
            will be either 'Euclidean' or 'Multiplicative'.
        args['decay_eq'] (str): a string identifying the equation that should
            be used in calculating the decay of stressor buffer influence. This
            can be 'None', 'Linear', or 'Exponential'.
        args['aoi_vector_path'] (str): a path to the shapefile containing one
            or more planning regions used to get the average risk value for
            each habitat-stressor combination over each area. Optionally, if
            each of the shapefile features contain a 'name' field, it will
            be used as a way of identifying each individual shape.
        args['override_max_overlapping_stressors'] (number): If provided, this
            number will be used in risk reclassification instead of the
            calculated maximum number of stressor layers that overlap.
        args['n_workers'] (int): the number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process. (optional)
        args['visualize_outputs'] (bool): if True, create output GeoJSONs and
            save them in a visualization_outputs folder, so users can visualize
            results on the web app. Default to True if not specified.
            (optional)

    Returns:
        None.

    """
    preprocessing_dir = os.path.join(args['workspace_dir'], 'file_preprocessing')
    intermediate_dir = os.path.join(args['workspace_dir'],
                                    'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])

    # parse the tables. info table, criteria table
