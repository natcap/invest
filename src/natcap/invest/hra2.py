"""Habitat risk assessment (HRA) model for InVEST."""
# -*- coding: UTF-8 -*-
import collections
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
            #dimensions of output rasters in meters.
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
    # TODO: is the preprocessing dir actually needed?
    preprocessing_dir = os.path.join(args['workspace_dir'], 'file_preprocessing')
    intermediate_dir = os.path.join(args['workspace_dir'],
                                    'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    taskgraph_working_dir = os.path.join(args['workspace_dir'], '.taskgraph')
    utils.make_directories([intermediate_dir, output_dir])
    suffix = utils.make_suffix_string(args, 'results_suffix')

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    graph = taskgraph.TaskGraph(taskgraph_working_dir, n_workers)

    # parse the info table and get habitats, stressors
    habitats, stressors = _parse_info_table(args['info_table_path'])

    # parse the criteria table to get the composite table
    composite_criteria_table_path = os.path.join(
        intermediate_dir, f'composite_criteria{suffix}.csv')

    #
    # For each habitat and stressor dataset:
    #   * simplify - Nyquist
    #   * rasterize
    #
    # Given rasterized habitats, make a habitat mask
    # Do EDT/buffer/decay on each stressor.
    #
    # Load the composite dataframe
    #   * for each row, calculate the criteria score
    #       * Track as needed (aggregating by habitats) for later summary
    #         rasters
    #
    # Calculate summary operations.

    for row in pandas.read_csv(composite_criteria_table_path):
        if row['stressor'] == 'RESILIENCE':
            decayed_distance_edt_path = None
        else:
            decayed_distance_edt_path = decayed_edts[row['stressor']]

        # calculate criteria score

    for habitat in habitats:
        pass
        # sum stressor criteria per habitat and reclassify
        # sum resilience criteria per habitat and reclassify





# TODO: support Excel and CSV both
def _parse_info_table(info_table_path):
    table = utils.read_csv_to_dataframe(info_table_path, to_lower=True)
    table = table.set_index('name')
    table = table.rename(columns={'stressor buffer (meters)': 'buffer'})

    # Drop the buffer column from the habitats list; we don't need it.
    habitats = table.loc[table['type'] == 'habitat'].drop(
        columns=['type', 'buffer']).to_dict(orient='index')

    # Keep the buffer column in the stressors dataframe.
    stressors = table.loc[table['type'] == 'stressor'].drop(
        columns=['type']).to_dict(orient='index')

    return (habitats, stressors)


# What do I need from this function?
# attributes for each habitat/stressor combination.
#    {habitat: {stressor: [{NAME: criterion, RATING: rating, DQ: dq, WEIGHT:
#                           weight, CRITERIA_TYPE: E/C}]}}
def _parse_criteria_table(criteria_table_path, known_stressors,
                          target_composite_csv_path):

    table = pandas.read_csv(criteria_table_path, header=None,
                            sep=None, engine='python').to_numpy()
    known_stressors = set(known_stressors)

    # Fill in habitat names in the table for easier reference.
    criteria_col = None
    habitats = set()
    for col_index, value in enumerate(table[0]):
        if value == 'HABITAT NAME':
            continue
        if value == 'CRITERIA TYPE':
            criteria_col = col_index
            break  # We're done with habitats
        if not isinstance(value, str):
            value = table[0][col_index-1]  # Fill in from column to the left
            habitats.add(value)
            table[0][col_index] = value

    habitat_columns = collections.defaultdict(list)
    for col_index, col_value in enumerate(table[0]):
        if col_value in habitats:
            habitat_columns[col_value].append(col_index)

    # the primary key of this table is (habitat_stressor, criterion)
    overlap_df = pandas.DataFrame(columns=['habitat', 'stressor', 'criterion',
                                           'rating', 'dq', 'weight', 'e/c'])

    current_stressor = None
    for row_index, row in enumerate(table[1:], start=1):
        if row[0] == 'HABITAT STRESSOR OVERLAP PROPERTIES':
            continue

        if (row[0] in known_stressors or
                row[0] == 'HABITAT RESILIENCE ATTRIBUTES'):
            if row[0] == 'HABITAT RESILIENCE ATTRIBUTES':
                row[0] = 'RESILIENCE'  # Shorten for convenience
            current_stressor = row[0]
            current_stressor_header_row = row_index
            continue  # can skip this row

        try:
            if numpy.all(numpy.isnan(row.astype(numpy.float32))):
                continue
        except (TypeError, ValueError):
            # Either of these exceptions are thrown when there are string types
            # in the row
            pass

        # {habitat: {rating/dq/weight/ec dict}
        stressor_habitat_data = {
            'stressor': current_stressor,
            'criterion': row[0],
            'e/c': row[criteria_col],
        }
        for habitat, habitat_col_indices in habitat_columns.items():
            stressor_habitat_data['habitat'] = habitat
            for col_index in habitat_col_indices:
                # attribute is rating, dq or weight
                attribute_name = table[current_stressor_header_row][col_index]
                attribute_value = row[col_index]
                stressor_habitat_data[
                    attribute_name.lower()] = attribute_value
            overlap_df = overlap_df.append(
                stressor_habitat_data, ignore_index=True)

    # TODO: figure out if we need a special case for the resilience data.
    overlap_df.to_csv(target_composite_csv_path, index=False)


def _calculate_decayed_distance(stressor_raster_path, decay_type,
                                buffer_distance, target_edt_path):
    # TODO: ensure we're working with square pixels in our raster stack
    pygeoprocessing.distance_transform_edt((stressor_raster_path, 1),
                                           target_edt_path)
    pixel_size = abs(pygeoprocessing.get_raster_info(
        stressor_raster_path)['pixel_size'][0])
    buffer_distance_in_pixels = buffer_distance / pixel_size

    target_edt_raster = gdal.OpenEx(target_edt_path, gdal.GA_Update)
    target_edt_band = target_edt_raster.GetRasterBand(1)
    edt_nodata = target_edt_band.GetNoDataValue()
    for block_info in pygeoprocessing.iterblocks((target_edt_path, 1),
                                                 offset_only=True):
        source_edt_block = target_edt_band.ReadAsArray(**block_info)

        # The pygeoprocessing target datatype for EDT is a float32
        decayed_edt = numpy.full(source_edt_block.shape, 0,
                                 dtype=numpy.float32)

        # The only valid pixels here are those that are within the buffer
        # distance and also valid in the source edt.
        pixels_within_buffer = (source_edt_block < buffer_distance_in_pixels)
        nodata_pixels = utils.array_equals_nodata(source_edt_block, edt_nodata)
        valid_pixels = (~nodata_pixels & pixels_within_buffer)

        if decay_type == 'linear':
            decayed_edt[valid_pixels] = (
                1 - (source_edt_block[valid_pixels] /
                     buffer_distance_in_pixels))
        elif decay_type == 'exponential':
            decayed_edt[valid_pixels] = numpy.exp(
                -source_edt_block[valid_pixels])
        else:
            raise AssertionError('Invalid decay type provided.')

        # Any values less than 1e-6 are numerically noise and should be 0.
        # Mostly useful for exponential decay, but should also apply to
        # linear decay.
        numpy.where(decayed_edt[valid_pixels] < 1e-6,
                    0.0,
                    decayed_edt[valid_pixels])

        # Reset any nodata pixels that were in the original block.
        decayed_edt[nodata_pixels] = edt_nodata

        target_edt_band.WriteArray(decayed_edt,
                                   xoff=block_info['xoff'],
                                   yoff=block_info['yoff'])

    target_edt_band = None
    target_edt_raster = None


# This is to calculate E or C for a single habitat/stressor pair.
def _calc_criteria(attributes_list, habitat_mask_raster_path,
                   target_criterion_path,
                   decayed_edt_raster_path=None):
    # Assume attributes_list is structured like so:
    #  [{"rating": int/path, "dq": int, "weight": int}, ... ]
    #
    # Stressor weighted distance raster path is the decayed, thresholded
    # stressor raster.
    #
    # Resilience scores are calculated with the same numerical method, but they
    # don't use the decayed EDT.

    pygeoprocessing.new_raster_from_base(
        habitat_mask_raster_path, target_criterion_path, _TARGET_PIXEL_FLT,
        [_TARGET_NODATA_FLT])

    habitat_mask_raster = gdal.OpenEx(habitat_mask_raster_path)
    habitat_band = habitat_mask_raster.GetRasterBand(1)

    target_criterion_raster = gdal.OpenEx(target_criterion_path,
                                          gdal.GA_Update)
    target_criterion_band = target_criterion_raster.GetRasterBand(1)

    if decayed_edt_raster_path:
        decayed_edt_raster = gdal.OpenEx(
            decayed_edt_raster_path, gdal.GA_Update)
        decayed_edt_band = decayed_edt_raster.GetRasterBand(1)

    for block_info in pygeoprocessing.iterblocks((habitat_mask_raster_path, 1),
                                                 offset_only=True):
        habitat_mask = habitat_band.ReadAsArray(**block_info)
        valid_mask = (habitat_mask == 1)

        criterion_score = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLT,
                                     dtype=numpy.float32)
        numerator = numpy.zeros(habitat_mask.shape, dtype=numpy.float32)
        denominator = numpy.zeros(habitat_mask.shape, dtype=numpy.float32)
        for attribute_dict in attributes_list:
            # TODO: a rating of 0 means that the criterion should be ignored
            #  THIS MEANS IGNORING IN BOTH NUMERATOR AND DENOMINATOR
            # RATING may be either a number or a raster.
            try:
                rating = float(attribute_dict['rating'])
                if rating == 0:
                    continue
            except ValueError:
                # When rating is a string filepath, it represents a raster.
                try:
                    rating_raster = gdal.OpenEx(attribute_dict['rating'])
                    rating_band = rating_raster.GetRasterBand(1)
                    rating = rating_band.ReadAsArray(**block_info)[valid_mask]
                finally:
                    rating_band = None
                    rating_raster = None
            data_quality = attribute_dict['data_quality']
            weight = attribute_dict['weight']

            # The (data_quality + weight) denominator is duplicated here
            # because it's easier to read this way and per the docs is
            # guaranteed to be a number and not a raster.
            numerator[valid_mask] += (rating / (data_quality * weight))
            denominator[valid_mask] += (1 / (data_quality * weight))

        # This is not clearly documented in the UG, but in the source code of
        # previous (3.3.1, 3.10.2) versions of HRA, the numerator is multiplied
        # by the stressor's weighted distance raster.
        # This will give highest values to pixels that overlap stressors and
        # decaying values further away from the overlapping pixels.
        if decayed_edt_raster_path:
            numerator[valid_mask] *= decayed_edt_band.ReadAsArray(
                **block_info)[valid_mask]
        criterion_score[valid_mask] = (
            numerator[valid_mask] / denominator[valid_mask])

        target_criterion_band.WriteArray(criterion_score,
                                         xoff=block_info['xoff'],
                                         yoff=block_info['yoff'])

    target_criterion_band = None
    target_criterion_raster = None


def _calculate_pairwise_risk(habitat_mask_raster_path, exposure_raster_path,
                             consequence_raster_path, risk_equation,
                             target_risk_raster_path):
    def _muliplicative_risk(habitat_mask, exposure, consequence):
        habitat_pixels = (habitat_mask == 1)
        risk_array = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLT,
                                dtype=numpy.float32)
        risk_array[habitat_pixels] = (
            exposure[habitat_pixels] * consequence[habitat_pixels])
        return risk_array

    def _euclidean_risk(habitat_mask, exposure, consequence):
        habitat_pixels = (habitat_mask == 1)
        risk_array = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLT,
                                dtype=numpy.float32)
        risk_array[habitat_pixels] = numpy.sqrt(
            (exposure[habitat_pixels] - 1) ** 2 +
            (consequence[habitat_pixels] - 1) ** 2)

    if risk_equation == 'multiplicative':
        risk_op = _muliplicative_risk
    elif risk_equation == 'euclidean':
        risk_op = _euclidean_risk
    else:
        raise AssertionError('Invalid risk equation provided')

    pygeoprocessing.raster_calculator(
        [(habitat_mask_raster_path, 1),
         (exposure_raster_path, 1),
         (consequence_raster_path, 1)],
        risk_op, target_risk_raster_path, _TARGET_PIXEL_FLT,
        _TARGET_NODATA_FLT)


# max pairwise risk is calculated based on user input and choice of risk
# equation.  Might as well pass in the numeric value rather than the risk
# equation type.
def _reclassify_pairwise_risk(habitat_mask, max_pairwise_risk, pairwise_risk):
    habitat_pixels = (habitat_mask == 1)
    reclassified = numpy.full(habitat_mask.shape, _TARGET_NODATA_INT,
                              dtype=numpy.uint8)
    reclassified[habitat_pixels] = numpy.digitize(
        pairwise_risk[habitat_pixels],
        [0, max_pairwise_risk*(1/3), max_pairwise_risk*(2/3)],
        right=True)  # bins[i-1] >= x > bins[i]
    return reclassified

# TODO: to sum rasters, use natcap.invest.ndr._sum_rasters
