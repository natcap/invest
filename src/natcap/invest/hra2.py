"""Habitat risk assessment (HRA) model for InVEST."""
# -*- coding: UTF-8 -*-
import collections
import itertools
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
from .ndr import ndr
from .spec_utils import u

LOGGER = logging.getLogger(__name__)

# Parameters to be used in dataframe and output stats CSV
_HABITAT_HEADER = 'HABITAT'
_STRESSOR_HEADER = 'STRESSOR'
_TOTAL_REGION_NAME = 'Total Region'

# Parameters for the spatially explicit criteria shapefiles
_RATING_FIELD = 'rating'

# Target cell type or values for raster files.
_TARGET_GDAL_TYPE_FLOAT32 = gdal.GDT_Float32
_TARGET_GDAL_TYPE_BYTE = gdal.GDT_Byte
_TARGET_NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
_TARGET_NODATA_BYTE = 255  # for unsigned 8-bit int

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
    intermediate_dir = os.path.join(args['workspace_dir'],
                                    'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    taskgraph_working_dir = os.path.join(args['workspace_dir'], '.taskgraph')
    utils.make_directories([intermediate_dir, output_dir])
    suffix = utils.make_suffix_string(args, 'results_suffix')

    resolution = float(args['resolution'])
    max_rating = float(args['max_rating'])
    max_n_stressors = float(args['override_max_overlapping_stressors'])
    target_srs_wkt = pygeoprocessing.get_vector_info(
        args['aoi_vector_path'])['projection_wkt']

    if args['risk_eq'].lower() == 'multiplicative':
        max_pairwise_risk = max_rating * max_rating
    elif args['risk_eq'].lower() == 'euclidean':
        max_pairwise_risk = math.sqrt(
            ((max_rating - 1) ** 2) + ((max_rating - 1) ** 2))
    else:
        raise ValueError(
            "args['risk_eq'] must be either 'Multiplicative' or 'Euclidean' "
            f"not {args['risk_eq']}")

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
    _parse_criteria_table(
        args['criteria_table_path'], set(stressors.keys()),
        composite_criteria_table_path)

    # Preprocess habitat and stressor datasets.
    # All of these are spatial in nature but might be rasters or vectors.
    alignment_source_raster_paths = {}
    alignment_source_vector_paths = {}
    aligned_raster_paths = []
    alignment_dependent_tasks = []
    aligned_habitat_raster_paths = []
    aligned_stressor_raster_paths = {}
    for name, attributes in itertools.chain(habitats.items(),
                                            stressors.items()):
        source_filepath = attributes['path']
        gis_type = pygeoprocessing.get_gis_type(source_filepath)
        aligned_raster_path = os.path.join(
            intermediate_dir, f'aligned_{name}{suffix}.tif')

        # If the input is already a raster, run it through raster_calculator to
        # ensure we know the nodata value and pixel values.
        if gis_type == pygeoprocessing.RASTER_TYPE:
            rewritten_raster_path = os.path.join(
                intermediate_dir, 'rewritten_{name}{suffix}.tif')
            alignment_source_raster_paths[
                rewritten_raster_path] = aligned_raster_path
            alignment_dependent_tasks.append(graph.add_task(
                func=_prep_input_raster,
                kwargs={
                    'source_raster_path': source_filepath,
                    'target_filepath': rewritten_raster_path,
                },
                task_name=f'Rewrite {name} raster for consistency',
                target_path_list=rewritten_raster_path,
                dependent_task_list=[]
            ))

        # If the input is a vector, reproject to the AOI SRS and simplify.
        # Rasterization happens in the alignment step.
        elif gis_type == pygeoprocessing.VECTOR_TYPE:
            # Using Shapefile here because its driver appears to not raise a
            # warning if a MultiPolygon geometry is inserted into a Polygon
            # layer, which was happening on a real-world sample dataset while
            # in development.
            target_reprojected_vector = os.path.join(
                intermediate_dir, f'reprojected_{name}{suffix}.shp')
            reprojected_vector_task = graph.add_task(
                pygeoprocessing.reproject_vector,
                kwargs={
                    'base_vector_path': source_filepath,
                    'target_projection_wkt': target_srs_wkt,
                    'target_path': target_reprojected_vector,
                },
                task_name=f'Reproject {name} to AOI',
                target_path_list=[target_reprojected_vector],
                dependent_task_list=[]
            )

            target_simplified_vector = os.path.join(
                intermediate_dir, f'simplified_{name}{suffix}.gpkg')
            alignment_source_vector_paths[
                target_simplified_vector] = aligned_raster_path
            alignment_dependent_tasks.append(graph.add_task(
                func=_simplify,
                kwargs={
                    'source_vector_path': source_filepath,
                    'tolerance': resolution / 2,  # by the nyquist theorem.
                    'target_vector_path': target_simplified_vector,
                },
                task_name=f'Simplify {name}',
                target_path_list=[target_simplified_vector],
                dependent_task_list=[reprojected_vector_task]
            ))

        # Later operations make use of the habitats rasters or the stressors
        # rasters, so it's useful to collect those here now.
        if name in habitats:
            aligned_habitat_raster_paths.append(aligned_raster_path)
        else:  # must be a stressor
            aligned_stressor_raster_paths[name] = aligned_raster_path

    alignment_task = graph.add_task(
        func=_align,
        kwargs={
            'raster_path_map': alignment_source_raster_paths,
            'vector_path_map': alignment_source_vector_paths,
            'target_pixel_size': (resolution, -resolution),
            'target_srs_wkt': target_srs_wkt,
        },
        task_name='Align raster stack',
        target_path_list=list(itertools.chain(
            alignment_source_raster_paths.values(),
            alignment_source_vector_paths.values())),
        dependent_task_list=alignment_dependent_tasks,
    )

    # --> Create a binary mask of habitat pixels.
    habitat_mask_path = os.path.join(
        intermediate_dir, f'habitat_mask{suffix}.tif')
    habitat_mask_task = graph.add_task(
        pygeoprocessing.raster_calculator,
        kwargs={
            'base_raster_path_band_const_list': [
                (path, 1) for path in aligned_habitat_raster_paths],
            'local_op': _habitat_mask_op,
            'target_raster_path': habitat_mask_path,
            'datatype_target': _TARGET_GDAL_TYPE_BYTE,
            'nodata_target': _TARGET_NODATA_BYTE,
        },
        task_name='Create habitat mask',
        target_path_list=[habitat_mask_path],
        dependent_task_list=[alignment_task]
    )

    # --> for stressor in stressors, do a decayed EDT.
    decayed_edt_paths = {}  # {stressor: decayed EDT raster}
    decayed_edt_tasks = {}  # {stressor: decayed EDT task}
    for stressor, stressor_path in aligned_stressor_raster_paths.items():
        decayed_edt_paths[stressor] = os.path.join(
            intermediate_dir, f'decayed_edt_{stressor}{suffix}.tif')
        decayed_edt_tasks[stressor] = graph.add_task(
            _calculate_decayed_distance,
            kwargs={
                'stressor_raster_path': stressor_path,
                'decay_type': args['decay_eq'],
                'buffer_distance': stressors[stressor]['buffer'],
                'target_edt_path': decayed_edt_paths[stressor],
            },
            task_name=f'Make decayed EDT for {stressor}',
            target_path_list=[decayed_edt_paths[stressor]],
            dependent_task_list=[alignment_task]
        )

    criteria_df = pandas.read_csv(composite_criteria_table_path)
    cumulative_risk_to_habitat_paths = []
    cumulative_risk_to_habitat_tasks = []
    pairwise_summary_data = []  # for the later summary statistics.
    for habitat in habitats:
        pairwise_risk_tasks = []
        pairwise_risk_paths = []
        reclassified_pairwise_risk_paths = []
        reclassified_pairwise_risk_tasks = []

        for stressor in stressors:
            criteria_tasks = {}  # {criteria type: task}
            criteria_rasters = {}  # {criteria type: score raster path}
            summary_data = {
                'habitat': habitat,
                'stressor': stressor,
            }

            for criteria_type in ['E', 'C']:
                criteria_raster_path = os.path.join(
                    intermediate_dir,
                    f'{habitat}_{stressor}_{criteria_type}_score{suffix}.tif')
                criteria_rasters[criteria_type] = criteria_raster_path
                summary_data[
                    f'{criteria_type.lower()}_path'] = criteria_raster_path

                # This rather complicated filter just grabs the rows matching
                # this habitat, stressor and criteria type.  It's the pandas
                # equivalent of SELECT * FROM criteria_df WHERE the habitat,
                # stressor and criteria type match.
                local_criteria_df = criteria_df[
                    (criteria_df['habitat'] == habitat) &
                    (criteria_df['stressor'] == stressor) &
                    (criteria_df['e/c'] == criteria_type)]

                # This produces a list of dicts in the form:
                # [{'rating': (score), 'weight': (score), 'dq': (score)}],
                # which is what _cal_criteria() expects.
                attributes_list = local_criteria_df[
                    ['rating', 'weight', 'dq']].to_dict(orient='records')

                criteria_tasks[criteria_type] = graph.add_task(
                    _calc_criteria,
                    kwargs={
                        'attributes_list': attributes_list,
                        'habitat_mask_raster_path': habitat_mask_path,
                        'target_criterion_path':
                            criteria_rasters[criteria_type],
                        'decayed_edt_raster_path':
                            decayed_edt_paths[stressor],
                    },
                    task_name=(
                        f'Calculate {criteria_type} score for '
                        f'{habitat} / {stressor}'),
                    target_path_list=[criteria_rasters[criteria_type]],
                    dependent_task_list=[
                        decayed_edt_tasks[stressor],
                        habitat_mask_task
                    ])

            pairwise_risk_path = os.path.join(
                intermediate_dir, f'risk_{habitat}_{stressor}{suffix}.tif')
            pairwise_risk_paths.append(pairwise_risk_path)

            summary_data['risk_path'] = pairwise_risk_path
            pairwise_summary_data.append(summary_data)

            pairwise_risk_task = graph.add_task(
                _calculate_pairwise_risk,
                kwargs={
                    'habitat_mask_raster_path': habitat_mask_path,
                    'exposure_raster_path': criteria_rasters['E'],
                    'consequence_raster_path': criteria_rasters['C'],
                    'risk_equation': args['risk_eq'],
                    'target_risk_raster_path': pairwise_risk_path,
                },
                task_name=f'Calculate pairwise risk for {habitat}/{stressor}',
                target_path_list=[pairwise_risk_path],
                dependent_task_list=sorted(criteria_tasks.values())
            )
            pairwise_risk_tasks.append(pairwise_risk_task)

            reclassified_pairwise_risk_path = os.path.join(
                intermediate_dir, f'reclass_{habitat}_{stressor}{suffix}.tif')
            reclassified_pairwise_risk_paths.append(
                reclassified_pairwise_risk_path)
            reclassified_pairwise_risk_tasks.append(graph.add_task(
                pygeoprocessing.raster_calculator,
                kwargs={
                    'base_raster_path_band_const_list': [
                        (habitat_mask_path, 1),
                        (max_pairwise_risk, 'raw'),
                        (pairwise_risk_path, 1)],
                    'local_op': _reclassify_score,
                    'target_raster_path': reclassified_pairwise_risk_path,
                    'datatype_target': _TARGET_GDAL_TYPE_FLOAT32,
                    'nodata_target': _TARGET_NODATA_FLOAT32
                },
                task_name=f'Reclassify risk for {habitat}/{stressor}',
                target_path_list=[reclassified_pairwise_risk_path],
                dependent_task_list=[pairwise_risk_task]
            ))

        # Sum the pairwise risk scores to get cumulative risk to the habitat.
        cumulative_risk_path = os.path.join(
            intermediate_dir, 'total_risk_{habitat}{suffix}.tif')
        cumulative_risk_to_habitat_paths.append(cumulative_risk_path)
        cumulative_risk_task = graph.add_task(
            ndr._sum_rasters,
            kwargs={
                'raster_path_list': pairwise_risk_paths,
                'target_nodata': _TARGET_NODATA_FLOAT32,
                'target_result_path': cumulative_risk_path,
            },
            task_name=f'Cumulative risk to {habitat}',
            target_path_list=[cumulative_risk_path],
            dependent_task_list=pairwise_risk_tasks
        )
        cumulative_risk_to_habitat_tasks.append(cumulative_risk_task)

        reclassified_cumulative_risk_path = os.path.join(
            intermediate_dir, f'reclass_total_risk_{habitat}{suffix}.tif')
        reclassified_cumulative_risk_task = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (habitat_mask_path, 1),
                    (max_pairwise_risk * max_n_stressors, 'raw'),
                    (cumulative_risk_path, 1)],
                'local_op': _reclassify_score,
                'target_raster_path': reclassified_cumulative_risk_path,
                'datatype_target': _TARGET_GDAL_TYPE_FLOAT32,
                'nodata_target': _TARGET_NODATA_FLOAT32,
            },
            task_name=f'Reclassify risk for {habitat}/{stressor}',
            target_path_list=[reclassified_cumulative_risk_path],
            dependent_task_list=[cumulative_risk_task]
        )

        max_risk_classification_path = os.path.join(
            output_dir, f'risk_{habitat}{suffix}.tif')
        _ = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (habitat_mask_path, 1)
                ] + [(path, 1) for path in reclassified_pairwise_risk_paths],
                'local_op': _maximum_reclassified_score,
                'target_raster_path': max_risk_classification_path,
                'datatype_target': _TARGET_GDAL_TYPE_BYTE,
                'nodata_target': _TARGET_NODATA_BYTE,
            },
            task_name=f'Maximum reclassification for {habitat}',
            target_path_list=[max_risk_classification_path],
            dependent_task_list=[
                reclassified_cumulative_risk_task,
                *reclassified_pairwise_risk_tasks,
            ]
        )

    # total risk is the sum of all cumulative risk rasters.
    ecosystem_risk_path = os.path.join(
        output_dir, f'TOTAL_RISK_Ecosystem{suffix}.tif')
    _ = graph.add_task(
        ndr._sum_rasters,
        kwargs={
            'raster_path_list': cumulative_risk_to_habitat_paths,
            'target_nodata': _TARGET_NODATA_FLOAT32,
            'target_result_path': ecosystem_risk_path,
        },
        task_name='Cumulative risk to ecosystem.',
        target_path_list=[ecosystem_risk_path],
        dependent_task_list=[
            habitat_mask_task,
            *cumulative_risk_to_habitat_tasks,
        ]
    )

    # Recovery attributes are calculated with the same numerical method as
    # other criteria, but are unweighted by distance to a stressor.
    #
    # TODO: verify with Katie, Jess and Jade that this is correct.
    # It's hard to tell from the UG how recovery is supposed to be calculated
    # and what it's supposed to represent.
    for habitat in habitats:
        resilience_criteria_df = criteria_df[
            (criteria_df['habitat'] == habitat) &
            (criteria_df['stressor'] == 'RESILIENCE')]
        criteria_attributes_list = resilience_criteria_df[
            ['rating', 'weight', 'dq']].to_dict(orient='records')

        recovery_score_path = os.path.join(
            intermediate_dir, f'recovery_{habitat}{suffix}.tif')
        recovery_score_task = graph.add_task(
            _calc_criteria,
            kwargs={
                'attributes_list': criteria_attributes_list,
                'habitat_mask_raster_path': habitat_mask_path,
                'target_criterion_path': recovery_score_path,
                'decayed_edt_raster_path': None,  # not a stressor so no EDT
            },
            task_name=f'Calculate recovery score for {habitat}',
            target_path_list=[recovery_score_path],
            dependent_task_list=[habitat_mask_task]
        )

        reclassified_recovery_path = os.path.join(
            intermediate_dir, 'reclass_recovery_{habitat}{suffix}.tif')
        reclassified_recovery_task = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (habitat_mask_path, 1),
                    (max_pairwise_risk, 'raw'),  # TODO: verify
                    (recovery_score_path, 1)],
                'local_op': _reclassify_score,
                'target_raster_path': reclassified_recovery_path,
                'datatype_target': _TARGET_GDAL_TYPE_FLOAT32,
                'nodata_target': _TARGET_NODATA_FLOAT32,
            },
            task_name=f'Reclassify risk for {habitat}/{stressor}',
            target_path_list=[reclassified_cumulative_risk_path],
            dependent_task_list=[habitat_mask_task, recovery_score_task]
        )

    # TODO: create summary statistics output file
    # TODO: output visualization folder.
    # TODO: visualize the graph of tasks to make sure it looks right
    # TODO: Make sure paths match what they're supposed to.

    simplified_aoi_path = os.path.join(
        intermediate_dir, 'simplified_aoi.gpkg')
    aoi_simplify_task = graph.add_task(
        func=_simplify,
        kwargs={
            'source_vector_path': args['aoi_vector_path'],
            'tolerance': resolution / 2,  # by the nyquist theorem
            'target_vector_path': simplified_aoi_path,
            'preserve_columns': ['name'],
        },
        task_name='Simplify AOI',
        target_path_list=[simplified_aoi_path],
        dependent_task_list=[]
    )

    # --> Rasterize the AOI regions for later - not needed until summary stats.
    aoi_subregions_dir = os.path.join(
        intermediate_dir, f'aoi_subregions{suffix}')
    aoi_subregions_json = os.path.join(
        aoi_subregions_dir, f'subregions{suffix}.json')
    rasterize_aoi_regions_task = graph.add_task(
        func=_rasterize_aoi_regions,
        kwargs={
            'source_aoi_vector_path': simplified_aoi_path,
            'habitat_mask_raster': habitat_mask_path,
            'target_raster_dir': aoi_subregions_dir,
            'target_info_json': aoi_subregions_json,
        },
        task_name='Rasterize AOI regions',
        dependent_task_list=[aoi_simplify_task],
        target_path_list=[aoi_subregions_json]
    )

    summary_csv_path = os.path.join(
        output_dir, f'SUMMARY_STATISTICS{suffix}.csv')
    summary_stats_csv_task = graph.add_task(
        func=_create_summary_statistics_file,
        kwargs={
            'aoi_raster_json_path': aoi_subregions_json,
            'habitat_mask_raster_path': habitat_mask_path,
            'pairwise_raster_dicts': pairwise_summary_data,
            'target_summary_csv_path': summary_csv_path,
        },
        task_name='Create summary statistics table',
        target_path_list=[summary_csv_path],
        dependent_task_list=[rasterize_aoi_regions_task]
    )

    graph.close()
    graph.join()

    import sys
    sys.path.insert(0, os.getcwd())
    import make_graph
    make_graph.doit(graph)


# TODO: use the habitats mask raster
def _create_summary_statistics_file(
        aoi_raster_json_path, habitat_mask_raster_path, pairwise_raster_dicts,
        target_summary_csv_path):
    # inputs:
    #  AOI vector (simplified to the nyquist limit)
    #  habitat mask raster
    #  list of dicts {
    #       'habitat': _, 'stressor': _, 'exposure_path': _,
    #       'consequence_path', 'risk_path'}
    #  Working dir
    #  Target CSV path

    # rasterize the AOI vector into distinct nonoverlapping sets.
    # for each nonoverlapping set raster:
    #     associate the exposure, consequence and risk values with the
    #     corresponding aoi regions.
    #     Looking for MIN, MAX, MEAN per subregion.
    #
    # Write out the CSV.

    json_data = json.load(open(aoi_raster_json_path))
    subregion_names = json_data['subregion_names']

    def _read_stats_from_block(raster_path, block_info, mask_array):
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        block = band.ReadAsArray(**block_info)
        valid_pixels = block[
            ~utils.array_equals_nodata(block, nodata) & mask_array]

        return (numpy.min(valid_pixels),
                numpy.max(valid_pixels),
                numpy.sum(valid_pixels),
                valid_pixels.size)

    pairwise_data = {}
    habitats = set()
    stressors = set()
    for info_dict in pairwise_raster_dicts:
        pairwise_data[info_dict['habitat'], info_dict['stressor']] = info_dict
        habitats.add(info_dict['habitat'])
        stressors.add(info_dict['stressor'])

    habitat_mask_raster = gdal.OpenEx(habitat_mask_raster_path, gdal.OF_RASTER)
    habitat_mask_band = habitat_mask_raster.GetRasterBand(1)

    # Track the table data as a list of dicts, to be written out later.
    # Helps reduce human errors in having to write out the columns in precisely
    # the right order as they are declared.
    records = []
    for habitat, stressor in itertools.product(sorted(habitats),
                                               sorted(stressors)):
        e_raster = pairwise_data[habitat, stressor]['e_path']
        c_raster = pairwise_data[habitat, stressor]['c_path']
        r_raster = pairwise_data[habitat, stressor]['risk_path']

        subregion_stats = {}

        for subregion_raster_path in json_data['subregion_rasters']:
            subregion_raster = gdal.OpenEx(subregion_raster_path,
                                           gdal.OF_RASTER)
            subregion_band = subregion_raster.GetRasterBand(1)
            subregion_nodata = subregion_band.GetNoDataValue()
            LOGGER.info(f'{habitat} {stressor} {subregion_raster_path}')

            for block_info in pygeoprocessing.iterblocks(
                    (subregion_raster_path, 1), offset_only=True):
                habitat_mask_block = habitat_mask_band.ReadAsArray(
                    **block_info)
                habitat_mask = (habitat_mask_block == 1)
                subregion_block = subregion_band.ReadAsArray(**block_info)

                local_unique_subregions = numpy.unique(
                    subregion_block[
                        (~utils.array_equals_nodata(
                            subregion_block, subregion_nodata)) &
                        habitat_mask])

                for subregion_id in local_unique_subregions:
                    subregion_mask = (subregion_block == subregion_id)
                    try:
                        stats = subregion_stats[subregion_id]
                    except KeyError:
                        stats = {}
                        for prefix in ('E', 'C', 'R'):
                            for suffix, initial_value in [
                                    ('MIN', float('inf')), ('MAX', 0),
                                    ('SUM', 0), ('N_PIXELS', 0)]:
                                stats[f'{prefix}_{suffix}'] = initial_value

                    for prefix, raster in [('E', e_raster),
                                           ('C', c_raster),
                                           ('R', r_raster)]:
                        pixel_min, pixel_max, pixel_sum, n_pixels = (
                            _read_stats_from_block(raster, block_info,
                                                   subregion_mask))
                        if pixel_min < stats[f'{prefix}_MIN']:
                            stats[f'{prefix}_MIN'] = pixel_min

                        if pixel_max > stats[f'{prefix}_MAX']:
                            stats[f'{prefix}_MAX'] = pixel_max

                        stats[f'{prefix}_SUM'] += pixel_sum
                        stats[f'{prefix}_N_PIXELS'] += n_pixels
                    subregion_stats[subregion_id] = stats

        for subregion_id, stats in subregion_stats.items():
            record = {
                'HABITAT': habitat,
                'STRESSOR': stressor,
                'SUBREGION': subregion_names[str(subregion_id)],
            }
            for prefix in ('E', 'C', 'R'):
                record[f'{prefix}_MIN'] = float(stats[f'{prefix}_MIN'])
                record[f'{prefix}_MAX'] = float(stats[f'{prefix}_MAX'])
                record[f'{prefix}_MEAN'] = float(
                    stats[f'{prefix}_SUM'] / stats[f'{prefix}_N_PIXELS'])
            records.append(record)

    out_dataframe = pandas.DataFrame.from_records(
        records, columns=[
            'HABITAT', 'STRESSOR', 'SUBREGION',
            'E_MIN', 'E_MAX', 'E_MEAN',
            'C_MIN', 'C_MAX', 'C_MEAN',
            'R_MIN', 'R_MAX', 'R_MEAN'])
    out_dataframe.to_csv(target_summary_csv_path, index=False)


def _rasterize_aoi_regions(source_aoi_vector_path, habitat_mask_raster,
                           target_raster_dir, target_info_json):
    # disjoint polygon set
    # add to a Memory layer (and simplify)
    # rasterize to target_raster_dir
    # target_json has structure:
    #   {'subregion_rasters': [paths],
    #    'subregion_ids_to_names': {FID: name}}

    source_aoi_vector = gdal.OpenEx(source_aoi_vector_path, gdal.OF_VECTOR)
    source_aoi_layer = source_aoi_vector.GetLayer()

    driver = ogr.GetDriverByName('MEMORY')
    disjoint_vector = driver.CreateDataSource('disjoint_vector')
    spat_ref = source_aoi_layer.GetSpatialRef()

    subregion_rasters = []
    subregion_names = {}  # {rasterized id: name}

    # locate the "name" field, case-insensitive.
    field_index = None
    for index, fieldname in enumerate([info.GetName() for info in
                                       source_aoi_layer.schema]):
        print(fieldname)
        if fieldname.lower() == 'name':
            field_index = index
            break

    def _write_info_json(subregion_rasters_list, subregion_ids_to_names):
        info_dict = {
            'subregion_rasters': subregion_rasters_list,
            'subregion_names': subregion_ids_to_names,
        }
        with open(target_info_json, 'w') as target_json:
            json.dump(info_dict, target_json, indent=4)

    # If the user did not provide a 'name' field (case-insensitive), then we
    # treat all features as though they're in the same region.
    if field_index is None:
        LOGGER.info(
            'Field "name" (case-insensitive) not found; all features will '
            'be treated as one region')
        target_raster_path = os.path.join(
            target_raster_dir, 'subregion_set_0.tif')
        pygeoprocessing.new_raster_from_base(
            habitat_mask_raster, target_raster_path, _TARGET_GDAL_TYPE_BYTE,
            [_TARGET_NODATA_BYTE])
        pygeoprocessing.rasterize(
            source_aoi_vector_path, target_raster_path,
            burn_values=[1], option_list=['ALL_TOUCHED=TRUE'])
        _write_info_json(
            subregion_rasters_list=[target_raster_path],
            subregion_ids_to_names={1: 'Total Region'})
        return

    for set_index, disjoint_fid_set in enumerate(
            pygeoprocessing.calculate_disjoint_polygon_set(
                source_aoi_vector_path)):
        disjoint_set_raster_path = os.path.join(
            target_raster_dir, f'subregion_set_{set_index}.tif')
        subregion_rasters.append(disjoint_set_raster_path)
        pygeoprocessing.new_raster_from_base(
            habitat_mask_raster, disjoint_set_raster_path, gdal.GDT_Int32,
            [-1])
        fid_raster = gdal.OpenEx(disjoint_set_raster_path, gdal.GA_Update)

        disjoint_layer = disjoint_vector.CreateLayer(
            'disjoint_vector', spat_ref, ogr.wkbPolygon)
        disjoint_layer.CreateField(
            ogr.FieldDefn('FID', ogr.OFTInteger))
        disjoint_layer_defn = disjoint_layer.GetLayerDefn()
        disjoint_layer.StartTransaction()
        for source_fid in disjoint_fid_set:
            source_feature = source_aoi_layer.GetFeature(source_fid)
            source_geometry = source_feature.GetGeometryRef()
            new_feature = ogr.Feature(disjoint_layer_defn)
            new_feature.SetGeometry(source_geometry.Clone())
            source_geometry = None

            new_feature.SetField('FID', source_fid)
            if field_index is None:
                subregion_name = source_fid
            else:
                subregion_name = source_feature.GetField(field_index)
            subregion_names[source_fid] = subregion_name

            disjoint_layer.CreateFeature(new_feature)
        disjoint_layer.CommitTransaction()
        gdal.RasterizeLayer(
            fid_raster, [1], disjoint_layer,
            options=["ALL_TOUCHED=FALSE", "ATTRIBUTE=FID"])
        fid_raster.FlushCache()
        fid_raster = None

        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

    _write_info_json(
        subregion_rasters_list=subregion_rasters,
        subregion_ids_to_names=subregion_names)


def _align(raster_path_map, vector_path_map, target_pixel_size, target_srs_wkt):
    # Determine the union bounding box of all inputs.
    # if rasters to align, align them with align_and_resize.
    # if vectors to align, create rasters based on bounding box and then
    # rasterize.

    bounding_box_list = []
    source_raster_paths = []
    aligned_raster_paths = []
    for source_raster_path, aligned_raster_path in raster_path_map.items():
        source_raster_paths.append(source_raster_path)
        aligned_raster_paths.append(aligned_raster_path)
        raster_info = pygeoprocessing.get_raster_info(source_raster_path)
        bounding_box_list.append(pygeoprocessing.transform_bounding_box(
            raster_info['bounding_box'], raster_info['projection_wkt'],
            target_srs_wkt))

    for source_vector_path in vector_path_map.keys():
        vector_info = pygeoprocessing.get_vector_info(source_vector_path)
        bounding_box_list.append(pygeoprocessing.transform_bounding_box(
            vector_info['bounding_box'], vector_info['projection_wkt'],
            target_srs_wkt))

    # Bounding box is in the order [minx, miny, maxx, maxy]
    target_bounding_box = pygeoprocessing.merge_bounding_box_list(
        bounding_box_list, 'union')

    if raster_path_map:
        LOGGER.info(f'Aligning {len(raster_path_map)} rasters')
        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list=source_raster_paths,
            target_raster_path_list=aligned_raster_paths,
            resample_method_list=['near'] * len(source_raster_paths),
            target_pixel_size=target_pixel_size,
            bounding_box_mode=target_bounding_box,
            target_srs_wkt=target_srs_wkt
        )

    if vector_path_map:
        LOGGER.info(f'Aligning {len(vector_path_map)} vectors')
        for source_vector_path, target_raster_path in vector_path_map.items():
            _create_raster_from_bounding_box(
                target_raster_path=target_raster_path,
                target_bounding_box=target_bounding_box,
                target_pixel_size=target_pixel_size,
                target_pixel_type=_TARGET_GDAL_TYPE_BYTE,
                target_srs_wkt=target_srs_wkt,
                fill_value=_TARGET_NODATA_BYTE
            )

            pygeoprocessing.rasterize(
                source_vector_path, target_raster_path,
                burn_values=[1], option_list=['ALL_TOUCHED=TRUE'])


def _create_raster_from_bounding_box(
        target_raster_path, target_bounding_box, target_pixel_size,
        target_pixel_type, target_srs_wkt, fill_value=None):

    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = target_bounding_box

    driver = gdal.GetDriverByName('GTiff')
    n_bands = 1
    n_cols = int(numpy.ceil(
        abs((bbox_maxx - bbox_minx) / target_pixel_size[0])))
    n_rows = int(numpy.ceil(
        abs((bbox_maxy - bbox_miny) / target_pixel_size[1])))

    raster = driver.Create(
        target_raster_path, n_cols, n_rows, n_bands, target_pixel_type,
        options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
                 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    raster.SetProjection(target_srs_wkt)

    # Set the transform based on the upper left corner and given pixel
    # dimensions.  Bounding box is in format [minx, miny, maxx, maxy]
    if target_pixel_size[0] < 0:
        x_source = bbox_maxx
    else:
        x_source = bbox_minx
    if target_pixel_size[1] < 0:
        y_source = bbox_maxy
    else:
        y_source = bbox_miny
    raster_transform = [
        x_source, target_pixel_size[0], 0.0,
        y_source, 0.0, target_pixel_size[1]]
    raster.SetGeoTransform(raster_transform)

    # Initialize everything to nodata
    if fill_value is not None:
        band = raster.GetRasterBand(1)
        band.Fill(fill_value)
        band = None
    raster = None


def _simplify(source_vector_path, tolerance, target_vector_path,
              preserve_columns=None):
    if preserve_columns is None:
        preserve_columns = []
    preserve_columns = set(name.lower() for name in preserve_columns)

    source_vector = gdal.OpenEx(source_vector_path)
    source_layer = source_vector.GetLayer()

    target_driver = gdal.GetDriverByName('GPKG')
    target_vector = target_driver.Create(
        target_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    target_layer_name = os.path.splitext(
        os.path.basename(target_vector_path))[0]

    # Using wkbUnknown is important here because a user can provide a single
    # vector with multiple geometry types.  GPKG can handle whatever geom types
    # we want it to use, but it will only be a conformant GPKG if and only if
    # we set the layer type to ogr.wkbUnknown.  Otherwise, the GPKG standard
    # would expect that all geometries in a layer match the geom type of the
    # layer and GDAL will raise a warning if that's not the case.
    target_layer = target_vector.CreateLayer(
        target_layer_name, source_layer.GetSpatialRef(), ogr.wkbUnknown)

    for field in source_layer.schema:
        if field.GetName().lower() in preserve_columns:
            new_definition = ogr.FieldDefn(field.GetName(), field.GetType())
            target_layer.CreateField(new_definition)

    target_layer_defn = target_layer.GetLayerDefn()
    target_layer.StartTransaction()

    for source_feature in source_layer:
        target_feature = ogr.Feature(target_layer_defn)
        source_geom = source_feature.GetGeometryRef()

        simplified_geom = source_geom.SimplifyPreserveTopology(tolerance)
        if simplified_geom is not None:
            target_geom = simplified_geom
        else:
            # If the simplification didn't work for whatever reason, fall back
            # to the original geometry.
            target_geom = source_geom

        for fieldname in [field.GetName() for field in target_layer.schema]:
            target_feature.SetField(
                fieldname, source_feature.GetField(fieldname))

        target_feature.SetGeometry(target_geom)
        target_layer.CreateFeature(target_feature)

    target_layer.CommitTransaction()
    target_layer = None
    target_vector = None


def _prep_input_raster(source_raster_path, target_raster_path):
    # The intent of this function is to take whatever raster the user gives us
    # and convert its pixel values to 1 or nodata.

    source_nodata = pygeoprocessing.get_raster_info(
        source_raster_path)['nodata'][0]

    def _translate_op(input_array):
        presence = numpy.full(input_array.shape, _TARGET_NODATA_BYTE,
                              dtype=numpy.uint8)
        valid_mask = ~utils.array_equals_nodata(input_array, source_nodata)
        presence[valid_mask & (input_array == 1)] = 1
        return presence

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _translate_op, target_raster_path,
        _TARGET_GDAL_TYPE_BYTE, _TARGET_NODATA_BYTE)


def _habitat_mask_op(*habitats):
    output_mask = numpy.full(habitats[0].shape, _TARGET_NODATA_BYTE,
                             dtype=numpy.uint8)
    for habitat_array in habitats:
        output_mask[habitat_array == 1] = 1

    return output_mask


# TODO: support Excel and CSV both
def _parse_info_table(info_table_path):
    info_table_path = os.path.abspath(info_table_path)

    table = utils.read_csv_to_dataframe(info_table_path, to_lower=True)
    table = table.set_index('name')
    table = table.rename(columns={'stressor buffer (meters)': 'buffer'})

    def _make_abspath(row):
        path = row['path'].replace('\\', '/')
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.dirname(info_table_path), path)

    table['path'] = table.apply(lambda row: _make_abspath(row), axis=1)

    # Drop the buffer column from the habitats list; we don't need it.
    habitats = table.loc[table['type'] == 'habitat'].drop(
        columns=['type', 'buffer']).to_dict(orient='index')

    # Keep the buffer column in the stressors dataframe.
    stressors = table.loc[table['type'] == 'stressor'].drop(
        columns=['type']).to_dict(orient='index')

    # TODO: check that habitats and stressor names are nonoverlapping sets.

    return (habitats, stressors)


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
    decay_type = decay_type.lower()
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
        elif decay_type == 'none':
            # everything within the buffer distance has a value of 1
            decayed_edt[valid_pixels] = 1
        else:
            raise AssertionError(f'Invalid decay type {decay_type} provided.')

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
        habitat_mask_raster_path, target_criterion_path,
        _TARGET_GDAL_TYPE_FLOAT32, [_TARGET_NODATA_FLOAT32])

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

        criterion_score = numpy.full(
            habitat_mask.shape, _TARGET_NODATA_FLOAT32, dtype=numpy.float32)
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
            data_quality = attribute_dict['dq']
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
        risk_array = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLOAT32,
                                dtype=numpy.float32)
        risk_array[habitat_pixels] = (
            exposure[habitat_pixels] * consequence[habitat_pixels])
        return risk_array

    def _euclidean_risk(habitat_mask, exposure, consequence):
        habitat_pixels = (habitat_mask == 1)
        risk_array = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLOAT32,
                                dtype=numpy.float32)
        risk_array[habitat_pixels] = numpy.sqrt(
            (exposure[habitat_pixels] - 1) ** 2 +
            (consequence[habitat_pixels] - 1) ** 2)
        return risk_array

    risk_equation = risk_equation.lower()
    if risk_equation == 'multiplicative':
        risk_op = _muliplicative_risk
    elif risk_equation == 'euclidean':
        risk_op = _euclidean_risk
    else:
        raise AssertionError(f'Invalid risk equation {risk_equation} provided')

    pygeoprocessing.raster_calculator(
        [(habitat_mask_raster_path, 1),
         (exposure_raster_path, 1),
         (consequence_raster_path, 1)],
        risk_op, target_risk_raster_path, _TARGET_GDAL_TYPE_FLOAT32,
        _TARGET_NODATA_FLOAT32)


# max pairwise risk or recovery is calculated based on user input and choice of
# risk equation.  Might as well pass in the numeric value rather than the risk
# equation type.
def _reclassify_score(habitat_mask, max_pairwise_risk, score):
    habitat_pixels = (habitat_mask == 1)
    reclassified = numpy.full(habitat_mask.shape, _TARGET_NODATA_BYTE,
                              dtype=numpy.uint8)
    reclassified[habitat_pixels] = numpy.digitize(
        score[habitat_pixels],
        [0, max_pairwise_risk*(1/3), max_pairwise_risk*(2/3)],
        right=True)  # bins[i-1] >= x > bins[i]
    return reclassified


def _maximum_reclassified_score(habitat_mask, *risk_classes):
    target_matrix = numpy.zeros(habitat_mask.shape, dtype=numpy.uint8)
    valid_pixels = (habitat_mask == 1)

    for risk_class_matrix in risk_classes:
        target_matrix[valid_pixels] = numpy.maximum(
            target_matrix[valid_pixels], risk_class_matrix[valid_pixels])

    target_matrix[~valid_pixels] = _TARGET_NODATA_BYTE
    return target_matrix


def build_datastack_archive(args, datastack_path):
    """Build a datastack-compliant archive of all spatial inputs to HRA.

    This function is implemented here and not in natcap.invest.datastack
    because HRA's inputs are too complicated to describe in ARGS_SPEC.  Because
    the input table and its linked spatial inputs are too custom, it warrants a
    custom datastack archive-generation function.

    Args:
        args (dict): The complete ``args`` dict to package up into a datastack
            archive.
        datastack_path (string): The path on disk to where the datastack should
            be written.

    Returns:
        ``None``
    """
    # TODO: flesh this out
    raise NotImplementedError()


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire ``args`` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    return validation.validate(args, ARGS_SPEC['args'])
