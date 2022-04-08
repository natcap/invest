"""Habitat risk assessment (HRA) model for InVEST."""
# -*- coding: UTF-8 -*-
import collections
import itertools
import json
import logging
import math
import os
import re
import shutil

import numpy
import pandas
import pygeoprocessing
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
_WGS84_SRS = osr.SpatialReference()
_WGS84_SRS.ImportFromEPSG(4326)
_WGS84_WKT = _WGS84_SRS.ExportToWkt()

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
                "multiplicative": {"display_name": _("Multiplicative")},
                "euclidean": {"display_name": _("Euclidean")}
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
        "n_overlapping_stressors": {
            "name": _("Number of Overlapping Stressors"),
            "type": "number",
            "required": True,
            "about": _(
                "The number of overlapping stressors to consider as "
                "'maximum' when reclassifying risk scores into "
                "high/medium/low.  Affects the breaks between risk "
                "classifications."),
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
        args['n_overlapping_stressors'] (number): If provided, this
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
    output_dir = os.path.join(args['workspace_dir'], 'outputs')
    taskgraph_working_dir = os.path.join(args['workspace_dir'], '.taskgraph')
    utils.make_directories([intermediate_dir, output_dir])
    suffix = utils.make_suffix_string(args, 'results_suffix')

    resolution = float(args['resolution'])
    max_rating = float(args['max_rating'])
    max_n_stressors = float(args['n_overlapping_stressors'])
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
    criteria_habitats, criteria_stressors = _parse_criteria_table(
        args['criteria_table_path'], composite_criteria_table_path)

    # Validate that habitats and stressors match precisely.
    for label, info_set, criteria_set in [
            ('habitats', set(habitats.keys()), criteria_habitats),
            ('stressors', set(stressors.keys()), criteria_stressors)]:
        if info_set != criteria_set:
            missing_from_info_table = ", ".join(
                sorted(criteria_set - info_set))
            missing_from_criteria_table = ", ".join(
                sorted(info_set - criteria_set))
            raise ValueError(
                f"The {label} in the info and criteria tables do not match:\n"
                f"  Missing from info table: {missing_from_info_table}\n"
                f"  Missing from criteria table: {missing_from_criteria_table}"
            )

    criteria_df = pandas.read_csv(composite_criteria_table_path,
                                  index_col=False)
    # Because criteria may be spatial, we need to prepare those spatial inputs
    # as well.
    spatial_criteria_attrs = {}
    for (habitat, stressor, criterion, rating) in criteria_df[
            ['habitat', 'stressor', 'criterion',
             'rating']].itertuples(index=False):
        if isinstance(rating, (int, float)):
            continue  # obviously a numeric rating
        try:
            float(rating)
            continue
        except ValueError:
            # If we can't cast it to a float, assume it's a string and
            # therefore spatial.
            pass

        # If the rating is non-numeric, assume it's a spatial criterion.
        # this dict matches the structure of the outputs for habitat/stressor
        # dicts, from _parse_info_table
        name = f'{habitat}-{stressor}-{criterion}'
        spatial_criteria_attrs[name] = {
            'name': name,
            'path': rating,  # Previously validated to be a GIS type.
        }

    # Preprocess habitat, stressor spatial criteria datasets.
    # All of these are spatial in nature but might be rasters or vectors.
    user_files_to_aligned_raster_paths = {}
    alignment_source_raster_paths = {}
    alignment_source_vector_paths = {}
    alignment_dependent_tasks = []
    aligned_habitat_raster_paths = {}
    aligned_stressor_raster_paths = {}
    for name, attributes in itertools.chain(habitats.items(),
                                            stressors.items(),
                                            spatial_criteria_attrs.items()):
        source_filepath = attributes['path']
        gis_type = pygeoprocessing.get_gis_type(source_filepath)
        aligned_raster_path = os.path.join(
            intermediate_dir, f'aligned_{name}{suffix}.tif')
        user_files_to_aligned_raster_paths[source_filepath] = aligned_raster_path

        # If the input is already a raster, run it through raster_calculator to
        # ensure we know the nodata value and pixel values.
        if gis_type == pygeoprocessing.RASTER_TYPE:
            rewritten_raster_path = os.path.join(
                intermediate_dir, f'rewritten_{name}{suffix}.tif')
            alignment_source_raster_paths[
                rewritten_raster_path] = aligned_raster_path
            # Habitats/stressors must have pixel values of 0 or 1.
            # Criteria may be between [0, max_criteria_score]
            if name in spatial_criteria_attrs:
                # Spatial criteria rasters can represent any positive real
                # values, though they're supposed to be in the range [0, max
                # criteria score], inclusive.
                prep_raster_task = graph.add_task(
                    func=_prep_input_criterion_raster,
                    kwargs={
                        'source_raster_path': source_filepath,
                        'target_filepath': rewritten_raster_path,
                    },
                    task_name=(
                        f'Rewrite {name} criteria raster for consistency'),
                    target_path_list=[rewritten_raster_path],
                    dependent_task_list=[]
                )
            else:
                # habitat/stressor rasters represent presence/absence in the
                # form of 1 or 0 pixel values.
                prep_raster_task = graph.add_task(
                    func=_mask_binary_presence_absence_rasters,
                    kwargs={
                        'source_raster_paths': [source_filepath],
                        'target_mask_path': rewritten_raster_path,
                    },
                    task_name=(
                        f'Rewrite {name} habitat/stressor raster for '
                        'consistency'),
                    target_path_list=[rewritten_raster_path],
                    dependent_task_list=[]
                )
            alignment_dependent_tasks.append(prep_raster_task)

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

            # Spatial habitats/stressors are rasterized as presence/absence, so
            # we don't need to preserve any columns.
            fields_to_preserve = None
            if name in spatial_criteria_attrs:
                # In spatial criteria vectors, the 'rating' field contains the
                # numeric rating that needs to be rasterized.
                fields_to_preserve = ['rating']

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
                    'preserve_columns': fields_to_preserve,
                },
                task_name=f'Simplify {name}',
                target_path_list=[target_simplified_vector],
                dependent_task_list=[reprojected_vector_task]
            ))

        # Later operations make use of the habitats rasters or the stressors
        # rasters, so it's useful to collect those here now.
        if name in habitats:
            aligned_habitat_raster_paths[name] = aligned_raster_path
        elif name in stressors:
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
    all_habitats_mask_path = os.path.join(
        intermediate_dir, f'habitat_mask{suffix}.tif')
    all_habitats_mask_task = graph.add_task(
        _mask_binary_presence_absence_rasters,
        kwargs={
            'source_raster_paths':
                list(aligned_habitat_raster_paths.values()),
            'target_mask_path': all_habitats_mask_path,
        },
        task_name='Create mask of all habitats',
        target_path_list=[all_habitats_mask_path],
        dependent_task_list=[alignment_task]
    )

    # --> for stressor in stressors, do a decayed EDT.
    decayed_edt_paths = {}  # {stressor name: decayed EDT raster}
    decayed_edt_tasks = {}  # {stressor name: decayed EDT task}
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

    # Save this dataframe to make indexing in this loop a little cheaper
    # Resilience/recovery calculations are only done for Consequence criteria.
    cumulative_risk_to_habitat_paths = []
    cumulative_risk_to_habitat_tasks = []
    reclassified_rasters = []  # For visualization geojson, if requested
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
                    f'{criteria_type}_{habitat}_{stressor}{suffix}.tif')
                criteria_rasters[criteria_type] = criteria_raster_path
                summary_data[
                    f'{criteria_type.lower()}_path'] = criteria_raster_path

                # This rather complicated filter just grabs the rows matching
                # this habitat, stressor and criteria type.  It's the pandas
                # equivalent of SELECT * FROM criteria_df WHERE the habitat,
                # stressor and criteria type match the current habitat,
                # stressor and criteria type.
                local_criteria_df = criteria_df[
                    (criteria_df['habitat'] == habitat) &
                    (criteria_df['stressor'] == stressor) &
                    (criteria_df['e/c'] == criteria_type)]

                # If we are doing consequence calculations, add in the
                # resilience/recovery parameters for this habitat as additional
                # criteria.
                # Note that if a user provides an E-type RESILIENCE criterion,
                # it will be ignored in all criteria calculations.
                if criteria_type == 'C':
                    local_resilience_df = criteria_df[
                        (criteria_df['habitat'] == habitat) &
                        (criteria_df['stressor'] == 'RESILIENCE') &
                        (criteria_df['e/c'] == 'C')]
                    local_criteria_df = pandas.concat(
                        [local_criteria_df, local_resilience_df])

                # This produces a list of dicts in the form:
                # [{'rating': (score), 'weight': (score), 'dq': (score)}],
                # which is what _calc_criteria() expects.
                attributes_list = []
                for attrs in local_criteria_df[
                        ['rating', 'weight', 'dq']].to_dict(orient='records'):
                    try:
                        float(attrs['rating'])
                    except ValueError:
                        # When attrs['rating'] is not a number, we should
                        # assume it's a spatial file.
                        attrs['rating'] = user_files_to_aligned_raster_paths[
                            attrs['rating']]
                    attributes_list.append(attrs)

                criteria_tasks[criteria_type] = graph.add_task(
                    _calc_criteria,
                    kwargs={
                        'attributes_list': attributes_list,
                        'habitat_mask_raster_path':
                            aligned_habitat_raster_paths[habitat],
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
                        all_habitats_mask_task
                    ])

            pairwise_risk_path = os.path.join(
                intermediate_dir, f'RISK_{habitat}_{stressor}{suffix}.tif')
            pairwise_risk_paths.append(pairwise_risk_path)

            summary_data['risk_path'] = pairwise_risk_path
            pairwise_summary_data.append(summary_data)

            pairwise_risk_task = graph.add_task(
                _calculate_pairwise_risk,
                kwargs={
                    'habitat_mask_raster_path':
                        aligned_habitat_raster_paths[habitat],
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
                        (aligned_habitat_raster_paths[habitat], 1),
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
            output_dir, f'TOTAL_RISK_{habitat}{suffix}.tif')
        cumulative_risk_to_habitat_paths.append(cumulative_risk_path)
        cumulative_risk_task = graph.add_task(
            _sum_rasters,
            kwargs={
                'raster_path_list': pairwise_risk_paths,
                'target_nodata': _TARGET_NODATA_FLOAT32,
                'target_result_path': cumulative_risk_path,
                'normalize': False,
            },
            task_name=f'Cumulative risk to {habitat}',
            target_path_list=[cumulative_risk_path],
            dependent_task_list=pairwise_risk_tasks
        )
        cumulative_risk_to_habitat_tasks.append(cumulative_risk_task)

        reclassified_cumulative_risk_path = os.path.join(
            intermediate_dir, f'reclass_total_risk_{habitat}{suffix}.tif')
        reclassified_rasters.append(reclassified_cumulative_risk_path)
        reclassified_cumulative_risk_task = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (aligned_habitat_raster_paths[habitat], 1),
                    (max_pairwise_risk * max_n_stressors, 'raw'),
                    (cumulative_risk_path, 1)],
                'local_op': _reclassify_score,
                'target_raster_path': reclassified_cumulative_risk_path,
                'datatype_target': _TARGET_GDAL_TYPE_BYTE,
                'nodata_target': _TARGET_NODATA_BYTE,
            },
            task_name=f'Reclassify risk for {habitat}/{stressor}',
            target_path_list=[reclassified_cumulative_risk_path],
            dependent_task_list=[cumulative_risk_task]
        )

        max_risk_classification_path = os.path.join(
            output_dir, f'RECLASS_RISK_{habitat}{suffix}.tif')
        _ = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (aligned_habitat_raster_paths[habitat], 1)
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

    # total risk to the ecosystem is the sum of all cumulative risk rasters
    # across all habitats.
    # InVEST 3.3.3 has this as the cumulative risk (a straight sum).
    # InVEST 3.10.2 has this as the mean risk per habitat.
    # This is currently implemented as mean risk per habitat.
    ecosystem_risk_path = os.path.join(
        output_dir, f'TOTAL_RISK_Ecosystem{suffix}.tif')
    ecosystem_risk_task = graph.add_task(
        _sum_rasters,
        kwargs={
            'raster_path_list': cumulative_risk_to_habitat_paths,
            'target_nodata': _TARGET_NODATA_FLOAT32,
            'target_result_path': ecosystem_risk_path,
            'normalize': True,
        },
        task_name='Cumulative risk to ecosystem.',
        target_path_list=[ecosystem_risk_path],
        dependent_task_list=[
            all_habitats_mask_task,
            *cumulative_risk_to_habitat_tasks,
        ]
    )

    # This represents the risk across all stressors.
    # I'm guessing about the risk break to use here, but since the
    # `ecosystem_risk_path` here is the sum across habitats, it makes sense to
    # use max_pairwise_risk * n_habitats.
    reclassified_ecosystem_risk_path = os.path.join(
        output_dir, f'RECLASS_RISK_Ecosystem{suffix}.tif')
    reclassified_ecosystem_risk_task = graph.add_task(
        pygeoprocessing.raster_calculator,
        kwargs={
            'base_raster_path_band_const_list': [
                (all_habitats_mask_path, 1),
                (max_pairwise_risk * len(habitats), 'raw'),
                (ecosystem_risk_path, 1)],
            'local_op': _reclassify_score,
            'target_raster_path': reclassified_ecosystem_risk_path,
            'datatype_target': _TARGET_GDAL_TYPE_BYTE,
            'nodata_target': _TARGET_NODATA_BYTE,
        },
        task_name=f'Reclassify risk for {habitat}/{stressor}',
        target_path_list=[reclassified_cumulative_risk_path],
        dependent_task_list=[all_habitats_mask_task, ecosystem_risk_task]
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
        criteria_attributes_list = []
        for attrs in resilience_criteria_df[
                ['rating', 'weight', 'dq']].to_dict(orient='records'):
            try:
                float(attrs['rating'])
            except ValueError:
                # When attrs['rating'] is not a number, we should assume it's a
                # spatial file.
                attrs['rating'] = user_files_to_aligned_raster_paths[attrs['rating']]
            criteria_attributes_list.append(attrs)

        recovery_score_path = os.path.join(
            intermediate_dir, f'RECOVERY_{habitat}{suffix}.tif')
        recovery_score_task = graph.add_task(
            _calc_criteria,
            kwargs={
                'attributes_list': criteria_attributes_list,
                'habitat_mask_raster_path':
                    aligned_habitat_raster_paths[habitat],
                'target_criterion_path': recovery_score_path,
                'decayed_edt_raster_path': None,  # not a stressor so no EDT
            },
            task_name=f'Calculate recovery score for {habitat}',
            target_path_list=[recovery_score_path],
            dependent_task_list=[all_habitats_mask_task]
        )

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

    # --> Rasterize the AOI regions for summary stats.
    aoi_subregions_dir = os.path.join(
        intermediate_dir, f'aoi_subregions{suffix}')
    aoi_subregions_json = os.path.join(
        aoi_subregions_dir, f'subregions{suffix}.json')
    rasterize_aoi_regions_task = graph.add_task(
        func=_rasterize_aoi_regions,
        kwargs={
            'source_aoi_vector_path': simplified_aoi_path,
            'habitat_mask_raster': all_habitats_mask_path,
            'target_raster_dir': aoi_subregions_dir,
            'target_info_json': aoi_subregions_json,
        },
        task_name='Rasterize AOI regions',
        dependent_task_list=[aoi_simplify_task, all_habitats_mask_task],
        target_path_list=[aoi_subregions_json]
    )

    summary_csv_path = os.path.join(
        output_dir, f'SUMMARY_STATISTICS{suffix}.csv')
    summary_stats_csv_task = graph.add_task(
        func=_create_summary_statistics_file,
        kwargs={
            'aoi_raster_json_path': aoi_subregions_json,
            'habitat_mask_raster_path': all_habitats_mask_path,
            'pairwise_raster_dicts': pairwise_summary_data,
            'target_summary_csv_path': summary_csv_path,
        },
        task_name='Create summary statistics table',
        target_path_list=[summary_csv_path],
        dependent_task_list=[rasterize_aoi_regions_task]
    )

    graph.join()
    if not args.get('visualize_outputs', False):
        graph.close()
        return

    # Although the generation of visualization outputs could have more precise
    # task dependencies, the effort involved in tracking them precisely doesn't
    # feel worth it when the visualization steps are such a standalone
    # component and it isn't clear if we'll end up keeping this in HRA or
    # refactoring it out (should we end up doing more such visualizations).
    LOGGER.info('Generating visualization outputs')
    visualization_dir = os.path.join(args['workspace_dir'],
                                     'visualization_outputs')
    utils.make_directories([visualization_dir])
    shutil.copy(  # copy in the summary table.
        summary_csv_path,
        os.path.join(visualization_dir, os.path.basename(summary_csv_path)))

    # For each raster in reclassified risk rasters + Reclass ecosystem risk:
    #   convert to geojson with fieldname "Risk Score"
    reclassified_rasters.append(reclassified_ecosystem_risk_path)
    for raster_paths, fieldname, geojson_prefix in [
            (reclassified_rasters, 'Risk Score', 'RECLASS_RISK'),
            (aligned_stressor_raster_paths.values(), 'Stressor', 'STRESSOR')]:
        for source_raster_path in raster_paths:
            basename = os.path.splitext(
                os.path.basename(source_raster_path))[0]

            # clean up the filename to what the viz webapp expects.
            for pattern in (f'^{geojson_prefix}_',
                            '^aligned_',
                            '^reclass_total_risk_'):
                basename = re.sub(pattern, '', basename)

            polygonize_mask_raster_path = os.path.join(
                intermediate_dir, f'polygonize_mask_{basename}.tif')
            rewrite_for_polygonize_task = graph.add_task(
                func=_create_mask_for_polygonization,
                kwargs={
                    'source_raster_path': source_raster_path,
                    'target_raster_path': polygonize_mask_raster_path,
                },
                task_name=f'Rewrite {basename} for polygonization',
                target_path_list=[polygonize_mask_raster_path],
                dependent_task_list=[]
            )

            polygonized_gpkg = os.path.join(
                intermediate_dir, f'polygonized_{basename}.gpkg')
            polygonize_task = graph.add_task(
                func=_polygonize,
                kwargs={
                    'source_raster_path': source_raster_path,
                    'mask_raster_path': polygonize_mask_raster_path,
                    'target_polygonized_vector': polygonized_gpkg,
                    'field_name': fieldname,
                },
                task_name=f'Polygonizing {basename}',
                target_path_list=[polygonized_gpkg],
                dependent_task_list=[rewrite_for_polygonize_task]
            )

            target_geojson_path = os.path.join(
                visualization_dir,
                f'{geojson_prefix}_{basename}.geojson')
            _ = graph.add_task(
                pygeoprocessing.reproject_vector,
                kwargs={
                    'base_vector_path': polygonized_gpkg,
                    'target_projection_wkt': _WGS84_WKT,
                    'target_path': target_geojson_path,
                    'driver_name': 'GeoJSON',
                },
                task_name=f'Reproject {name} to AOI',
                target_path_list=[target_geojson_path],
                dependent_task_list=[polygonize_task]
            )

    graph.close()
    graph.join()

    # TODO: check the task graph - AST most likely
    # TODO: tables - support excel and also CSV.
    # TODO: docstrings
    # TODO: contructed tests.
    # TODO: function to build a datastack archive.
    # TODO: proposed plan for migrating relevant functions to pygeoprocessing.

    # import sys
    # sys.path.insert(0, os.getcwd())
    # import make_graph
    # make_graph.doit(graph)


def _create_mask_for_polygonization(source_raster_path, target_raster_path):
    """Create a mask of non-nodata pixels.

    This mask raster is intended to be used as a mask input for GDAL's
    polygonization function.

    Args:
        source_raster_path (string): The source raster from which the mask
            raster will be created. This raster is assumed to be an integer
            raster.
        target_raster_path (string): The path to where the target raster should
            be written.

    Returns:
        ``None``
    """
    nodata = pygeoprocessing.get_raster_info(source_raster_path)['nodata'][0]

    def _rewrite(raster_values):
        """Convert any non-nodata values to 1, all other values to 0.

        Args:
            raster_values (numpy.array): Integer pixel values from the source
                raster.

        Returns:
            out_array (numpy.array): An unsigned byte mask with pixel values of
            0 (on nodata pixels) or 1 (on non-nodata pixels)."""
        return (raster_values != nodata).astype(numpy.uint8)

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _rewrite, target_raster_path,
        gdal.GDT_Byte, 0)


def _convert_to_binary_mask(source_raster_path, target_raster_path,
                            target_nodata):
    nodata = pygeoprocessing.get_raster_info(source_raster_path)['nodata'][0]

    def _rewrite(*raster_values):
        """Convert any non-nodata values to 1, all other values to 0.

        Args:
            raster_values (numpy.array): Integer pixel values from the source
                raster.

        Returns:
            out_array (numpy.array): An unsigned byte mask with pixel values of
            0 (on nodata pixels) or 1 (on non-nodata pixels)."""
        return (raster_values != nodata).astype(numpy.uint8)

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _rewrite, target_raster_path,
        gdal.GDT_Byte, target_nodata)


def _polygonize(source_raster_path, mask_raster_path,
                target_polygonized_vector, field_name):
    """Polygonize a raster.

    Args:
        source_raster_path (string): The source raster to polygonize.  This
            raster must be an integer raster.
        mask_raster_path (string): The path to a mask raster, where pixel
            values are 1 where polygons should be created, 0 otherwise.
        target_polygonized_vector (string): The path to a vector into which the
            new polygons will be inserted.  A new GeoPackage will be created at
            this location.
        field_name (string): The name of a field into which the polygonized
            region's numerical value should be recorded.  A new field with this
            name will be created in ``target_polygonized_vector``, and with an
            Integer field type.

    Returns:
        ``None``
    """
    raster = gdal.OpenEx(source_raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)

    mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())

    driver = gdal.GetDriverByName('GPKG')
    vector = driver.Create(
        target_polygonized_vector, 0, 0, 0, gdal.GDT_Unknown)
    layer_name = os.path.splitext(
        os.path.basename(target_polygonized_vector))[0]
    layer = vector.CreateLayer('', raster_srs, ogr.wkbPolygon)

    # Create an integer field that contains values from the raster
    field_defn = ogr.FieldDefn(str(field_name), ogr.OFTInteger)
    field_defn.SetWidth(3)
    field_defn.SetPrecision(0)
    layer.CreateField(field_defn)

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())

    layer.StartTransaction()
    # 0 represents field index 0, into which pixel values will be written.
    gdal.Polygonize(band, mask_band, layer, 0)
    layer.CommitTransaction()


def _create_summary_statistics_file(
        aoi_raster_json_path, habitat_mask_raster_path, pairwise_raster_dicts,
        target_summary_csv_path):
    """Summarize pairwise habitat/stressor rasters by AOI subregions.

    Functionally, this is a modified version of
    ``pygeoprocessing.zonal_statistics`` that doesn't force re-rasterization of
    subregions.

    Args:
        aoi_raster_json_path (string): The path to a JSON file with information
            about the provided AOI subregions.  The following keys must exist
            in the dict:

                * ``subregion_names``: A mapping of integer subregion IDs to
                  string names identifying the subregion.
                * ``subregion_rasters``: A list of rasters containing
                  disjoint sets of rasterized AOI geometries, where each
                  geometry has been rasterized with a unique integer subregion
                  ID that is also identified in ``subregion_names``.

        habitat_mask_raster_path (string): The path to a raster with 1s
            indicating the presence of habitats (any number) and 0 or nodata
            otherwise.
        pairwise_raster_dicts (list): A list of dicts for each habitat/stressor
            pair containing the following keys:

                * ``"habitat"`` the string habitat name.
                * ``"stressor"`` the string stressor name.
                * ``"e_path"`` the exposure criteria score raster.
                * ``"c_path"`` the consequence criteria score raster.
                * ``"risk_path"`` the raster of calculated risk.

        target_summary_csv_path (string): The path to where the summary CSV
            should be written

    Returns:
        ``None``
    """
    with open(aoi_raster_json_path) as aoi_json_file:
        json_data = json.load(aoi_json_file)
    subregion_names = json_data['subregion_names']

    def _read_stats_from_block(raster_path, block_info, mask_array):
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        block = band.ReadAsArray(**block_info)
        valid_pixels = block[
            ~utils.array_equals_nodata(block, nodata) & mask_array]

        if valid_pixels.size == 0:
            return (float('inf'), 0, 0, 0)

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

    # itertools.product is roughly the equivalent of a nested for-loop.
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
                        # If subregion_id is not in the subregion_stats dict,
                        # we need to initialize it.
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
    """Rasterize AOI subregions.

    If the source AOI vector has a "NAME" field (case-insensitive), then
    non-overlapping sets of polygons are rasterized onto as many rasters as are
    needed to cover all of the AOI's subregions.

    If the source AOI vector does not have a "NAME" field (case-insensitive),
    then all AOI features are considered to be in the same region and all
    features are rasterized onto a single raster.  In this case, the subregion
    name is "Total Region".

    In both cases, an output JSON file is written with information about the
    rasters created and the names of the target subregions.

    Args:
        source_aoi_vector_path (string): The path to the source AOI vector
            containing AOI geometries and (optionally) a "NAME" column
            (case-insensitive).
        habitat_mask_path (string): The path to a raster where pixel values of
            1 indicate the presence of habitats and pixel values of 0 or nodata
            indicate the absence of habitats.
        target_raster_dir (string): The path to a directory where rasterized
            AOI subregions should be stored.
        target_info_json (string): The path to where a target info JSON file
            should be written.

    Returns:
        ``None``
    """
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
        if fieldname.lower() == 'name':
            field_index = index
            break

    def _write_info_json(subregion_rasters_list, subregion_ids_to_names):
        """Write an info JSON file.  Abstracted into a function for DRY."""
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
        burn_value = 0  # burning onto a nodata mask (nodata=255)
        pygeoprocessing.rasterize(
            source_aoi_vector_path, target_raster_path,
            burn_values=[0], option_list=['ALL_TOUCHED=TRUE'])
        _write_info_json(
            subregion_rasters_list=[target_raster_path],
            subregion_ids_to_names={burn_value: 'Total Region'})
        return

    # If we've reached this point, then the user provided a name field and we
    # need to rasterize the sets of non-overlapping polygons.
    # If 2 features have the same name, they should have the same ID.
    # TODO: if 2 features have the same name, they should have the same Id.
    # should they, though?  If they have overlapping area, then we need to be
    # able to handle that too.
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


def _align(raster_path_map, vector_path_map, target_pixel_size,
           target_srs_wkt):
    """Align a stack of rasters and/or vectors.

    In HRA, habitats and stressors (and optionally criteria) may be defined as
    a stack of rasters and/or vectors.  This function enables this stack to be
    precisely aligned and rasterized, while minimizing sampling error in
    rasterization.

    Rasters passed in to this function will be aligned and resampled using
    nearest-neighbor interpolation.

    Vectors passed in to this function will be rasterized onto new rasters that
    align with the rest of the stack.

    All aligned rasters and rasterized vectors will have a bounding box that
    matches the union of the bounding boxes of all spatial inputs to this
    function, and with the target pixel size and SRS.

    Args:
        raster_path_map (dict): A dict mapping source raster paths to aligned
            raster paths.  This dict may be empty.
        vector_path_map (dict): A dict mapping source vector paths to aligned
            raster paths.  This dict may be empty.  These source vectors must
            already be in the target projection's SRS.  If a 'rating'
            (case-insensitive) column is present, then those values will be
            rasterized.  Otherwise, 1 will be rasterized.
        target_pixel_size (tuple): The pixel size of the target rasters, in
            the form (x, y), expressed in meters.
        target_srs_wkt (string): The target SRS of the aligned rasters.

    Returns:
        ``None``
    """
    # Step 1: Create a bounding box of the union of all input spatial rasters
    # and vectors.  To be safe, we're assuming that the source SRS of a dataset
    # might be different from the target SRS and this is taken into account.
    bounding_box_list = []
    source_raster_paths = []
    aligned_raster_paths = []
    resample_method_list = []
    for source_raster_path, aligned_raster_path in raster_path_map.items():
        source_raster_paths.append(source_raster_path)
        aligned_raster_paths.append(aligned_raster_path)
        raster_info = pygeoprocessing.get_raster_info(source_raster_path)

        # Integer (discrete) rasters should be nearest-neighbor, continuous
        # rasters should be interpolated with bilinear.
        if numpy.issubdtype(raster_info['numpy_type'], numpy.integer):
            resample_method_list.append('near')
        else:
            resample_method_list.append('bilinear')

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

    # Step 2: Align raster inputs.
    # If any rasters were provided, they will be aligned to the bounding box
    # we determined, interpolating and warping as needed.
    if raster_path_map:
        LOGGER.info(f'Aligning {len(raster_path_map)} rasters')
        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list=source_raster_paths,
            target_raster_path_list=aligned_raster_paths,
            resample_method_list=resample_method_list,
            target_pixel_size=target_pixel_size,
            bounding_box_mode=target_bounding_box,
            target_projection_wkt=target_srs_wkt,
            raster_align_index=0,  # just assume alignment with first raster
        )

    # Step 3: Rasterize vectors onto aligned rasters.
    # If any vectors were provided, they will be rasterized onto new rasters
    # that align with the bounding box we determined earlier.
    # This approach yields more precise rasters than resampling an
    # already-rasterized vector through align_and_resize_raster_stack.
    if vector_path_map:
        LOGGER.info(f'Aligning {len(vector_path_map)} vectors')
        for source_vector_path, target_raster_path in vector_path_map.items():
            # if there's a 'rating' column, then we're rasterizing a vector
            # attribute that represents a positive floating-point value.
            vector = gdal.OpenEx(source_vector_path, gdal.OF_VECTOR)
            layer = vector.GetLayer()
            raster_type = _TARGET_GDAL_TYPE_BYTE
            nodata_value = _TARGET_NODATA_BYTE
            burn_values = [1]
            rasterize_option_list = ['ALL_TOUCHED=TRUE']

            for field in layer.schema:
                fieldname = field.GetName()
                if fieldname.lower() == 'rating':
                    rasterize_option_list.append(
                        f'ATTRIBUTE={fieldname}')
                    raster_type = _TARGET_GDAL_TYPE_FLOAT32
                    nodata_value = _TARGET_NODATA_FLOAT32
                    burn_values = None
                    break

            layer = None
            vector = None

            _create_raster_from_bounding_box(
                target_raster_path=target_raster_path,
                target_bounding_box=target_bounding_box,
                target_pixel_size=target_pixel_size,
                target_pixel_type=raster_type,
                target_srs_wkt=target_srs_wkt,
                target_nodata=nodata_value,
                fill_value=nodata_value
            )

            pygeoprocessing.rasterize(
                source_vector_path, target_raster_path,
                burn_values=burn_values, option_list=rasterize_option_list)


def _create_raster_from_bounding_box(
        target_raster_path, target_bounding_box, target_pixel_size,
        target_pixel_type, target_srs_wkt, target_nodata=None,
        fill_value=None):
    """Create a raster from a given bounding box.

    Args:
        target_raster_path (string): The path to where the new raster should be
            created on disk.
        target_bounding_box (tuple): a 4-element iterable of (minx, miny,
            maxx, maxy) in projected units matching the SRS of
            ``target_srs_wkt``.
        target_pixel_size (tuple): A 2-element tuple of the (x, y) pixel size
            of the target raster.  Elements are in units of the target SRS.
        target_pixel_type (int): The GDAL GDT_* type of the target raster.
        target_srs_wkt (string): The SRS of the target raster, in Well-Known
            Text format.
        target_nodata (float): If provided, the nodata value of the target
            raster.
        fill_value=None (number): If provided, the value that the target raster
            should be filled with.

    Returns:
        ``None``
    """
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

    # Fill the band if requested.
    band = raster.GetRasterBand(1)
    if fill_value is not None:
        band.Fill(fill_value)

    # Set the nodata value.
    if target_nodata is not None:
        band.SetNoDataValue(float(target_nodata))

    band = None
    raster = None


def _simplify(source_vector_path, tolerance, target_vector_path,
              preserve_columns=None):
    """Simplify a geometry to a given tolerance.

    This function uses the GEOS SimplifyPreserveTopology function under the
    hood.  For docs, see GEOSTopologyPreserveSimplify() at
    https://libgeos.org/doxygen/geos__c_8h.html

    Args:
        source_vector_path (string): The path to a source vector to simplify.
        tolerance (number): The numerical tolerance to simplify by.
        target_vector_path (string): Where the simplified geometry should be
            stored on disk.
        preserve_columns=None (iterable or None): If provided, this is an
            iterable of string column names (case-insensitive) that should be
            carried over from the source vector to the target vector.  If
            ``None``, no columns will be carried over.

    Returns:
        ``None``.
    """
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


def _prep_input_criterion_raster(
        source_raster_path, target_filepath):
    """Prepare an input criterion raster for internal use.

    In order to make raster calculations more consistent within HRA, it's
    helpful to preprocess spatially-explicity criteria rasters for consistency.
    Rasters produced by this function will have:

        * A nodata value matching ``_TARGET_NODATA_FLOAT32``
        * Source values < 0 are converted to ``_TARGET_NODATA_FLOAT32``

    Args:
        source_raster_path (string): The path to a user-provided criterion
            raster.
        target_filepath (string): The path to where the translated raster
            should be written on disk.

    Returns:
        ``None``.
    """
    source_nodata = pygeoprocessing.get_raster_info(
        source_raster_path)['nodata'][0]

    def _translate_op(source_rating_array):
        """Translate rating pixel values.

            * Nodata values should have the new (internal) nodata.
            * Anything < 0 should become nodata.
            * Anything >= 0 is assumed to be valid and left as-is.

        Args:
            source_rating_array (numpy.array): An array of rating values.

        Returns:
            ``target_rating_array`` (numpy.array): An array with
                potentially-translated values.
        """
        target_rating_array = numpy.full(
            source_rating_array.shape, _TARGET_NODATA_FLOAT32,
            dtype=numpy.float32)
        valid_mask = (
             (~utils.array_equals_nodata(
                 source_rating_array, source_nodata)) &
             (source_rating_array >= 0.0))
        target_rating_array[valid_mask] = source_rating_array[valid_mask]
        return target_rating_array

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _translate_op, target_filepath,
        _TARGET_GDAL_TYPE_FLOAT32, _TARGET_NODATA_FLOAT32)


def _mask_binary_presence_absence_rasters(
        source_raster_paths, target_mask_path):
    """Create a mask where any values in a raster stack are 1.

    Given a stack of aligned source rasters, if any pixel values in a pixel
    stack are exactly equal to 1, the output mask at that pixel will be 1.

    This can be applied to user-defined rasters or in creating a mask across a
    stack of presence/absence rasters.

    Args:
        source_raster_paths (list): A list of string paths to rasters.
        target_mask_path (string): The path to there the mask raster
            will be written.

    Returns:
        ``None``
    """
    def _translate_op(*input_arrays):
        """Translate the input array to nodata, except where values are 1."""
        presence = numpy.full(input_arrays[0].shape, _TARGET_NODATA_BYTE,
                              dtype=numpy.uint8)
        for input_array in input_arrays:
            presence[input_array == 1] = 1
        return presence

    pygeoprocessing.raster_calculator(
        [(source_path, 1) for source_path in source_raster_paths],
        _translate_op, target_mask_path,
        _TARGET_GDAL_TYPE_BYTE, _TARGET_NODATA_BYTE)


# TODO: figure out if we really need this shared function.
def _open_table_as_dataframe(table_path, **kwargs):
    extension = os.path.splitext(table_path)[1].lower()
    # Technically, pandas.read_excel can handle xls, xlsx, xlsm, xlsb, odf, ods
    # and odt file extensions, but I have not tested anything other than XLS
    # and XLSX, so leaving this as-is from the prior HRA implementation.
    if extension in {'.xls', '.xlsx'}:
        excel_df = pandas.read_excel(table_path, **kwargs)
        excel_df.columns = excel_df.columns.str.lower()
        return excel_df
    else:
        return utils.read_csv_to_dataframe(
            table_path, sep=None, to_lower=True, engine='python', **kwargs)


def _parse_info_table(info_table_path):
    """Parse the HRA habitat/stressor info table.

    Args:
        info_table_path (string): The path to the info table.  May be either
            CSV or Excel format.  The columns 'name', 'path', 'type' and
            'stressor buffer (meters)' are all required.  The stressor buffer
            only needs values for those rows representing stressors.

    Returns:
        (habitats, stressors) (tuple): a 2-tuple of dicts where the habitat or
            stressor name (respectively) maps to attributes about that habitat
            or stressor:

                * Habitats have a structure of
                  ``{habitat_name: {'path': path to spatial layer}}``
                * Stressors have a structure of
                  ``{stressor_name: {'path': path to spatial layer, 'buffer':
                      buffer distance}}``
    """
    info_table_path = os.path.abspath(info_table_path)

    table = _open_table_as_dataframe(info_table_path)
    table = table.set_index('name')
    table = table.rename(columns={'stressor buffer (meters)': 'buffer'})

    def _make_abspath(row):
        path = row['path'].replace('\\', '/')
        if os.path.isabs(path):
            return path
        return os.path.join(
            os.path.dirname(info_table_path), path).replace('\\', '/')

    table['path'] = table.apply(lambda row: _make_abspath(row), axis=1)

    # Drop the buffer column from the habitats list; we don't need it.
    habitats = table.loc[table['type'] == 'habitat'].drop(
        columns=['type', 'buffer']).to_dict(orient='index')

    # Keep the buffer column in the stressors dataframe.
    stressors = table.loc[table['type'] == 'stressor'].drop(
        columns=['type']).to_dict(orient='index')

    # TODO: check that habitats and stressor names are nonoverlapping sets.

    return (habitats, stressors)


# TODO: validate spatial criteria can be opened by GDAL.
def _parse_criteria_table(criteria_table_path, target_composite_csv_path):
    # This function requires that the table is read as a numpy array, so it's
    # easiest to read the table directly.
    extension = os.path.splitext(criteria_table_path)[1].lower()
    if extension in {'.xls', '.xlsx'}:
        df = pandas.read_excel(criteria_table_path, header=None)
    else:
        df = pandas.read_csv(criteria_table_path, header=None, sep=None,
                             engine='python')
    table = df.to_numpy()

    # Habitats are loaded from the top row (table[0])
    known_habitats = set(table[0]).difference(
        {'HABITAT NAME', numpy.nan, 'CRITERIA TYPE'})

    # Stressors are loaded from the first column (table[:, 0])
    overlap_section_header = 'HABITAT STRESSOR OVERLAP PROPERTIES'
    known_stressors = set()
    for row_index, value in enumerate(table[:, 0]):
        try:
            if value == overlap_section_header or numpy.isnan(value):
                known_stressors.add(table[row_index + 1, 0])
        except TypeError:
            # calling numpy.isnan on a string raises a TypeError.
            pass
    # The overlap section header is presented after an empty line, so it'll
    # normally show up in this set.  Remove it.
    # It's also all too easy to end up with multiple nan rows between sections,
    # so if a numpy.nan ends up in the set, remove it.
    for value in (overlap_section_header, numpy.nan):
        try:
            known_stressors.remove(value)
        except KeyError:
            pass

    # Fill in habitat names in the table's top row for easier reference.
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

    records = []
    current_stressor = None
    for row_index, row in enumerate(table[1:], start=1):
        if row[0] == overlap_section_header:
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
                try:
                    # Just a test to see if this value is a number.
                    # It's OK if it stays an int.
                    float(attribute_value)
                except ValueError:
                    # If we can't cast it to a float, assume it's a string path
                    # to a raster or vector.
                    attribute_value = attribute_value.replace('\\', '/')
                    if not os.path.isabs(attribute_value):
                        attribute_value = os.path.join(
                            os.path.dirname(criteria_table_path),
                            attribute_value)
                stressor_habitat_data[
                    attribute_name.lower()] = attribute_value
            # Keep the copy() unless you want to all of the 'records' dicts to
            # have the same contents!  This is normal behavior given python's
            # memory model.
            records.append(stressor_habitat_data.copy())

    # the primary key of this table is (habitat, stressor, criterion)
    overlap_df = pandas.DataFrame.from_records(
        records, columns=['habitat', 'stressor', 'criterion', 'rating', 'dq',
                          'weight', 'e/c'])
    overlap_df.to_csv(target_composite_csv_path, index=False)

    return (known_habitats, known_stressors)


def _calculate_decayed_distance(stressor_raster_path, decay_type,
                                buffer_distance, target_edt_path):
    """Decay the influence of a stressor given decay type and buffer distance.

    Args:
        stressor_raster_path (string): The path to a stressor raster, where
            pixel values of 1 indicate presence of the stressor and 0 or nodata
            indicate absence of the stressor.
        decay_type (string): The type of decay.  Valid values are:

            * ``linear``: Pixel values decay linearly from the stressor pixels
              out to the buffer distance.
            * ``exponential``: Pixel values decay exponentially out to the
              buffer distance.
            * ``none``: Pixel values do not decay out to the buffer distance.
              Instead, all pixels within the buffer distance have the same
              influence as though it were overlapping the stressor itself.
        buffer_distance (number): The distance out to which the stressor has an
            influence.  This is in linearly projected units and should be in
            meters.
        target_edt_path (string): The path where the target EDT raster should
            be written.

    Returns:
        ``None``

    Raises:
        ``AssertionError``: When an invalid ``decay_type`` is provided.
    """
    pygeoprocessing.distance_transform_edt((stressor_raster_path, 1),
                                           target_edt_path)
    # We're assuming that we're working with square pixels
    pixel_size = abs(pygeoprocessing.get_raster_info(
        stressor_raster_path)['pixel_size'][0])
    buffer_distance_in_pixels = buffer_distance / pixel_size

    target_edt_raster = gdal.OpenEx(target_edt_path, gdal.GA_Update)
    target_edt_band = target_edt_raster.GetRasterBand(1)
    edt_nodata = target_edt_band.GetNoDataValue()
    decay_type = decay_type.lower()
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


def _calc_criteria(attributes_list, habitat_mask_raster_path,
                   target_criterion_path,
                   decayed_edt_raster_path=None):
    """Calculate Exposure or Consequence for a single habitat/stressor pair.

    Args:
        attributes_list (list): A list of dicts for all of the criteria for
            this criterion type (E or C).  Each dict must have the following
            keys:

                * ``rating``: A numeric criterion rating (if consistent across
                  the whole study area) or the path to a raster with
                  spatially-explicit ratings.  A rating raster must align with
                  ``habitat_mask_raster_path``.
                * ``dq``: The numeric data quality rating for this criterion.
                * ``weight``: The numeric weight for this criterion.

        habitat_mask_raster_path (string): The path to a raster with pixel
            values of 1 indicating presence of habitats, and 0 or nodata
            representing the absence of habitats.
        target_criterion_path (string): The path to where the calculated
            criterion layer should be written.
        decayed_edt_raster_path=None (string or None): If provided, this is the
            path to an raster of weights that should be applied to the
            numerator of the criterion calculation.

    Returns:
        ``None``
    """
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
            # A rating of 0 means that the criterion should be ignored.
            # RATING may be either a number or a raster.
            try:
                rating = float(attribute_dict['rating'])
                if rating == 0:
                    continue
            except ValueError:
                # When rating is a string filepath, it represents a raster.
                try:
                    # Opening a raster is fairly inexpensive, so it should be
                    # fine to re-open the raster on each block iteration.
                    rating_raster = gdal.OpenEx(attribute_dict['rating'])
                    rating_band = rating_raster.GetRasterBand(1)
                    rating_nodata = rating_band.GetNoDataValue()
                    rating = rating_band.ReadAsArray(**block_info)[valid_mask]

                    # Any habitat pixels with a nodata rating (no rating
                    # specified by the user) should be
                    # interpreted as having a rating of 0.
                    rating[utils.array_equals_nodata(
                        rating, rating_nodata)] = 0
                finally:
                    rating_band = None
                    rating_raster = None
            data_quality = attribute_dict['dq']
            weight = attribute_dict['weight']

            # The (data_quality + weight) denominator running sum is
            # re-calculated for each block.  While this is inefficient,
            # ``dq`` and ``weight`` are always scalars and so the wasted CPU
            # time is pretty trivial, even on large habitat/stressor matrices.
            # Plus, it's way easier to read and more maintainable to just have
            # everything be recalculated.
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
    decayed_edt_band = None
    decayed_edt_raster = None
    target_criterion_band = None
    target_criterion_raster = None


def _calculate_pairwise_risk(habitat_mask_raster_path, exposure_raster_path,
                             consequence_raster_path, risk_equation,
                             target_risk_raster_path):
    """Calculate risk for a single habitat/stressor pair.

    Args:
        habitat_mask_raster_path (string): The path to a habitat mask raster,
            where pixels with a value of 1 represent the presence of habitats,
            and values of 0 or nodata represent the absence of habitats.
        exposure_raster_path (string): The path to a raster of computed
            exposure scores.  Must align with ``habitat_mask_raster_path``.
        consequence_raster_path (string): The path to a raster of computed
            consequence scores.  Must align with ``habitat_mask_raster_path``.
        risk_equation (string): The risk equation to use.  Must be one of
            ``multiplicative`` or ``euclidean``.

    Returns:
        ``None``

    Raises:
        ``AssertionError`` when an invalid risk equation is provided.
    """
    risk_equation = risk_equation.lower()
    if risk_equation not in _VALID_RISK_EQS:
        raise AssertionError(
            f'Invalid risk equation {risk_equation} provided')

    def _calculate_risk(habitat_mask, exposure, consequence):
        habitat_pixels = (habitat_mask == 1)
        risk_array = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLOAT32,
                                dtype=numpy.float32)
        if risk_equation == 'multiplicative':
            risk_array[habitat_pixels] = (
                exposure[habitat_pixels] * consequence[habitat_pixels])
        else:  # risk_equation == 'euclidean':
            # The numpy.maximum guards against low E, C values ending up less
            # than 1 and, when squared, have positive, larger-than-reasonable
            # risk values.
            risk_array[habitat_pixels] = numpy.sqrt(
                numpy.maximum(0, (exposure[habitat_pixels] - 1)) ** 2 +
                numpy.maximum(0, (consequence[habitat_pixels] - 1)) ** 2)
        return risk_array

    pygeoprocessing.raster_calculator(
        [(habitat_mask_raster_path, 1),
         (exposure_raster_path, 1),
         (consequence_raster_path, 1)],
        _calculate_risk, target_risk_raster_path, _TARGET_GDAL_TYPE_FLOAT32,
        _TARGET_NODATA_FLOAT32)


def _reclassify_score(habitat_mask, max_pairwise_risk, scores):
    """Reclassify risk scores into high/medium/low.

    The output raster will break values into 3 buckets based on the
    ``max_pairwise_risk`` (shortened to "MPR"):

        * Scores in the range [ 0, MPR*(1/3) ) have a value of 1
          indicating low risk.
        * Scores in the range [ MPR*(1/3), MPR*(2/3) ) have a value of 2
          indicating medium risk.
        * Scores in the range [ MPR*(2/3), infinity ) have a value of 3
          indicating high risk.

    Args:
        habitat_mask (numpy.array): A numpy array where 1 indicates presence of
            habitats and 0 or ``_TARGET_NODATA_BYTE`` indicate absence.
        max_pairwise_risk (float): The maximum likely pairwise risk value.
        scores (numpy.array): A numpy array of floating-point risk scores.

    Returns:
        ``reclassified`` (numpy.array): An unsigned byte numpy array of a
            shape/size matching ``habitat_mask`` and ``scores``.
    """
    habitat_pixels = (habitat_mask == 1)
    reclassified = numpy.full(habitat_mask.shape, _TARGET_NODATA_BYTE,
                              dtype=numpy.uint8)
    reclassified[habitat_pixels] = numpy.digitize(
        scores[habitat_pixels],
        [0, max_pairwise_risk*(1/3), max_pairwise_risk*(2/3)],
        right=True)  # bins[i-1] >= x > bins[i]
    return reclassified


def _maximum_reclassified_score(habitat_mask, *risk_classes):
    """Determine the maximum risk score in a stack of risk rasters.

    Args:
        habitat_mask (numpy.array): A numpy array where values of 1 indicate
            presence of habitats.  Values of 0 or ``_TARGET_NODATA_BYTE``
            indicate absence of habitats.
        *risk_classes (list of numpy.arrays): A variable number of unsigned
            integer reclassified risk arrays.

    Returns:
        A numpy.array with each pixel that has habitats on it has the highest
        risk score in the ``risk_classes`` stack of arrays.
    """
    target_matrix = numpy.zeros(habitat_mask.shape, dtype=numpy.uint8)
    valid_pixels = (habitat_mask == 1)

    for risk_class_matrix in risk_classes:
        target_matrix[valid_pixels] = numpy.maximum(
            target_matrix[valid_pixels], risk_class_matrix[valid_pixels])

    target_matrix[~valid_pixels] = _TARGET_NODATA_BYTE
    return target_matrix


def _sum_rasters(raster_path_list, target_nodata, target_result_path,
                 normalize=False):
    """Sum a stack of rasters.

    Where all rasters agree about nodata, the output raster will also be
    nodata.  Otherwise, pixel values will be the sum of the stack, where nodata
    values are converted to 0.

    Args:
        raster_path_list (list): list of raster paths to sum
        target_nodata (float): desired target nodata value
        target_result_path (string): path to write out the sum raster
        normalize=False (bool): whether to normalize each pixel value by the
            number of valid pixels in the stack.  Defaults to False.

    Returns:
        ``None``
    """
    nodata_list = [pygeoprocessing.get_raster_info(
        path)['nodata'][0] for path in raster_path_list]

    def _sum_op(*array_list):
        result = numpy.zeros(array_list[0].shape, dtype=numpy.float32)
        pixels_have_valid_values = numpy.zeros(result.shape, dtype=bool)
        valid_pixel_count = numpy.zeros(result.shape, dtype=numpy.uint16)
        for array, nodata in zip(array_list, nodata_list):
            non_nodata_pixels = ~utils.array_equals_nodata(array, nodata)
            pixels_have_valid_values |= non_nodata_pixels
            valid_pixel_count += non_nodata_pixels

            result[non_nodata_pixels] += array[non_nodata_pixels]

        if normalize:
            result[pixels_have_valid_values] /= valid_pixel_count[pixels_have_valid_values]
        result[~pixels_have_valid_values] = target_nodata
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list],
        _sum_op, target_result_path, gdal.GDT_Float32, target_nodata)


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
