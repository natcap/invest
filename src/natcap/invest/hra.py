"""Habitat risk assessment (HRA) model for InVEST."""
# -*- coding: UTF-8 -*-
import logging
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
from osgeo import gdal, ogr, osr

from . import spec_utils, utils, validation
from .model_metadata import MODEL_METADATA
from .spec_utils import u
from . import gettext


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
    "userguide": MODEL_METADATA["habitat_risk_assessment"].userguide,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "info_table_path": {
            "name": gettext("habitat stressor table"),
            "about": gettext("A table describing each habitat and stressor."),
            "type": "csv",
            "columns": {
                "name": {
                    "type": "freestyle_string",
                    "about": gettext(
                        "A unique name for each habitat or stressor. These "
                        "names must match the habitat and stressor names in "
                        "the Criteria Scores Table.")},
                "path": {
                    "type": {"vector", "raster"},
                    "bands": {1: {
                        "type": "number",
                        "units": u.none,
                        "about": gettext(
                            "Pixel values are 1, indicating presence of the "
                            "habitat/stressor, or 0 indicating absence. Any "
                            "values besides 0 or 1 will be treated as 0.")
                    }},
                    "fields": {},
                    "geometries": spec_utils.POLYGONS,
                    "about": gettext(
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
                        "habitat": {"description": gettext("habitat")},
                        "stressor": {"description": gettext("stressor")}
                    },
                    "about": gettext(
                        "Whether this row is for a habitat or a stressor.")
                },
                "stressor buffer (meters)": {
                    "type": "number",
                    "units": u.meter,
                    "about": gettext(
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
            "name": gettext("criteria scores table"),
            "about": gettext(
                "A table of criteria scores for all habitats and stressors."),
            "type": "csv",
            "excel_ok": True,
        },
        "resolution": {
            "name": gettext("resolution of analysis"),
            "about": gettext(
                "The resolution at which to run the analysis. The model "
                "outputs will have this resolution."),
            "type": "number",
            "units": u.meter,
            "expression": "value > 0",
        },
        "max_rating": {
            "name": gettext("maximum criteria score"),
            "about": gettext(
                "The highest possible criteria score in the scoring system."),
            "type": "number",
            "units": u.none,
            "expression": "value > 0"
        },
        "risk_eq": {
            "name": gettext("risk equation"),
            "about": gettext(
                "The equation to use to calculate risk from exposure and "
                "consequence."),
            "type": "option_string",
            "options": {
                "Multiplicative": {"display_name": gettext("multiplicative")},
                "Euclidean": {"display_name": gettext("Euclidean")}
            }
        },
        "decay_eq": {
            "name": gettext("decay equation"),
            "about": gettext(
                "The equation to model effects of stressors in buffer areas."),
            "type": "option_string",
            "options": {
                "None": {
                    "display_name": gettext("none"),
                    "description": gettext(
                        "No decay. Stressor has full effect in the buffer "
                        "area.")},
                "Linear": {
                    "display_name": gettext("linear"),
                    "description": gettext(
                        "Stressor effects in the buffer area decay linearly "
                        "with distance from the stressor.")},
                "Exponential": {
                    "display_name": gettext("exponential"),
                    "description": gettext(
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
                    "about": gettext(
                        "Uniquely identifies each feature. Required if "
                        "the vector contains more than one feature.")
                }
            },
            "about": gettext(
                "A GDAL-supported vector file containing feature containing "
                "one or more planning regions or subregions."),
        },
        "visualize_outputs": {
            "name": gettext("Generate GeoJSONs"),
            "about": gettext("Generate GeoJSON outputs for web visualization."),
            "type": "boolean"
        }
    }
}


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
    LOGGER.info('Validating arguments')
    invalid_parameters = validate(args)
    if invalid_parameters:
        raise ValueError("Invalid parameters passed: %s" % invalid_parameters)

    # Validate and store inputs
    LOGGER.info('Validating criteria table file and return cleaned dataframe.')
    criteria_df = _get_criteria_dataframe(args['criteria_table_path'])

    # Create initial working directories and determine file suffixes
    intermediate_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    file_preprocessing_dir = os.path.join(
        intermediate_dir, 'file_preprocessing')
    output_dir = os.path.join(args['workspace_dir'], 'outputs')
    work_dirs = [output_dir, intermediate_dir, file_preprocessing_dir]

    # Add visualization_outputs folder if in an electron-Node.js based UI
    if args['visualize_outputs']:
        viz_dir = os.path.join(args['workspace_dir'], 'visualization_outputs')
        work_dirs.append(viz_dir)
    utils.make_directories(work_dirs)
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Initialize a TaskGraph
    taskgraph_working_dir = os.path.join(
        intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_working_dir, n_workers)

    # Calculate recovery for each habitat, and overlap scores for each
    # habitat-stressor, and store data in the dataframes
    info_df, habitat_names, stressor_names = _get_info_dataframe(
        args['info_table_path'], file_preprocessing_dir, intermediate_dir,
        output_dir, file_suffix)
    resilience_attributes, stressor_attributes = \
        _get_attributes_from_df(criteria_df, habitat_names, stressor_names)
    max_rating = float(args['max_rating'])
    recovery_df = _get_recovery_dataframe(
        criteria_df, habitat_names, resilience_attributes, max_rating,
        file_preprocessing_dir, intermediate_dir, file_suffix)
    overlap_df = _get_overlap_dataframe(
        criteria_df, habitat_names, stressor_attributes, max_rating,
        file_preprocessing_dir, intermediate_dir, file_suffix)

    # Append spatially explicit criteria rasters to info_df
    criteria_file_dir = os.path.dirname(args['criteria_table_path'])
    info_df = _append_spatial_raster_row(
        info_df, recovery_df, overlap_df, criteria_file_dir,
        file_preprocessing_dir, file_suffix)

    # Get target projection from the AOI vector file
    if 'aoi_vector_path' in args and args['aoi_vector_path'] != '':
        target_sr_wkt = pygeoprocessing.get_vector_info(
            args['aoi_vector_path'])['projection_wkt']
        target_sr = osr.SpatialReference()
        if target_sr_wkt:
            target_sr.ImportFromWkt(target_sr_wkt)
        if not target_sr.IsProjected():
            raise ValueError(
                'The AOI vector file %s is provided but not projected.' %
                args['aoi_vector_path'])
        else:
            # Get the value to multiply by linear distances in order to
            # transform them to meters
            linear_unit = target_sr.GetLinearUnits()
            LOGGER.info(
                'Target projection from AOI: %s. EPSG: %s. Linear unit: '
                '%s.' % (target_sr.GetAttrValue('PROJECTION'),
                         target_sr.GetAttrValue("AUTHORITY", 1), linear_unit))

    # Rasterize habitat and stressor layers if they are vectors.
    # Divide resolution (meters) by linear unit to convert to projection units
    target_pixel_size = (float(args['resolution'])/linear_unit,
                         -float(args['resolution'])/linear_unit)

    # Simplify the AOI vector for faster run on zonal statistics
    simplified_aoi_vector_path = os.path.join(
        file_preprocessing_dir, 'simplified_aoi%s.gpkg' % file_suffix)
    aoi_tolerance = (float(args['resolution']) / linear_unit) / 2

    # Check if subregion field exists in the AOI vector
    subregion_field_exists = _has_field_name(
        args['aoi_vector_path'], _SUBREGION_FIELD_NAME)

    # Simplify the AOI and preserve the subregion field if it exists
    aoi_preserved_field = None
    aoi_field_name = None
    if subregion_field_exists:
        aoi_preserved_field = (_SUBREGION_FIELD_NAME, ogr.OFTString)
        aoi_field_name = _SUBREGION_FIELD_NAME
        LOGGER.info('Simplifying AOI vector while preserving field %s.' %
                    aoi_field_name)
    else:
        LOGGER.info('Simplifying AOI vector without subregion field.')

    simplify_aoi_task = task_graph.add_task(
        func=_simplify_geometry,
        args=(args['aoi_vector_path'], aoi_tolerance,
              simplified_aoi_vector_path),
        kwargs={'preserved_field': aoi_preserved_field},
        target_path_list=[simplified_aoi_vector_path],
        task_name='simplify_aoi_vector')

    # Use the simplified AOI vector to run analyses
    aoi_vector_path = simplified_aoi_vector_path

    # Rasterize AOI vector for later risk statistics calculation
    LOGGER.info('Rasterizing AOI vector.')
    rasterized_aoi_pickle_path = os.path.join(
        file_preprocessing_dir, 'rasterized_aoi_dictionary%s.pickle' %
        file_suffix)

    rasterize_aoi_dependent_tasks = [simplify_aoi_task]
    # Rasterize AOI vector geometries. If field name doesn't exist, rasterize
    # the entire vector onto a raster with values of 1
    if aoi_field_name is None:
        # Fill the raster with 1s on where a vector geometry touches any pixel
        # on the raster
        target_raster_path = os.path.join(
            file_preprocessing_dir, 'rasterized_simplified_aoi%s.tif' %
            file_suffix)
        create_raster_task = task_graph.add_task(
            func=pygeoprocessing.create_raster_from_vector_extents,
            args=(aoi_vector_path, target_raster_path,
                  target_pixel_size, _TARGET_PIXEL_INT, _TARGET_NODATA_INT),
            target_path_list=[target_raster_path],
            task_name='rasterize_single_AOI_vector',
            dependent_task_list=rasterize_aoi_dependent_tasks)
        rasterize_aoi_dependent_tasks.append(create_raster_task)

        # Fill the raster with 1s on where a vector geometry exists
        rasterize_kwargs = {'burn_values': [1],
                            'option_list': ["ALL_TOUCHED=TRUE"]}
        task_graph.add_task(
            func=pygeoprocessing.rasterize,
            args=(aoi_vector_path, target_raster_path),
            kwargs=rasterize_kwargs,
            target_path_list=[target_raster_path],
            task_name='rasterize_single_vector',
            dependent_task_list=rasterize_aoi_dependent_tasks)

        pickle.dump(
            {_TOTAL_REGION_NAME: target_raster_path},
            open(rasterized_aoi_pickle_path, 'wb'))

    # If field name exists., rasterize AOI geometries with same field value
    # onto separate rasters
    else:
        geom_pickle_path = os.path.join(
            file_preprocessing_dir, 'aoi_geometries%s.pickle' % file_suffix)

        get_vector_geoms_task = task_graph.add_task(
            func=_get_vector_geometries_by_field,
            args=(aoi_vector_path, aoi_field_name, geom_pickle_path),
            target_path_list=[geom_pickle_path],
            task_name='get_AOI_vector_geoms_by_field_"%s"' % aoi_field_name,
            dependent_task_list=rasterize_aoi_dependent_tasks)
        rasterize_aoi_dependent_tasks.append(get_vector_geoms_task)

        task_graph.add_task(
            func=_create_rasters_from_geometries,
            args=(geom_pickle_path, file_preprocessing_dir,
                  rasterized_aoi_pickle_path, target_pixel_size),
            target_path_list=[rasterized_aoi_pickle_path],
            task_name='create_rasters_from_AOI_geometries',
            dependent_task_list=rasterize_aoi_dependent_tasks)

    # Create a raster from vector extent with 0's, then burn the vector
    # onto the raster with 1's, for all the H/S layers that are not a raster
    align_and_resize_dependency_list = []
    for _, row in info_df.iterrows():
        if not row['IS_RASTER']:
            vector_name = row['NAME']
            vector_type = row['TYPE']
            vector_path = row['PATH']
            simplified_vector_path = row['SIMPLE_VECTOR_PATH']
            tolerance = (float(args['resolution']) / row['LINEAR_UNIT']) / 2
            target_raster_path = row['BASE_RASTER_PATH']
            LOGGER.info('Rasterizing %s vector.' % vector_name)

            # Simplify the vector geometry first, with a tolerance of half the
            # target resolution
            simplify_geometry_task = task_graph.add_task(
                func=_simplify_geometry,
                args=(vector_path, tolerance, simplified_vector_path),
                kwargs={'preserved_field': (_RATING_FIELD, ogr.OFTReal)},
                target_path_list=[simplified_vector_path],
                task_name='simplify_%s_vector' % vector_name)

            rasterize_kwargs = {'burn_values': None, 'option_list': None}
            if vector_type == _SPATIAL_CRITERIA_TYPE:
                # Fill value for the target raster should be nodata float,
                # since criteria rating could be float
                fill_value = _TARGET_NODATA_FLT

                # If it's a spatial criteria vector, burn the values from the
                # ``rating`` attribute
                rasterize_kwargs['option_list'] = [
                    "ATTRIBUTE=" + _RATING_FIELD]
                rasterize_nodata = _TARGET_NODATA_FLT
                rasterize_pixel_type = _TARGET_PIXEL_FLT

            else:  # Could be a habitat or stressor vector
                # Initial fill values for the target raster should be nodata
                # int
                fill_value = _TARGET_NODATA_INT

                # Fill the raster with 1s on where a vector geometry exists
                rasterize_kwargs['burn_values'] = [1]
                rasterize_kwargs['option_list'] = ["ALL_TOUCHED=TRUE"]
                rasterize_nodata = _TARGET_NODATA_INT
                rasterize_pixel_type = _TARGET_PIXEL_INT

            align_and_resize_dependency_list.append(task_graph.add_task(
                func=_create_raster_and_rasterize_vector,
                args=(simplified_vector_path, target_raster_path,
                      target_pixel_size, rasterize_pixel_type,
                      rasterize_nodata, fill_value, rasterize_kwargs),
                target_path_list=[target_raster_path],
                task_name='rasterize_%s' % vector_name,
                dependent_task_list=[simplify_geometry_task]))

    # Align and resize all the rasters, including rasters provided by the user,
    # and rasters created from the vectors.
    base_raster_list = info_df.BASE_RASTER_PATH.tolist()
    align_raster_list = info_df.ALIGN_RASTER_PATH.tolist()

    LOGGER.info('Starting align_and_resize_raster_task.')
    align_and_resize_rasters_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(base_raster_list, align_raster_list,
              [_RESAMPLE_METHOD] * len(base_raster_list),
              target_pixel_size, 'union'),
        kwargs={'target_projection_wkt': target_sr_wkt},
        target_path_list=align_raster_list,
        task_name='align_and_resize_raster_task',
        dependent_task_list=align_and_resize_dependency_list)

    # Make buffer stressors based on their impact distance and decay function
    align_stressor_raster_list = info_df[
        info_df.TYPE == _STRESSOR_TYPE].ALIGN_RASTER_PATH.tolist()
    dist_stressor_raster_list = info_df[
        info_df.TYPE == _STRESSOR_TYPE].DIST_RASTER_PATH.tolist()
    stressor_names = info_df[info_df.TYPE == _STRESSOR_TYPE].NAME.tolist()

    LOGGER.info('Calculating euclidean distance transform on stressors.')
    # Convert pixel size from meters to projection unit
    sampling_distance = (float(args['resolution'])/linear_unit,
                         float(args['resolution'])/linear_unit)
    distance_transform_tasks = []
    for (align_raster_path, dist_raster_path, stressor_name) in zip(
        align_stressor_raster_list, dist_stressor_raster_list,
            stressor_names):

        distance_transform_task = task_graph.add_task(
            func=pygeoprocessing.distance_transform_edt,
            args=((align_raster_path, 1), dist_raster_path),
            kwargs={'sampling_distance': sampling_distance,
                    'working_dir': intermediate_dir},
            target_path_list=[dist_raster_path],
            task_name='distance_transform_on_%s' % stressor_name,
            dependent_task_list=[align_and_resize_rasters_task])
        distance_transform_tasks.append(distance_transform_task)

    LOGGER.info('Calculating number of habitats on each pixel.')
    align_habitat_raster_list = info_df[
        info_df.TYPE == _HABITAT_TYPE].ALIGN_RASTER_PATH.tolist()
    habitat_path_band_list = [
        (raster_path, 1) for raster_path in align_habitat_raster_list]
    habitat_count_raster_path = os.path.join(
        file_preprocessing_dir, 'habitat_count%s.tif' % file_suffix)
    count_habitat_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(habitat_path_band_list, _count_habitats_op,
              habitat_count_raster_path, gdal.GDT_Byte, _TARGET_NODATA_INT),
        target_path_list=[habitat_count_raster_path],
        task_name='counting_habitats',
        dependent_task_list=[align_and_resize_rasters_task])

    # A dependent task list for calculating ecosystem risk from all habitat
    # risk rasters
    ecosystem_risk_dependent_tasks = [count_habitat_task]

    # For each habitat, calculate the individual and cumulative exposure,
    # consequence, and risk scores from each stressor.
    for habitat in habitat_names:
        LOGGER.info('Calculating recovery scores on habitat %s.' % habitat)
        # Get a dataframe with information on raster paths for the habitat.
        habitat_info_df = info_df.loc[info_df.NAME == habitat]

        # On a habitat raster, a pixel value of 0 indicates the existence of
        # habitat, whereas 1 means non-existence.
        habitat_raster_path = habitat_info_df['ALIGN_RASTER_PATH'].item()
        habitat_recovery_df = recovery_df.loc[habitat]
        recovery_raster_path = habitat_recovery_df['R_RASTER_PATH']
        recovery_num_raster_path = habitat_recovery_df[
            'R_NUM_RASTER_PATH']
        habitat_recovery_denom = habitat_recovery_df['R_DENOM']

        calc_habitat_recovery_task_list = []
        calc_habitat_recovery_task_list.append(
            task_graph.add_task(
                func=_calc_habitat_recovery,
                args=(habitat_raster_path, habitat_recovery_df, max_rating),
                target_path_list=[
                    recovery_raster_path, recovery_num_raster_path],
                task_name='calculate_%s_recovery' % habitat,
                dependent_task_list=[align_and_resize_rasters_task]))

        total_expo_dependent_tasks = []
        total_conseq_dependent_tasks = []
        total_risk_dependent_tasks = []

        # Calculate exposure/consequence scores on each stressor-habitat pair
        for (distance_transform_task, stressor) in zip(
                distance_transform_tasks, stressor_names):
            LOGGER.info('Calculating exposure, consequence, and risk scores '
                        'from stressor %s to habitat %s.' %
                        (stressor, habitat))

            # Get a dataframe with information on distance raster path,
            # buffer distance, and linear unit for the stressor
            stressor_info_df = info_df.loc[info_df.NAME == stressor]

            # Get habitat-stressor overlap dataframe with information on
            # numerator, denominator, spatially explicit criteria files, and
            # target paths
            habitat_stressor_overlap_df = overlap_df.loc[(habitat, stressor)]

            stressor_dist_raster_path = stressor_info_df[
                'DIST_RASTER_PATH'].item()

            # Convert stressor buffer from meters to projection unit
            stressor_buffer = stressor_info_df[_BUFFER_HEADER].item() / float(
                stressor_info_df['LINEAR_UNIT'].item())

            # Calculate exposure scores on each habitat-stressor pair
            pair_expo_target_path_list = [
                habitat_stressor_overlap_df.loc[raster_path] for
                raster_path in ['E_NUM_RASTER_PATH', 'E_RASTER_PATH']]

            pair_expo_task = task_graph.add_task(
                func=_calc_pair_criteria_score,
                args=(habitat_stressor_overlap_df, habitat_raster_path,
                      stressor_dist_raster_path, stressor_buffer,
                      args['decay_eq'], 'E'),
                target_path_list=pair_expo_target_path_list,
                task_name='calculate_%s_%s_exposure' % (habitat, stressor),
                dependent_task_list=[
                    align_and_resize_rasters_task, distance_transform_task])
            total_expo_dependent_tasks.append(pair_expo_task)

            # Calculate consequence scores on each habitat-stressor pair.
            # Add recovery numerator and denominator to the scores
            pair_conseq_target_path_list = [
                habitat_stressor_overlap_df.loc[raster_path] for
                raster_path in ['C_NUM_RASTER_PATH', 'C_RASTER_PATH']]
            pair_conseq_task = task_graph.add_task(
                func=_calc_pair_criteria_score,
                args=(habitat_stressor_overlap_df, habitat_raster_path,
                      stressor_dist_raster_path, stressor_buffer,
                      args['decay_eq'], 'C'),
                kwargs={'recov_params':
                        (recovery_num_raster_path, habitat_recovery_denom)},
                target_path_list=pair_conseq_target_path_list,
                task_name='calculate_%s_%s_consequence' % (habitat, stressor),
                dependent_task_list=[
                    align_and_resize_rasters_task,
                    distance_transform_task] + calc_habitat_recovery_task_list)
            total_conseq_dependent_tasks.append(pair_conseq_task)

            # Calculate pairwise habitat-stressor risks.
            pair_e_raster_path, pair_c_raster_path, \
                target_pair_risk_raster_path = [
                    habitat_stressor_overlap_df.loc[path] for path in
                    ['E_RASTER_PATH', 'C_RASTER_PATH',
                     'PAIR_RISK_RASTER_PATH']]
            pair_risk_calculation_list = [
                (pair_e_raster_path, 1), (pair_c_raster_path, 1),
                ((max_rating, 'raw')), (args['risk_eq'], 'raw')]
            pair_risk_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=(pair_risk_calculation_list, _pair_risk_op,
                      target_pair_risk_raster_path, _TARGET_PIXEL_FLT,
                      _TARGET_NODATA_FLT),
                target_path_list=[target_pair_risk_raster_path],
                task_name='calculate_%s_%s_risk' % (habitat, stressor),
                dependent_task_list=[pair_expo_task, pair_conseq_task])
            total_risk_dependent_tasks.append(pair_risk_task)

        # Calculate cumulative E, C & risk scores on each habitat
        total_e_habitat_path = habitat_info_df['TOT_E_RASTER_PATH'].item()
        total_c_habitat_path = habitat_info_df['TOT_C_RASTER_PATH'].item()

        LOGGER.info(
            'Calculating total exposure scores on habitat %s.' % habitat)
        habitat_overlap_df = overlap_df.loc[habitat]
        e_num_path_const_list = [
            (path, 1) for path in
            habitat_overlap_df['E_NUM_RASTER_PATH'].tolist()]
        e_denom_list = [
            (denom, 'raw') for denom in habitat_overlap_df['E_DENOM'].tolist()]

        total_e_path_band_list = list(
            [(habitat_raster_path, 1)] + e_num_path_const_list + e_denom_list)

        # Calculate total exposure on the habitat
        task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(total_e_path_band_list,
                  _total_exposure_op,
                  total_e_habitat_path,
                  _TARGET_PIXEL_FLT,
                  _TARGET_NODATA_FLT),
            target_path_list=[total_e_habitat_path],
            task_name='calculate_total_exposure_%s' % habitat,
            dependent_task_list=total_expo_dependent_tasks)

        LOGGER.info(
            'Calculating total consequence scores on habitat %s.' % habitat)
        recov_num_raster_path = habitat_recovery_df['R_NUM_RASTER_PATH']
        c_num_path_const_list = [(path, 1) for path in habitat_overlap_df[
            'C_NUM_RASTER_PATH'].tolist()]
        c_denom_list = [(denom, 'raw') for denom in habitat_overlap_df[
            'C_DENOM'].tolist()]

        total_c_path_const_list = list(
            [(habitat_raster_path, 1), (recov_num_raster_path, 1),
             (habitat_recovery_denom, 'raw')] +
            c_num_path_const_list + c_denom_list)

        # Calculate total consequence on the habitat
        task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(total_c_path_const_list,
                  _total_consequence_op,
                  total_c_habitat_path,
                  _TARGET_PIXEL_FLT,
                  _TARGET_NODATA_FLT),
            target_path_list=[total_c_habitat_path],
            task_name='calculate_total_consequence_%s' % habitat,
            dependent_task_list=total_conseq_dependent_tasks)

        LOGGER.info('Calculating total risk score and reclassified risk scores'
                    ' on habitat %s.' % habitat)

        total_habitat_risk_path, reclass_habitat_risk_path = [
            habitat_info_df[column_header].item() for column_header in [
                'TOT_RISK_RASTER_PATH', 'RECLASS_RISK_RASTER_PATH']]

        # Get a list of habitat path and individual risk paths on that habitat
        # for the final risk calculation
        total_risk_path_band_list = [(habitat_raster_path, 1)]
        pair_risk_path_list = habitat_overlap_df[
            'PAIR_RISK_RASTER_PATH'].tolist()
        total_risk_path_band_list = total_risk_path_band_list + [
            (path, 1) for path in pair_risk_path_list]

        # Calculate the cumulative risk on the habitat from all stressors
        calc_risk_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(total_risk_path_band_list, _tot_risk_op,
                  total_habitat_risk_path, _TARGET_PIXEL_FLT,
                  _TARGET_NODATA_FLT),
            target_path_list=[total_habitat_risk_path],
            task_name='calculate_%s_risk' % habitat,
            dependent_task_list=total_risk_dependent_tasks)
        ecosystem_risk_dependent_tasks.append(calc_risk_task)

        # Reclassify the risk score into three categories by dividing the total
        # risk score by 3, and return the ceiling
        task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(total_habitat_risk_path, 1), (max_rating, 'raw')],
                  _reclassify_risk_op, reclass_habitat_risk_path,
                  _TARGET_PIXEL_INT, _TARGET_NODATA_INT),
            target_path_list=[reclass_habitat_risk_path],
            task_name='reclassify_%s_risk' % habitat,
            dependent_task_list=[calc_risk_task])

    # Calculate ecosystem risk scores. This task depends on every task above,
    # so join the graph first.
    LOGGER.info('Calculating average and reclassified ecosystem risks.')

    # Create input list for calculating average & reclassified ecosystem risks
    ecosystem_risk_raster_path = os.path.join(
        output_dir, 'TOTAL_RISK_Ecosystem%s.tif' % file_suffix)
    reclass_ecosystem_risk_raster_path = os.path.join(
        output_dir, 'RECLASS_RISK_Ecosystem%s.tif' % file_suffix)

    # Append individual habitat risk rasters to the input list
    hab_risk_raster_path_list = info_df.loc[info_df.TYPE == _HABITAT_TYPE][
        'TOT_RISK_RASTER_PATH'].tolist()
    hab_risk_path_band_list = [(habitat_count_raster_path, 1)]
    for hab_risk_raster_path in hab_risk_raster_path_list:
        hab_risk_path_band_list.append((hab_risk_raster_path, 1))

    # Calculate average ecosystem risk
    ecosystem_risk_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(hab_risk_path_band_list, _ecosystem_risk_op,
              ecosystem_risk_raster_path, _TARGET_PIXEL_FLT,
              _TARGET_NODATA_FLT),
        target_path_list=[ecosystem_risk_raster_path],
        task_name='calculate_average_ecosystem_risk',
        dependent_task_list=ecosystem_risk_dependent_tasks)

    # Calculate reclassified ecosystem risk
    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(ecosystem_risk_raster_path, 1), (max_rating, 'raw')],
              _reclassify_ecosystem_risk_op,
              reclass_ecosystem_risk_raster_path,
              _TARGET_PIXEL_INT, _TARGET_NODATA_INT),
        target_path_list=[reclass_ecosystem_risk_raster_path],
        task_name='reclassify_ecosystem_risk',
        dependent_task_list=[ecosystem_risk_task])

    # Calculate the mean criteria scores on the habitat pixels within the
    # polygons in the AOI vector
    LOGGER.info('Calculating zonal statistics.')

    # Join here because zonal_rasters needs to be loaded from the pickle file
    task_graph.join()
    zonal_rasters = pickle.load(open(rasterized_aoi_pickle_path, 'rb'))
    region_list = list(zonal_rasters)

    # Dependent task list used when converting all the calculated stats to CSV
    zonal_stats_dependent_tasks = []

    # Filter habitat rows from the information dataframe
    habitats_info_df = info_df.loc[info_df.TYPE == _HABITAT_TYPE]

    # Calculate and pickle zonal stats to files
    for region_name, zonal_raster_path in zonal_rasters.items():
        # Compute zonal E and C stats on each habitat-stressor pair
        for hab_str_idx, row in overlap_df.iterrows():
            # Get habitat-stressor name without extension
            habitat_stressor = '_'.join(hab_str_idx)
            LOGGER.info('Calculating zonal stats of %s in %s.' %
                        (habitat_stressor, region_name))

            # Compute pairwise E/C zonal stats
            for criteria_type in ['E', 'C']:
                criteria_raster_path = row[criteria_type + '_RASTER_PATH']
                # Append _[region] suffix to the generic pickle file path
                target_pickle_stats_path = row[
                    criteria_type + '_PICKLE_STATS_PATH'].replace(
                        '.pickle', region_name + '.pickle')
                zonal_stats_dependent_tasks.append(task_graph.add_task(
                    func=_calc_and_pickle_zonal_stats,
                    args=(criteria_raster_path, zonal_raster_path,
                          target_pickle_stats_path, file_preprocessing_dir),
                    target_path_list=[target_pickle_stats_path],
                    task_name='calc_%s_%s_stats_in_%s' % (
                        habitat_stressor, criteria_type, region_name)))

            # Compute pairwise risk zonal stats
            pair_risk_raster_path = row['PAIR_RISK_RASTER_PATH']
            target_pickle_stats_path = row[
                'PAIR_RISK_PICKLE_STATS_PATH'].replace(
                    '.pickle', region_name + '.pickle')
            zonal_stats_dependent_tasks.append(task_graph.add_task(
                func=_calc_and_pickle_zonal_stats,
                args=(pair_risk_raster_path, zonal_raster_path,
                      target_pickle_stats_path, file_preprocessing_dir),
                kwargs={'max_rating': max_rating},
                target_path_list=[target_pickle_stats_path],
                task_name='calc_%s_risk_stats_in_%s' % (
                    habitat_stressor, region_name)))

        # Calculate the overall stats of exposure, consequence, and risk for
        # each habitat from all stressors
        for _, row in habitats_info_df.iterrows():
            habitat_name = row['NAME']
            total_risk_raster_path = row['TOT_RISK_RASTER_PATH']
            target_pickle_stats_path = row[
                'TOT_RISK_PICKLE_STATS_PATH'].replace(
                    '.pickle', region_name + '.pickle')

            LOGGER.info('Calculating overall zonal stats of %s in %s.' %
                        (habitat_name, region_name))

            zonal_stats_dependent_tasks.append(task_graph.add_task(
                func=_calc_and_pickle_zonal_stats,
                args=(total_risk_raster_path, zonal_raster_path,
                      target_pickle_stats_path, file_preprocessing_dir),
                kwargs={'max_rating': max_rating},
                target_path_list=[target_pickle_stats_path],
                task_name='calc_%s_risk_stats_in_%s' % (
                    habitat_name, region_name)))

            # Compute pairwise E/C zonal stats
            for criteria_type in ['E', 'C']:
                total_criteria_raster_path = row[
                    'TOT_' + criteria_type + '_RASTER_PATH']
                target_pickle_stats_path = row[
                    'TOT_' + criteria_type + '_PICKLE_STATS_PATH'].replace(
                        '.pickle', region_name + '.pickle')

                zonal_stats_dependent_tasks.append(task_graph.add_task(
                    func=_calc_and_pickle_zonal_stats,
                    args=(total_criteria_raster_path, zonal_raster_path,
                          target_pickle_stats_path, file_preprocessing_dir),
                    target_path_list=[target_pickle_stats_path],
                    task_name='calc_%s_%s_stats_in_%s' % (
                        habitat_name, criteria_type, region_name)))

    # Convert the statistics dataframe to a CSV file
    target_stats_csv_path = os.path.join(
        output_dir, 'SUMMARY_STATISTICS%s.csv' % file_suffix)

    task_graph.add_task(
        func=_zonal_stats_to_csv,
        args=(
            overlap_df, habitats_info_df, region_list, target_stats_csv_path),
        target_path_list=[target_stats_csv_path],
        task_name='zonal_stats_to_csv',
        dependent_task_list=zonal_stats_dependent_tasks)

    # Finish the model if no visualization outputs need to be generated
    if not args['visualize_outputs']:
        task_graph.close()
        task_graph.join()
        LOGGER.info('HRA model completed.')
        return
    else:
        LOGGER.info('Generating visualization outputs.')

    # Unproject output rasters to WGS84 (World Mercator), and then convert
    # the rasters to GeoJSON files for visualization
    LOGGER.info('Unprojecting output rasters')
    out_risk_raster_paths = info_df[
        info_df.TYPE == _HABITAT_TYPE].RECLASS_RISK_RASTER_PATH.tolist()
    out_stressor_raster_paths = info_df[
        info_df.TYPE == _STRESSOR_TYPE].ALIGN_RASTER_PATH.tolist()
    out_raster_paths = out_risk_raster_paths + out_stressor_raster_paths + [
        reclass_ecosystem_risk_raster_path]

    # Convert the rasters to GeoJSON files in WGS84 for web visualization,
    # since only this format would be recognized by leaflet
    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(_WGS84_ESPG_CODE)
    wgs84_wkt = wgs84_sr.ExportToWkt()
    for out_raster_path in out_raster_paths:
        # Get raster basename without file extension and remove prefix
        prefix = 'aligned_'
        file_basename = os.path.splitext(os.path.basename(out_raster_path))[0]
        if file_basename.startswith(prefix):
            file_basename = file_basename[len(prefix):]

        # Make a GeoJSON from the unprojected raster with an appropriate field
        # name
        if file_basename.startswith('RECLASS_RISK_'):
            field_name = 'Risk Score'
        else:
            # Append 'STRESSOR_' prefix if it's not a risk layer
            file_basename = 'STRESSOR_' + file_basename
            field_name = 'Stressor'

        geojson_path = os.path.join(viz_dir, file_basename + '.geojson')
        task_graph.add_task(
            func=_raster_to_geojson,
            args=(out_raster_path, geojson_path, file_basename, field_name),
            kwargs={'target_sr_wkt': wgs84_wkt},
            target_path_list=[geojson_path],
            task_name='create_%s_geojson' % file_basename)

    task_graph.close()
    task_graph.join()

    # Copy summary stats CSV to the viz output folder for scatter plots
    # visualization. This keeps all the viz files in one place
    viz_stats_csv_path = os.path.join(
        viz_dir, 'SUMMARY_STATISTICS%s.csv' % file_suffix)
    shutil.copyfile(target_stats_csv_path, viz_stats_csv_path)

    LOGGER.info(
        'HRA model completed. Please visit http://marineapps.'
        'naturalcapitalproject.org/ to visualize your outputs.')


def _create_raster_and_rasterize_vector(
        simplified_vector_path, target_raster_path, target_pixel_size,
        rasterize_pixel_type, rasterize_nodata, fill_value, rasterize_kwargs):
    """Wrap these related operations so they can be captured in one task."""
    pygeoprocessing.create_raster_from_vector_extents(
        simplified_vector_path, target_raster_path,
        target_pixel_size, rasterize_pixel_type, rasterize_nodata,
        fill_value=fill_value)
    pygeoprocessing.rasterize(
        simplified_vector_path, target_raster_path,
        burn_values=rasterize_kwargs['burn_values'],
        option_list=rasterize_kwargs['option_list'])


def _raster_to_geojson(
        base_raster_path, target_geojson_path, layer_name, field_name,
        target_sr_wkt=None):
    """Convert a raster to a GeoJSON file with layer and field name.

    Typically the base raster will be projected and the target GeoJSON should
    end up with geographic coordinates (EPSG: 4326 per the GeoJSON spec).
    So a GPKG serves as intermediate storage for the polygonized projected
    features.

    Args:
        base_raster_path (str): the raster that needs to be turned into a
            GeoJSON file.
        target_geojson_path (str): the desired path for the new GeoJSON.
        layer_name (str): the name of the layer going into the new shapefile.
        field_name (str): the name of the field to write raster values in.
        target_sr_wkt (str): the target projection for vector in Well Known
            Text (WKT) form.

    Returns:
        None.

    """
    raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    mask = band.GetMaskBand()

    # Use raster SRS for the temp GPKG
    base_sr = osr.SpatialReference()
    base_sr_wkt = raster.GetProjectionRef()
    base_sr.ImportFromWkt(base_sr_wkt)

    # Polygonize onto a GPKG
    gpkg_driver = gdal.GetDriverByName('GPKG')
    temp_gpkg_path = os.path.splitext(target_geojson_path)[0] + '.gpkg'
    vector = gpkg_driver.Create(temp_gpkg_path, 0, 0, 0, gdal.GDT_Unknown)

    vector.StartTransaction()
    vector_layer = vector.CreateLayer(layer_name, base_sr, ogr.wkbPolygon)

    # Create an integer field that contains values from the raster
    field_defn = ogr.FieldDefn(str(field_name), ogr.OFTInteger)
    field_defn.SetWidth(3)
    field_defn.SetPrecision(0)
    vector_layer.CreateField(field_defn)

    gdal.Polygonize(band, mask, vector_layer, 0)
    vector_layer.SyncToDisk()
    vector.CommitTransaction()
    vector_layer = None
    vector = None
    band = None
    raster = None

    # Convert GPKG to GeoJSON, reprojecting if necessary
    if target_sr_wkt and base_sr_wkt != target_sr_wkt:
        pygeoprocessing.reproject_vector(
            temp_gpkg_path, target_sr_wkt, target_geojson_path,
            driver_name='GeoJSON')
    else:
        geojson_driver = gdal.GetDriverByName('GeoJSON')
        geojson_driver.CreateCopy(target_geojson_path, temp_gpkg_path)

    os.remove(temp_gpkg_path)


def _calc_and_pickle_zonal_stats(
        score_raster_path, zonal_raster_path, target_pickle_stats_path,
        working_dir, max_rating=None):
    """Calculate zonal stats on a score raster where zonal raster is 1.

    Clip the score raster with the bounding box of zonal raster first, so
    we only look at blocks that intersect.

    Args:
        score_raster_path (str): a path to the E/C/risk score raster to be
            analyzed.
        zonal_raster_path (str): a path to the zonal raster with 1s
            representing the regional extent, used for getting statistics
            from score raster in that region.
        target_pickle_stats_path (str): a path to the pickle file for storing
            zonal statistics, including count, sum, min, max, and mean.
        working_dir (str): a path to the working folder for saving clipped
            score raster file.
        max_rating (float): if exists, it's used for classifying risks into
            three categories and calculating percentage area of high/medium/low
            scores.

    Returns:
        None

    """
    # Create a stats dictionary for saving zonal statistics, including
    # mean, min, and max.
    stats_dict = {}
    stats_dict['MIN'] = float('inf')
    stats_dict['MAX'] = float('-inf')
    for stats_type in ['MEAN', '%HIGH', '%MEDIUM', '%LOW']:
        stats_dict[stats_type] = 0.

    # Clip score raster to the extent of zonal raster. The file will be deleted
    # at the end.
    with tempfile.NamedTemporaryFile(
            prefix='clipped_', suffix='.tif', delete=False,
            dir=working_dir) as clipped_raster_file:
        clipped_score_raster_path = clipped_raster_file.name

    zonal_raster_info = pygeoprocessing.get_raster_info(zonal_raster_path)
    target_pixel_size = zonal_raster_info['pixel_size']
    target_bounding_box = zonal_raster_info['bounding_box']
    target_sr_wkt = zonal_raster_info['projection_wkt']
    pygeoprocessing.warp_raster(
        score_raster_path, target_pixel_size, clipped_score_raster_path,
        _RESAMPLE_METHOD, target_bb=target_bounding_box,
        target_projection_wkt=target_sr_wkt)

    # Return a dictionary with values of 0, if the two input rasters do not
    # intersect at all.
    score_raster = gdal.OpenEx(clipped_score_raster_path, gdal.OF_RASTER)
    try:
        score_band = score_raster.GetRasterBand(1)
    except ValueError as e:
        if 'Bounding boxes do not intersect' in repr(e):
            LOGGER.info('Bounding boxes of %s and %s do not intersect.' %
                        (score_raster_path, zonal_raster_path))
        for stats_type in stats_dict:
            stats_dict[stats_type] = None  # This will leave blank in CSV table
        score_raster = None
        pickle.dump(stats_dict, open(target_pickle_stats_path, 'wb'))
        os.remove(clipped_score_raster_path)
        return

    score_nodata = score_band.GetNoDataValue()
    zonal_raster = gdal.OpenEx(zonal_raster_path, gdal.OF_RASTER)
    zonal_band = zonal_raster.GetRasterBand(1)
    pixel_count = 0.
    pixel_sum = 0.

    if max_rating:
        high_score_count = 0.
        med_score_count = 0.
        low_score_count = 0.

    # Iterate through each data block and calculate stats.
    for score_offsets in pygeoprocessing.iterblocks(
            (clipped_score_raster_path, 1), offset_only=True):
        score_block = score_band.ReadAsArray(**score_offsets)
        zonal_block = zonal_band.ReadAsArray(**score_offsets)

        valid_mask = (
            ~utils.array_equals_nodata(score_block, score_nodata) &
            (zonal_block == 1))
        valid_score_block = score_block[valid_mask]
        if valid_score_block.size == 0:
            continue

        # Calculate min and max values, and sum and count of valid pixels.
        pixel_count += valid_score_block.size
        pixel_sum += numpy.sum(valid_score_block)
        stats_dict['MIN'] = min(
            stats_dict['MIN'], numpy.amin(valid_score_block))
        stats_dict['MAX'] = max(
            stats_dict['MAX'], numpy.amax(valid_score_block))

        # Calculate percentage of high, medium, and low rating areas.
        if max_rating:
            high_score_count += valid_score_block[
                (valid_score_block > max_rating/3*2)].size
            med_score_count += valid_score_block[
                (valid_score_block <= max_rating/3*2) &
                (valid_score_block > max_rating/3)].size
            low_score_count += valid_score_block[
                (valid_score_block <= max_rating/3)].size

    if pixel_count > 0:
        stats_dict['MEAN'] = pixel_sum / pixel_count
        if max_rating:
            stats_dict['%HIGH'] = high_score_count/pixel_count*100.
            stats_dict['%MEDIUM'] = med_score_count/pixel_count*100.
            stats_dict['%LOW'] = low_score_count/pixel_count*100.
    else:
        for stats_type in stats_dict:
            stats_dict[stats_type] = None  # This will leave blank in CSV table

    score_raster = None
    zonal_raster = None
    zonal_band = None
    score_band = None

    pickle.dump(stats_dict, open(target_pickle_stats_path, 'wb'))
    os.remove(clipped_score_raster_path)


def _zonal_stats_to_csv(
        overlap_df, info_df, region_list, target_stats_csv_path):
    """Unpickle zonal stats from files and concatenate the dataframe into CSV.

    Args:
        overlap_df (dataframe): a multi-index dataframe with exposure and
            consequence raster paths, as well as pickle path columns for
            getting zonal statistics dictionary from.
        habitat_info_df (dataframe): a dataframe with information on total
            exposure, consequence, and risk raster/pickle file paths for each
            habitat.
        region_list (list): a list of subregion names used as column values of
            the ``SUBREGION`` column in the zonal stats dataframe.
        target_stats_csv_path (str): path to the CSV file for saving the final
            merged zonal stats dataframe.

    Returns:
        None

    """
    # Create a stats dataframe with habitat and stressor index from overlap
    # dataframe
    crit_stats_cols = ['MEAN', 'MIN', 'MAX']
    risk_stats_cols = crit_stats_cols + ['%HIGH', '%MEDIUM', '%LOW']
    len_crit_cols = len(crit_stats_cols)
    len_risk_cols = len(risk_stats_cols)
    columns = map(
        str.__add__,
        ['E_']*len_crit_cols + ['C_']*len_crit_cols + ['R_']*len_risk_cols,
        crit_stats_cols*2 + risk_stats_cols)

    stats_df = pandas.DataFrame(index=overlap_df.index, columns=list(columns))

    # Add a ``SUBREGION`` column to the dataframe and update it with the
    # corresponding stats in each subregion
    region_df_list = []
    for region in region_list:
        region_df = stats_df.copy()
        # Insert the new column in the beginning
        region_df.insert(loc=0, column='SUBREGION', value=region)

        for hab_str_idx, row in overlap_df.iterrows():
            # Unpack pairwise criteria stats
            for criteria_type in ['E', 'C']:
                crit_stats_dict = pickle.load(
                    open(row[criteria_type + '_PICKLE_STATS_PATH'].replace(
                        '.pickle', region + '.pickle'), 'rb'))
                for stats_type in crit_stats_cols:
                    header = criteria_type + '_' + stats_type
                    region_df.loc[hab_str_idx, header] = crit_stats_dict[
                        stats_type]

            # Unpack pairwise risk stats
            risk_stats_dict = pickle.load(
                open(row['PAIR_RISK_PICKLE_STATS_PATH'].replace(
                    '.pickle', region + '.pickle'), 'rb'))
            for stats_type in risk_stats_cols:
                header = 'R_' + stats_type
                region_df.loc[hab_str_idx, header] = risk_stats_dict[
                    stats_type]

        for _, row in info_df.iterrows():
            habitat_name = row['NAME']

            # An index used as values for HABITAT and STRESSOR columns
            hab_only_idx = (habitat_name, '(FROM ALL STRESSORS)')
            region_df.loc[hab_only_idx, 'SUBREGION'] = region

            # Unpack total criteria stats
            for criteria_type in ['E', 'C']:
                crit_stats_dict = pickle.load(
                    open(row[
                        'TOT_' + criteria_type + '_PICKLE_STATS_PATH'].replace(
                        '.pickle', region + '.pickle'), 'rb'))
                for stats_type in crit_stats_cols:
                    header = criteria_type + '_' + stats_type
                    region_df.loc[hab_only_idx, header] = crit_stats_dict[
                        stats_type]

            # Unpack total risk stats
            risk_stats_dict = pickle.load(
                open(row['TOT_RISK_PICKLE_STATS_PATH'].replace(
                    '.pickle', region + '.pickle'), 'rb'))
            for stats_type in risk_stats_cols:
                header = 'R_' + stats_type
                region_df.loc[hab_only_idx, header] = risk_stats_dict[
                    stats_type]

        region_df_list.append(region_df)

    # Merge all the subregion dataframes
    final_stats_df = pandas.concat(region_df_list)

    # Sort habitat and stressor by their names in ascending order
    final_stats_df.sort_values(
        [_HABITAT_HEADER, _STRESSOR_HEADER], inplace=True)

    final_stats_df.to_csv(target_stats_csv_path)


def _create_rasters_from_geometries(
        geom_pickle_path, working_dir, target_pickle_path, target_pixel_size):
    """Create a blank integer raster from a list of geometries.

    Pixel value of 1 on the target rasters indicates the existence of the
    geometry collection, and everywhere else is nodata.

    Args:
        geom_pickle_path (str): a path to a tuple of pickled
            geom_sets_by_field, a list of shapely geometry objects in Well
            Known Binary (WKB) format, and target spatial reference in Well
            Known Text (WKT) format.
        working_dir (str): a path indicating where raster files should be
            created.
        target_pickle_path (str): path to location of generated geotiff;
            the upper left hand corner of this raster will be aligned with the
            bounding box the extent will be exactly equal or contained the
            bounding box depending on whether the pixel size divides evenly
            into the bounding box; if not coordinates will be rounded up to
            contain the original extent.
        target_pixel_size (list/tuple): the x/y pixel size as a sequence,
            ex: [30.0, -30.0].

    Returns:
        None

    """
    # Get the geometry collections and desired projection for target rasters
    geom_sets_by_field, target_sr_wkt = pickle.load(
        open(geom_pickle_path, 'rb'))
    raster_paths_by_field = {}

    for field_value, shapely_geoms_wkb in geom_sets_by_field.items():
        # Create file basename based on field value
        if not isinstance(field_value, str):
            field_value = str(field_value)

        field_value = field_value
        file_basename = 'rasterized_' + field_value
        target_raster_path = os.path.join(working_dir, file_basename + '.tif')

        # Add the field value and file path pair to dictionary
        raster_paths_by_field[field_value] = target_raster_path

        # Create raster from bounding box of the merged geometry
        LOGGER.info('Rasterizing geometries of field value %s.' % field_value)

        # Get the union of the geometries in the list
        union_geom = shapely.ops.unary_union(shapely_geoms_wkb)
        geom = ogr.CreateGeometryFromWkb(union_geom.wkb)
        bounding_box = geom.GetEnvelope()

        # Round up on the rows and cols so that the target raster is larger
        # than or equal to the bounding box
        n_cols = int(numpy.ceil(
            abs((bounding_box[1] - bounding_box[0]) / target_pixel_size[0])))
        n_rows = int(numpy.ceil(
            abs((bounding_box[3] - bounding_box[2]) / target_pixel_size[1])))

        raster_driver = gdal.GetDriverByName('GTiff')
        n_bands = 1
        target_raster = raster_driver.Create(
            target_raster_path, n_cols, n_rows, n_bands, _TARGET_PIXEL_INT,
            options=_DEFAULT_GTIFF_CREATION_OPTIONS)

        # Initialize everything to nodata
        target_raster.GetRasterBand(1).SetNoDataValue(_TARGET_NODATA_INT)

        # Set the transform based on the upper left corner and given pixel
        # dimensions
        if target_pixel_size[0] < 0:
            x_source = bounding_box[1]
        else:
            x_source = bounding_box[0]
        if target_pixel_size[1] < 0:
            y_source = bounding_box[3]
        else:
            y_source = bounding_box[2]
        raster_transform = [
            x_source, target_pixel_size[0], 0.0,
            y_source, 0.0, target_pixel_size[1]]
        target_raster.SetGeoTransform(raster_transform)

        # Set target projection
        target_raster.SetProjection(target_sr_wkt)

        # Make a temporary vector so we can create layer and add the geometry
        # to it
        vector_driver = gdal.GetDriverByName('MEMORY')
        temp_vector = vector_driver.Create('temp', 0, 0, 0, gdal.GDT_Unknown)
        target_spat_ref = osr.SpatialReference()
        target_spat_ref.ImportFromWkt(target_sr_wkt)
        temp_layer = temp_vector.CreateLayer(
            'temp', target_spat_ref, ogr.wkbPolygon)
        layer_defn = temp_layer.GetLayerDefn()

        # Add geometries to the layer
        temp_layer.StartTransaction()
        temp_feat = temp_layer.GetNextFeature()
        temp_feat = ogr.Feature(layer_defn)
        temp_feat.SetGeometry(geom)
        temp_layer.CreateFeature(temp_feat)
        temp_layer.CommitTransaction()

        # Burn the geometry onto the raster with values of 1
        gdal.RasterizeLayer(target_raster, [1], temp_layer, burn_values=[1])
        target_raster.FlushCache()
        target_raster = None

        # Delete the layer we just added to the vector
        temp_layer = None
        temp_vector.DeleteLayer(0)
        temp_vector = None

    pickle.dump(raster_paths_by_field, open(target_pickle_path, 'wb'))


def _get_vector_geometries_by_field(
        base_vector_path, field_name, target_geom_pickle_path):
    """Get a dictionary of field values with list of geometries from a vector.

    Args:
        base_vector_path (str): a path to the vector the get geometry
            collections based on the given field name.
        field_name (str): the field name used to aggregate the geometries
            of features in the base vector.
        target_geom_pickle_path (str): a target path to a tuple of pickled
            geom_sets_by_field and target spatial reference.
            geom_sets_by_field is a list of shapely geometry objects in Well
            Known Binary (WKB) format.

    Returns:
        None

    Raises:
        ValueError if a value on field name of the base vector is None.

    """
    LOGGER.info('Collecting geometries on field %s.' % field_name)
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()
    spat_ref = base_layer.GetSpatialRef()

    geom_sets_by_field = {}
    for feat in base_layer:
        field_value = feat.GetField(field_name)
        geom = feat.GetGeometryRef()
        geom_wkb = shapely.wkb.loads(bytes(geom.ExportToWkb()))
        if field_value is None:
            base_vector = None
            base_layer = None
            raise ValueError('Field value in field "%s" in the AOI vector is '
                             'None.' % field_name)
        # Append buffered geometry to prevent invalid geometries
        elif field_value in geom_sets_by_field:
            geom_sets_by_field[field_value].append(geom_wkb.buffer(0))
        else:
            geom_sets_by_field[field_value] = [geom_wkb.buffer(0)]

    base_vector = None
    base_layer = None

    pickle.dump(
        (geom_sets_by_field, spat_ref.ExportToWkt()),
        open(target_geom_pickle_path, 'wb'))


def _has_field_name(base_vector_path, field_name):
    """Check if the vector attribute table has the designated field name.

    Args:
        base_vector_path (str): a path to the vector to check the field name
            with.
        field_name (str): the field name to be inspected.

    Returns:
        True if the field name exists, False if it doesn't.

    """
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()

    fields = [field.GetName().lower() for field in base_layer.schema]
    base_vector = None
    base_layer = None
    if field_name not in fields:
        LOGGER.info('The %s field is not provided in the vector.' % field_name)
        return False
    else:
        return True


def _ecosystem_risk_op(habitat_count_arr, *hab_risk_arrays):
    """Calculate average habitat risk scores from hab_risk_arrays.

    Divide the total risk by the number of habitats on each pixel.

    Args:
        habitat_count_arr (array): an integer array with each pixel indicating
            the number of habitats existing on that pixel.
        *hab_risk_arrays: a list of arrays representing reclassified risk
            scores for each habitat.

    Returns:
        ecosystem_risk_arr (array): an average risk float array calculated by
            dividing the cumulative habitat risks by the habitat count in
            that pixel.

    """
    ecosystem_risk_arr = numpy.full(
        habitat_count_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    ecosystem_mask = (habitat_count_arr > 0) & ~utils.array_equals_nodata(
        habitat_count_arr, _TARGET_NODATA_INT)
    ecosystem_risk_arr[ecosystem_mask] = 0

    # Add up all the risks of each habitat
    for hab_risk_arr in hab_risk_arrays:
        valid_risk_mask = ~utils.array_equals_nodata(
            hab_risk_arr, _TARGET_NODATA_FLT)
        ecosystem_risk_arr[valid_risk_mask] += hab_risk_arr[valid_risk_mask]

    # Divide risk score by the number of habitats in each pixel. This way we
    # could normalize the risk and not be biased by any large risk score
    # resulting from the existence of multiple habitats
    ecosystem_risk_arr[ecosystem_mask] /= habitat_count_arr[ecosystem_mask]

    return ecosystem_risk_arr


def _reclassify_ecosystem_risk_op(ecosystem_risk_arr, max_rating):
    """Reclassify the ecosystem risk into three categories.

    If 0 < 3*(risk/max rating) <= 1, classify the risk score to 1.
    If 1 < 3*(risk/max rating) <= 2, classify the risk score to 2.
    If 2 < 3*(risk/max rating) <= 3 , classify the risk score to 3.
    Note: If 3*(risk/max rating) == 0, it will remain 0, meaning that there's
    no stressor on the ecosystem.

    Args:
        ecosystem_risk_arr (array): an average risk score calculated by
            dividing the cumulative habitat risks by the habitat count in
            that pixel.
        max_rating (float): the maximum possible risk score used for
            reclassifying the risk score into discrete categories.

    Returns:
        reclass_ecosystem_risk_arr (array): a reclassified ecosystem risk
            integer array.

    """
    reclass_ecosystem_risk_arr = numpy.full(
        ecosystem_risk_arr.shape, _TARGET_NODATA_INT, dtype=numpy.int8)
    valid_pixel_mask = ~utils.array_equals_nodata(
        ecosystem_risk_arr, _TARGET_NODATA_FLT)

    # Divide risk score by (maximum possible risk score/3) to get an integer
    # ranging from 0 to 3, then return the ceiling of it
    reclass_ecosystem_risk_arr[valid_pixel_mask] = numpy.ceil(
        ecosystem_risk_arr[valid_pixel_mask] / (max_rating/3.)).astype(
            numpy.int8)

    return reclass_ecosystem_risk_arr


def _count_habitats_op(*habitat_arrays):
    """Adding pixel values together from multiple arrays.

    Args:
        *habitat_arrays: a list of arrays with 1s and 0s values.

    Returns:
        habitat_count_arr (array): an integer array with each pixel indicating
            the summation value of input habitat arrays.

    """
    # Since the habitat arrays have been aligned, we can just use the shape
    # of the first habitat array
    habitat_count_arr = numpy.full(
        habitat_arrays[0].shape, 0, dtype=numpy.int8)

    for habitat_arr in habitat_arrays:
        habiat_mask = ~utils.array_equals_nodata(
            habitat_arr, _TARGET_NODATA_INT)
        habitat_count_arr[habiat_mask] += habitat_arr.astype(
            numpy.int8)[habiat_mask]

    return habitat_count_arr


def _reclassify_risk_op(risk_arr, max_rating):
    """Reclassify total risk score on each pixel into 0 to 3, discretely.

    Divide total risk score by (max_risk_score/3) to get a continuous risk
    score of 0 to 3, then use numpy.ceil to get discrete score.

    If 0 < risk <= 1, classify the risk score to 1.
    If 1 < risk <= 2, classify the risk score to 2.
    If 2 < risk <= 3 , classify the risk score to 3.
    Note: If risk == 0, it will remain 0, meaning that there's no
    stressor on that habitat.

    Args:
        risk_arr (array): an array of cumulative risk scores from all stressors
        max_rating (float): the maximum possible risk score used for
            reclassifying the risk score into 0 to 3 on each pixel.

    Returns:
        reclass_arr (array): an integer array of reclassified risk scores for a
            certain habitat. The values are discrete on the array.

    """
    reclass_arr = numpy.full(
        risk_arr.shape, _TARGET_NODATA_INT, dtype=numpy.int8)
    valid_pixel_mask = ~utils.array_equals_nodata(risk_arr, _TARGET_NODATA_FLT)

    # Return the ceiling of the continuous risk score
    reclass_arr[valid_pixel_mask] = numpy.ceil(
        risk_arr[valid_pixel_mask] / (max_rating/3.)).astype(numpy.int8)

    return reclass_arr


def _tot_risk_op(habitat_arr, *pair_risk_arrays):
    """Calculate the cumulative risks to a habitat from all stressors.

    The risk score is calculated by summing up all the risk scores on each
    valid pixel of the habitat.

    Args:
        habitat_arr (array): an integer habitat array where 1's indicates
            habitat existence and 0's non-existence.
        *pair_risk_arrays: a list of individual risk float arrays from each
            stressor to a certain habitat.

    Returns:
        tot_risk_arr (array): a cumulative risk float array calculated by
            summing up all the individual risk arrays.

    """
    # Fill 0s to the total risk array on where habitat exists
    habitat_mask = (habitat_arr == 1)
    tot_risk_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    tot_risk_arr[habitat_mask] = 0

    for pair_risk_arr in pair_risk_arrays:
        valid_pixel_mask = ~utils.array_equals_nodata(
            pair_risk_arr, _TARGET_NODATA_FLT)
        tot_risk_arr[valid_pixel_mask] += pair_risk_arr[valid_pixel_mask]

    # Rescale total risk to 0 to max_rating
    final_valid_mask = ~utils.array_equals_nodata(
        tot_risk_arr, _TARGET_NODATA_FLT)
    tot_risk_arr[final_valid_mask] /= len(pair_risk_arrays)

    return tot_risk_arr


def _pair_risk_op(exposure_arr, consequence_arr, max_rating, risk_eq):
    """Calculate habitat-stressor risk array based on the risk equation.

    Euclidean risk equation: R = sqrt((E-1)^2 + (C-1)^2)
    Multiplicative risk equation: R = E * C

    Args:
        exosure_arr (array): a float array with total exposure scores.
        consequence_arr (array): a float array with total consequence scores.
        max_rating (float): a number representing the highest potential value
            that should be represented in rating in the criteria table,
            used for calculating the maximum possible risk score.
        risk_eq (str): a string identifying the equation that should be
            used in calculating risk scores. It could be either 'Euclidean' or
            'Multiplicative'.

    Returns:
        risk_arr (array): a risk float array calculated based on the risk
            equation.

    """
    # Calculate the maximum possible risk score
    if risk_eq == 'Multiplicative':
        # The maximum risk from a single stressor is max_rating*max_rating
        max_risk_score = max_rating*max_rating
    else:  # risk_eq is 'Euclidean'
        # The maximum risk score for a habitat from a single stressor is
        # sqrt( (max_rating-1)^2 + (max_rating-1)^2 )
        max_risk_score = numpy.sqrt(numpy.power((max_rating-1), 2)*2)

    risk_arr = numpy.full(
        exposure_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    zero_pixel_mask = (exposure_arr == 0) | (consequence_arr == 0)
    valid_pixel_mask = (
        ~utils.array_equals_nodata(exposure_arr, _TARGET_NODATA_FLT) &
        ~utils.array_equals_nodata(consequence_arr, _TARGET_NODATA_FLT))
    nonzero_valid_pixel_mask = ~zero_pixel_mask & valid_pixel_mask

    # Zero pixels are where non of the stressor exists in the habitat
    risk_arr[zero_pixel_mask] = 0

    if risk_eq == 'Euclidean':
        # If E-1 or C-1 is less than 0, replace the pixel value with 0
        risk_arr[nonzero_valid_pixel_mask] = numpy.sqrt(
            numpy.power(
                numpy.maximum(
                    exposure_arr[nonzero_valid_pixel_mask]-1, 0), 2) +
            numpy.power(
                numpy.maximum(
                    consequence_arr[nonzero_valid_pixel_mask]-1, 0), 2))

    else:  # Multiplicative
        risk_arr[nonzero_valid_pixel_mask] = numpy.multiply(
            exposure_arr[nonzero_valid_pixel_mask],
            consequence_arr[nonzero_valid_pixel_mask])

    risk_arr[nonzero_valid_pixel_mask] *= (max_rating/max_risk_score)

    return risk_arr


def _total_exposure_op(habitat_arr, *num_denom_list):
    """Calculate the exposure score for a habitat layer from all stressors.

    Add up all the numerators and denominators respectively, then divide
    the total numerator by the total denominator on habitat pixels, to get
    the final exposure or consequence score.

    Args:
        habitat_arr (array): a habitat integer array where 1's indicates
            habitat existence and 0's non-existence.
        *num_denom_list (list): if exists, it's a list of numerator float
            arrays in the first half of the list, and denominator scores
            (float) in the second half. Must always be even-number
            of elements.

    Returns:
        tot_expo_arr (array): an exposure float array calculated by dividing
            the total numerator by the total denominator. Pixel values are
            nodata outside of habitat, and will be 0 if there is no valid
            numerator value on that pixel.

    """
    habitat_mask = (habitat_arr == 1)

    # Fill each array with value of 0 on the habitat pixels, assuming that
    # the risk score on that habitat is 0 before adding numerator/denominator
    tot_num_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    tot_num_arr[habitat_mask] = 0

    tot_expo_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    tot_expo_arr[habitat_mask] = 0

    tot_denom = 0

    # Numerator arrays are in the first half of the list
    num_arr_list = num_denom_list[:len(num_denom_list)//2]
    denom_list = num_denom_list[len(num_denom_list)//2:]

    # Calculate the cumulative numerator and denominator values
    for num_arr in num_arr_list:
        valid_num_mask = ~utils.array_equals_nodata(
            num_arr, _TARGET_NODATA_FLT)
        tot_num_arr[valid_num_mask] += num_arr[valid_num_mask]

    for denom in denom_list:
        tot_denom += denom

    # If the numerator is nodata, do not divide the arrays
    final_valid_mask = ~utils.array_equals_nodata(
        tot_num_arr, _TARGET_NODATA_FLT)

    tot_expo_arr[final_valid_mask] = tot_num_arr[
        final_valid_mask] / tot_denom

    return tot_expo_arr


def _total_consequence_op(
        habitat_arr, recov_num_arr, recov_denom, *num_denom_list):
    """Calculate the consequence score for a habitat layer from all stressors.

    Add up all the numerators and denominators (including ones from recovery)
    respectively, then divide the total numerator by the total denominator on
    habitat pixels, to get the final consequence score.

    Args:
        habitat_arr (array): a habitat integer array where 1's indicates
            habitat existence and 0's non-existence.
        recov_num_arr (array): a float array of the numerator score from
            recovery potential, to be added to the consequence numerator scores
        recov_denom (float): the precalculated cumulative recovery denominator
            score.
        *num_denom_list (list): if exists, it's a list of numerator float
            arrays in the first half of the list, and denominator scores
            (float) in the second half. Must always be even-number
            of elements.

    Returns:
        tot_conseq_arr (array): a consequence float array calculated by
            dividing the total numerator by the total denominator. Pixel values
            are nodata outside of habitat, and will be 0 if there is no valid
            numerator value on that pixel.

    """
    habitat_mask = (habitat_arr == 1)

    tot_num_arr = numpy.copy(recov_num_arr)

    # Fill each array with value of 0 on the habitat pixels, assuming that
    # criteria score is 0 before adding numerator/denominator
    tot_conseq_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    tot_conseq_arr[habitat_mask] = 0

    tot_denom = recov_denom

    # Numerator arrays are in the first half of the list
    num_arr_list = num_denom_list[:len(num_denom_list)//2]
    denom_list = num_denom_list[len(num_denom_list)//2:]

    # Calculate the cumulative numerator and denominator values
    for num_arr in num_arr_list:
        valid_num_mask = ~utils.array_equals_nodata(
            num_arr, _TARGET_NODATA_FLT)
        tot_num_arr[valid_num_mask] += num_arr[valid_num_mask]

    for denom in denom_list:
        tot_denom += denom

    # If the numerator is nodata, do not divide the arrays
    final_valid_mask = ~utils.array_equals_nodata(
        tot_num_arr, _TARGET_NODATA_FLT)

    tot_conseq_arr[final_valid_mask] = tot_num_arr[
        final_valid_mask] / tot_denom

    return tot_conseq_arr


def _pair_exposure_op(
        habitat_arr, stressor_dist_arr, stressor_buffer, num_arr, denom):
    """Calculate individual E/C scores by dividing num by denom arrays.

    The equation for calculating the score is numerator/denominator. This
    function will only calculate the score on pixels where both habitat
    and stressor (including buffer zone) exist.

    Args:
        habitat_arr (array): a habitat integer array where 1's indicates
            habitat existence and 0's non-existence.
        stressor_dist_arr (array): a stressor distance float array where pixel
            values represent the distance of that pixel to a stressor
            pixel.
        stressor_buffor (float): a number representing how far down the
            influence is from the stressor pixel.
        num_arr (array): a float array of the numerator scores calculated based
            on the E/C equation.
        denom (float): a cumulative value pre-calculated based on the criteria
            table. It will be used to divide the numerator.

    Returns:
        exposure_arr (array): a float array of the scores calculated based on
            the E/C equation in users guide.

    """
    habitat_mask = (habitat_arr == 1)
    stressor_mask = (stressor_dist_arr == 0)
    # Habitat-stressor overlap mask that excludes stressor buffer
    hab_stress_overlap_mask = (habitat_mask & stressor_mask)

    # Mask stressor buffer zone
    stressor_buff_mask = (
        (stressor_dist_arr > 0) & (stressor_dist_arr < stressor_buffer))
    hab_buff_overlap_mask = (habitat_mask & stressor_buff_mask)

    # Denominator would always be unaffected by ratings in the area where
    # habitat and stressor + stressor buffer overlap
    hab_stress_buff_mask = (hab_stress_overlap_mask |
                            hab_buff_overlap_mask)

    # Initialize output exposure or consequence score array
    exposure_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    exposure_arr[habitat_mask] = 0

    exposure_arr[hab_stress_buff_mask] = num_arr[hab_stress_buff_mask] / denom

    return exposure_arr


def _pair_consequence_op(
        habitat_arr, stressor_dist_arr, stressor_buffer, conseq_num_arr,
        conseq_denom, recov_num_arr, recov_denom):
    """Calculate individual E/C scores by dividing num by denom arrays.

    The equation for calculating the score is numerator/denominator. This
    function will only calculate the score on pixels where both habitat
    and stressor (including buffer zone) exist.

    Args:
        habitat_arr (array): a habitat integer array where 1's indicates
            habitat existence and 0's non-existence.
        stressor_dist_arr (array): a stressor distance float array where pixel
            values represent the distance of that pixel to a stressor
            pixel.
        stressor_buffor (float): a number representing how far down the
            influence is from the stressor pixel.
        conseq_num_arr (array): a float array of the numerator scores
            calculated by rating/(dq*weight).
        conseq_denom (float): a cumulative value pre-calculated based on the
            criteria table. It will be used to divide the numerator.
        recov_num_arr (array): a float array of the recovery numerator scores
            calculated based on habitat resilience attribute.
        recov_denom (float): the precalculated cumulative recovery denominator
            score.

    Returns:
        consequence_arr (array): a float array of the scores calculated based
            on the E/C equation in users guide.

    """
    habitat_mask = (habitat_arr == 1)
    stressor_mask = (stressor_dist_arr == 0)
    # Habitat-stressor overlap mask that excludes stressor buffer
    hab_stress_overlap_mask = (habitat_mask & stressor_mask)

    # Mask stressor buffer zone
    stressor_buff_mask = (
        (stressor_dist_arr > 0) & (stressor_dist_arr < stressor_buffer))
    hab_buff_overlap_mask = (habitat_mask & stressor_buff_mask)

    # Denominator would always be unaffected by ratings in the area where
    # habitat and stressor + stressor buffer overlap
    hab_stress_buff_mask = (hab_stress_overlap_mask |
                            hab_buff_overlap_mask)

    # Initialize output exposure or consequence score array
    consequence_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    consequence_arr[habitat_mask] = 0

    consequence_arr[hab_stress_buff_mask] = (
        conseq_num_arr[hab_stress_buff_mask] +
        recov_num_arr[hab_stress_buff_mask]) / (conseq_denom + recov_denom)

    return consequence_arr


def _pair_criteria_num_op(
        habitat_arr, stressor_dist_arr, stressor_buffer, decay_eq, num,
        *spatial_explicit_arr_const):
    """Calculate E or C numerator scores with distance decay equation.

    The equation for calculating the numerator is rating/(dq*weight). This
    function will only calculate the score on pixels where both habitat
    and stressor (including buffer zone) exist. A spatial criteria will be
    added if spatial_explicit_arr_const is provided.

    Args:
        habitat_arr (array): a habitat integer array where 1s indicates habitat
            existence and 0s non-existence.
        stressor_dist_arr (array): a stressor distance float array where pixel
            values represent the distance of that pixel to a stressor
            pixel.
        stressor_buffer (float): a number representing how far down the
            influence is from the stressor pixel.
        decay_eq (str): a string representing the decay format of the
            stressor in the buffer zone. Could be ``None``, ``Linear``
            or ``Exponential``
        num (float): a cumulative value pre-calculated based on the criteria
            table. It will be divided by denominator to get exposure score.
        *spatial_explicit_arr_const: if exists, it is a list of variables
            representing rating float array, DQ, weight, and nodata
            on every four items.

    Returns:
        num_arr (array): a float array of the numerator scores calculated based
            on the E/C equation.

    """
    habitat_mask = (habitat_arr == 1)
    stressor_buff_mask = (stressor_dist_arr <= stressor_buffer)
    # Habitat-stressor overlap mask that includes stressor buffer
    hab_stress_buff_mask = (habitat_mask & stressor_buff_mask)

    # Initialize numerator and denominator arrays and fill the habitat-
    # stressor overlapping pixels with corresponding values
    num_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    num_arr[hab_stress_buff_mask] = num

    # Loop through every 4 items in spatial_explicit_arr_const, and compute
    # the cumulative numerator values over the array
    for spatial_arr, dq, weight, nodata in zip(
            spatial_explicit_arr_const[0::4],
            spatial_explicit_arr_const[1::4],
            spatial_explicit_arr_const[2::4],
            spatial_explicit_arr_const[3::4]):
        # Mask pixels where both habitat, stressor, and spatial array exist
        overlap_mask = hab_stress_buff_mask & ~utils.array_equals_nodata(
                spatial_arr, nodata)

        # Compute the cumulative numerator score
        num_arr[overlap_mask] += spatial_arr[overlap_mask]/(dq*weight)

    # Mask habitat-stressor buffer zone, excluding the stressor itself
    hab_buff_mask = (hab_stress_buff_mask & (stressor_dist_arr > 0))

    # Both linear and exponential decay equations assumed that numerator
    # value is zero outside of buffer zone, and decays over distance
    # in the overlapping area of habitat and buffer zone
    if decay_eq == 'Linear':
        # Linearly decays over distance
        num_arr[hab_buff_mask] = num_arr[hab_buff_mask] * (
            1. - stressor_dist_arr[hab_buff_mask] /
            stressor_buffer)

    elif decay_eq == 'Exponential':
        # This decay rate makes the target numerator zero outside of the
        # stressor buffer
        decay_rate = numpy.log(
            _EXP_DEDAY_CUTOFF/num_arr[hab_buff_mask]) / stressor_buffer
        # Only calculate the decaying numerator score within the buffer zone
        num_arr[hab_buff_mask] = num_arr[hab_buff_mask] * numpy.exp(
            decay_rate *
            stressor_dist_arr[hab_buff_mask])

    return num_arr


def _calc_pair_criteria_score(
        habitat_stressor_overlap_df, habitat_raster_path,
        stressor_dist_raster_path, stressor_buffer, decay_eq, criteria_type,
        recov_params=None):
    """Calculate exposure or consequence scores for a habitat-stressor pair.

    Args:
        habitat_stressor_overlap_df (dataframe): a dataframe that has
            information on stressor and habitat overlap property.
        habitat_raster_path (str): a path to the habitat raster where 0's
            indicate no habitat and 1's indicate habitat existence. 1's will be
            used for calculating recovery potential output raster.
        stressor_dist_raster_path (str): a path to a raster where each pixel
            represents the Euclidean distance to the closest stressor pixel.
        stressor_buffor (float): a number representing how far down the
            influence is from the stressor pixel.
        decay_eq (str): a string representing the decay format of the
            stressor in the buffer zone. Could be ``None``, ``Linear``, or
            ``Exponential``.
        criteria_type (str): a string indicating that this function calculates
            exposure or consequence scores. Could be ``C`` or ``E``. If ``C``,
            recov_score_paths needs to be added.
        recov_params (tuple): a tuple of recovery numerator path and
            denominator score. The former is a path to a raster calculated
            based on habitat resilience attribute. The array  values will be
            added to consequence scores. The later is a precalculated
            cumulative recovery denominator score. Required when criteria_type
            is ``C``.

    Returns:
        None.

    """
    header_list = ['NUM', 'DENOM', 'SPATIAL', 'NUM_RASTER_PATH',
                   'RASTER_PATH']
    header_list = [criteria_type + '_' + header for header in header_list]

    num, denom, spatial_explicit_dict, target_criteria_num_path, \
        target_pair_criteria_raster_path = [
            habitat_stressor_overlap_df.loc[header] for header in header_list]

    # A path and/or constant list for calculating numerator rasters
    num_list = [
        (habitat_raster_path, 1), (stressor_dist_raster_path, 1),
        (stressor_buffer, 'raw'), (decay_eq, 'raw'), (num, 'raw')]

    # A path and/or constant list for calculating final E or C score
    pair_score_list = [
        (habitat_raster_path, 1), (stressor_dist_raster_path, 1),
        (stressor_buffer, 'raw'), (target_criteria_num_path, 1),
        (denom, 'raw')]

    # Iterate through each stressor overlap attribute and append spatial
    # explicit path, DQ, and weight to the path band constant list
    for stressor_attribute_key in spatial_explicit_dict:
        attr_raster_path, dq, weight = spatial_explicit_dict[
            stressor_attribute_key]
        attr_nodata = pygeoprocessing.get_raster_info(attr_raster_path)[
            'nodata'][0]
        num_list.append((attr_raster_path, 1))
        num_list.append((float(dq), 'raw'))
        num_list.append((float(weight), 'raw'))
        num_list.append((attr_nodata, 'raw'))

    # Calculate numerator raster for the habitat-stressor pair
    pygeoprocessing.raster_calculator(
        num_list, _pair_criteria_num_op, target_criteria_num_path,
        _TARGET_PIXEL_FLT, _TARGET_NODATA_FLT)

    # Calculate E or C raster for the habitat-stressor pair. This task is
    # dependent upon the numerator calculation task
    if criteria_type == 'E':
        pygeoprocessing.raster_calculator(
            pair_score_list, _pair_exposure_op,
            target_pair_criteria_raster_path, _TARGET_PIXEL_FLT,
            _TARGET_NODATA_FLT)
    else:
        recov_num_path, recov_denom = recov_params
        # Add recovery numerator raster and denominator scores when calculating
        # consequence score
        pair_score_list.extend([(recov_num_path, 1), (recov_denom, 'raw')])
        pygeoprocessing.raster_calculator(
            pair_score_list, _pair_consequence_op,
            target_pair_criteria_raster_path, _TARGET_PIXEL_FLT,
            _TARGET_NODATA_FLT)


def _tot_recovery_op(habitat_arr, num_arr, denom, max_rating):
    """Calculate and reclassify habitat recovery scores to 1 to 3.

    The equation for calculating reclassified recovery score is:
        score = 3 * (1 - num/denom/max_rating)
    If 0 < score <= 1, reclassify it to 1.
    If 1 < score <= 2, reclassify it to 2.
    If 2 < score <= 3, reclassify it to 3.

    Args:
        habitat_arr (array): a habitat integer array where 1's indicates
            habitat existence and 0's non-existence.
        num_arr (array): a float array of the numerator score for recovery
            potential.
        denom (float): the precalculated cumulative denominator score.
        max_rating (float): the rating used to define the recovery
            reclassified.

    Returns:
        recov_reclass_arr (array): a integer array of the reclassified
            recovery potential scores.

    """
    # Initialize numerator and denominator arrays and fill the habitat
    # pixels with corresponding values
    habitat_mask = (habitat_arr == 1)

    recov_reclass_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_INT, dtype=numpy.int8)

    # Calculate the recovery score by dividing numerator by denominator
    # and then convert it to reclassified by using max_rating
    recov_reclass_arr[habitat_mask] = numpy.ceil(
        3. - num_arr[habitat_mask] / denom / max_rating * 3.).astype(
            numpy.int8)

    return recov_reclass_arr


def _recovery_num_op(habitat_arr, num, *spatial_explicit_arr_const):
    """Calculate the numerator score for recovery potential on a habitat array.

    The equation for calculating the numerator score is rating/(dq*weight).
    This function will only calculate the score on pixels where habitat exists,
    and use a spatial criteria if spatial_explicit_arr_const is provided.

    Args:
        habitat_arr (array): a habitat integer array where 1's indicates
            habitat existence and 0's non-existence.
        num (float): a cumulative value pre-calculated based on the criteria
            table. It will be divided by denominator to get exposure score.
        *spatial_explicit_arr_const: if exists, it is a list of variables
            representing resilience float array, DQ, weight, and nodata
            on every four items.

    Returns:
        num_arr (array): a float array of the numerator score for recovery
            potential.

    """
    # Initialize numerator and denominator arrays and fill the habitat
    # pixels with corresponding values
    habitat_mask = (habitat_arr == 1)

    num_arr = numpy.full(
        habitat_arr.shape, _TARGET_NODATA_FLT, dtype=numpy.float32)
    num_arr[habitat_mask] = num

    # Loop through every 4 items in spatial_explicit_arr_const, and compute the
    # numerator values cumulatively
    for resilience_arr, dq, weight, nodata in zip(
            spatial_explicit_arr_const[0::4],
            spatial_explicit_arr_const[1::4],
            spatial_explicit_arr_const[2::4],
            spatial_explicit_arr_const[3::4]):
        # Mask pixels where both habitat and resilience score exist
        hab_res_overlap_mask = habitat_mask & ~utils.array_equals_nodata(
                resilience_arr, nodata)

        # Compute cumulative numerator score
        num_arr[hab_res_overlap_mask] += resilience_arr[
            hab_res_overlap_mask]/(dq*weight)

    return num_arr


def _calc_habitat_recovery(
        habitat_raster_path, habitat_recovery_df, max_rating):
    """Calculate habitat raster recovery potential based on recovery scores.

    Args:
        habitat_raster_path (str): a path to the habitat raster where 0's
            indicate no habitat and 1's indicate habitat existence. 1's will be
            used for calculating recovery potential output raster.
        recovery_df (dataframe): the dataframe with recovery information such
            as numerator and denominator scores, spatially explicit criteria
            dictionary, and target habitat recovery raster paths for a
            particular habitat.
        max_rating (float): the rating used to reclassify the recovery score.

    Returns:
        None

    """
    # Get a list of cumulative numerator and denominator scores, spatial
    # explicit dict which has habitat-resilience as key and resilience raster
    # path, DQ and weight as values, and an output file paths
    num, denom, spatial_explicit_dict, target_r_num_raster_path, \
        target_recov_raster_path = [
            habitat_recovery_df[column_header] for column_header in [
                'R_NUM', 'R_DENOM', 'R_SPATIAL', 'R_NUM_RASTER_PATH',
                'R_RASTER_PATH']]

    # A list for calculating arrays of cumulative numerator scores
    num_list = [(habitat_raster_path, 1), (num, 'raw')]

    # A list for calculating recovery potential
    recov_potential_list = [
        (habitat_raster_path, 1), (target_r_num_raster_path, 1),
        (denom, 'raw'), (max_rating, 'raw')]

    # Iterate through the spatially explicit criteria dictionary and append its
    # raster path, DQ, and weight to num_list
    for habitat_resilience_key in spatial_explicit_dict:
        resilinece_raster_path, dq, weight = spatial_explicit_dict[
            habitat_resilience_key]
        resilience_nodata = pygeoprocessing.get_raster_info(
            resilinece_raster_path)['nodata'][0]
        num_list.append((resilinece_raster_path, 1))
        num_list.append((float(dq), 'raw'))
        num_list.append((float(weight), 'raw'))
        num_list.append((resilience_nodata, 'raw'))

    # Calculate cumulative numerator score for the habitat
    pygeoprocessing.raster_calculator(
        num_list, _recovery_num_op, target_r_num_raster_path,
        _TARGET_PIXEL_FLT, _TARGET_NODATA_FLT)

    # Finally calculate recovery potential for the habitat
    pygeoprocessing.raster_calculator(
        recov_potential_list, _tot_recovery_op, target_recov_raster_path,
        _TARGET_PIXEL_INT, _TARGET_NODATA_INT)


def _append_spatial_raster_row(info_df, recovery_df, overlap_df,
                               spatial_file_dir, output_dir, suffix_end):
    """Append spatial raster to NAME, PATH, and TYPE column of info_df.

    Args:
        info_df (dataframe): the dataframe to append spatial raster information
            to.
        recovery_df (dataframe): the dataframe that has the spatial raster
            information on its ``R_SPATIAL`` column.
        overlap_df (dataframe): the multi-index dataframe that has the spatial
            raster information on its ``E_SPATIAL`` and ``C_SPATIAL`` columns.
        spatial_file_dir (str): the path to the root directory where the
            absolute paths of spatial files will be created based on.
        output_dir (str): a path to the folder for creating new raster paths at
        suffix_end (str): a suffix to be appended a the end of the filenames.

    Returns:
        info_df (dataframe): a dataframe appended with spatial raster info.

    """
    raster_dicts_list = recovery_df['R_SPATIAL'].tolist() + overlap_df[
        'E_SPATIAL'].tolist() + overlap_df['C_SPATIAL'].tolist()
    # Starting index would be the last index in info_df + 1
    start_idx = info_df.index.values[-1] + 1

    for raster_dict in raster_dicts_list:
        for raster_name in raster_dict:
            # The first item in the list of that raster_name key would be the
            # path to that raster
            raster_path = raster_dict[raster_name][0]
            info_df.loc[start_idx, 'NAME'] = raster_name
            info_df.loc[start_idx, 'PATH'] = raster_path
            info_df.loc[start_idx, 'TYPE'] = _SPATIAL_CRITERIA_TYPE

            # Convert all relative paths to absolute paths
            info_df['PATH'] = info_df.apply(
                lambda row: _to_abspath(row['PATH'], spatial_file_dir), axis=1)
            # Check if the file on the path is a raster or vector
            info_df['IS_RASTER'] = info_df.apply(
                lambda row: _label_raster(row['PATH']), axis=1)
            # Generate simplified vector path if the file is a vector
            info_df['SIMPLE_VECTOR_PATH'] = info_df.apply(
                lambda row: _generate_vector_path(
                    row, output_dir, 'simplified_', suffix_end), axis=1)
            # Generate raster paths which vectors will be rasterized onto
            info_df['BASE_RASTER_PATH'] = info_df.apply(
                lambda row: _generate_raster_path(
                    row, output_dir, 'base_', suffix_end), axis=1)
            # Generate raster paths which will be aligned & resized from base
            # rasters
            info_df['ALIGN_RASTER_PATH'] = info_df.apply(
                lambda row: _generate_raster_path(
                    row, output_dir, 'aligned_', suffix_end), axis=1)
            # Get raster linear unit, and raise an exception if the projection
            # is missing
            info_df['LINEAR_UNIT'] = info_df.apply(
                lambda row: _label_linear_unit(row), axis=1)

            # Replace the raster path in that dict with the new aligned
            # raster path
            raster_dict[raster_name][0] = info_df.loc[
                start_idx, 'ALIGN_RASTER_PATH']

            start_idx += 1

    return info_df


def _to_abspath(base_path, dir_path):
    """Return an absolute path within dir_path if the given path is relative.

    Args:
        base_path (str): a path to the file to be examined.
        dir_path (str): a path to the directory which will be used to create
            absolute file paths.

    Returns:
        target_abs_path (str): an absolutized version of the path.

    Raises:
        ValueError if the file doesn't exist.

    """
    if not os.path.isabs(base_path):
        target_abs_path = os.path.join(dir_path, base_path)

        if not os.path.exists(target_abs_path):
            # the sample data uses windows-style backslash directory separators
            # if the file wasn't found, try converting to posix format,
            # replacing backslashes with forward slashes
            # note that if there's a space in the filename, this won't work
            if os.name == 'posix':
                target_abs_path = target_abs_path.replace('\\', '/')
                if os.path.exists(target_abs_path):
                    return target_abs_path

            raise ValueError(
                'The file on %s does not exist.' % target_abs_path)
        else:
            return target_abs_path

    return base_path


def _label_raster(path):
    """Open a file given the path, and label whether it's a raster.

    If the provided path is a relative path, join it with the dir_path provide.

    Args:
        path (str): a path to the file to be opened with GDAL.

    Returns:
        A string of either 'true', 'false', or 'invalid', indicating the
        file path has a raster, vector, or invalid file type.

    Raises:
        ValueError if the file can't be opened by GDAL.

    """
    raster = gdal.OpenEx(path, gdal.OF_RASTER)
    if raster:
        raster = None
        return True
    else:
        vector = gdal.OpenEx(path, gdal.OF_VECTOR)
        if vector:
            vector = None
            return False
        else:
            raise ValueError(
                'The file on %s is a not a valid GDAL file.' % path)


def _generate_raster_path(row, dir_path, suffix_front, suffix_end):
    """Generate a raster file path with suffixes in dir_path.

    Args:
        row (pandas.Series): a row on the dataframe to get path value from.
        dir_path (str): a path to the folder which raster paths will be
            created based on.
        suffix_end (str): a file suffix to append to the front of filenames.
        suffix_end (str): a file suffix to append to the end of filenames.

    Returns:
        Original path if path is already a raster, or target_raster_path within
        dir_path if it's a vector.

    """
    path = row['PATH']
    # Get file base name from the NAME column
    basename = row['NAME']
    target_raster_path = os.path.join(
        dir_path,
        suffix_front + basename + suffix_end + '.tif')

    # Return the original file path from ``PATH`` if it's already a raster
    if suffix_front == 'base_' and row['IS_RASTER']:
        return path
    # Habitat rasters do not need to be transformed
    elif (suffix_front == 'dist_' or suffix_front == 'buff_') and (
            row['TYPE'] == _HABITAT_TYPE):
        return None

    return target_raster_path


def _generate_vector_path(row, dir_path, suffix_front, suffix_end):
    """Generate a vector file path with suffixes in dir_path for vector types.

    Args:
        row (pandas.Series): a row on the dataframe to get path value from.
        dir_path (str): a path to the folder which vector paths will be
            created based on.
        suffix_end (str): a file suffix to append to the front of filenames.
        suffix_end (str): a file suffix to append to the end of filenames.

    Returns:
        A vector path on dir_path if PATH doesn't contain a raster
        (i.e. a vector), or None if PATH contains a raster.

    """
    if not row['IS_RASTER']:
        # Get file base name from the NAME column
        basename = row['NAME']
        target_vector_path = os.path.join(
            dir_path,
            suffix_front + basename + suffix_end + '.gpkg')
        return target_vector_path

    return None


def _generate_pickle_path(row, dir_path, suffix_front, suffix_end):
    """Generate a pickle file path with suffixes in dir_path for habitat types.

    Args:
        row (pandas.Series): a row on the dataframe to get path value from.
        dir_path (str): a path to the folder which raster paths will be
            created based on.
        suffix_end (str): a file suffix to append to the front of filenames.
        suffix_end (str): a file suffix to append to the end of filenames.

    Returns:
        A pickle path on dir_path if TYPE is habitat, or None if not a habitat.

    """
    if row['TYPE'] == _HABITAT_TYPE:
        # Get file base name from the NAME column
        basename = row['NAME']
        target_pickle_path = os.path.join(
            dir_path,
            suffix_front + basename + suffix_end + '_.pickle')
        return target_pickle_path

    return None


def _label_linear_unit(row):
    """Get linear unit from path, and keep track of paths w/o projection.

    Args:
        row (pandas.Series): a row on the dataframe to get path value from.

    Returns:
        linear_unit (float): the value to multiply by linear distances in order
            to transform them to meters

    Raises:
        ValueError if any of the file's spatial reference is missing or if
            any of the file's are not linearly projected.

    """
    if row['IS_RASTER']:
        raster = gdal.OpenEx(row['PATH'], gdal.OF_RASTER)
        sr_wkt = raster.GetProjection()
        spat_ref = osr.SpatialReference()
        spat_ref.ImportFromWkt(sr_wkt)
        raster = None
    else:
        vector = gdal.OpenEx(row['PATH'], gdal.OF_VECTOR)
        layer = vector.GetLayer()
        spat_ref = layer.GetSpatialRef()
        layer = None
        vector = None

    if not spat_ref or not spat_ref.IsProjected():
        raise ValueError(
            "The following layer does not have a spatial reference or is not"
            f"projected (in linear units): {row['PATH']}")
    else:
        return float(spat_ref.GetLinearUnits())


def _get_info_dataframe(base_info_table_path, file_preprocessing_dir,
                        intermediate_dir, output_dir, suffix_end):
    """Read info table as dataframe and add data info to new columns.

    Add new columns that provide file information and target file paths of each
    given habitat or stressors to the dataframe.

    Args:
        base_info_table_path (str): a path to the CSV or excel file that
            contains the path and buffer information.
        file_preprocessing_dir (str): a path to the folder where simplified
            vectors paths, and base, aligned and distance raster paths will
            be created in.
        intermediate_dir (str): a path to the folder where cumulative
            exposure and consequence raster paths for each habitat will be
            created in.
        output_dir (str): a path to the folder where risk raster path for each
            habitat will be created in.
        suffix_end (str): a file suffix to append to the end of filenames.

    Returns:
        info_df (dataframe): a dataframe that has the information on whether a
            file is a vector, and a raster path column.
        habitat_names (list): a list of habitat names obtained from info file.
        stressor_names (list): a list of stressor names obtained from info file

    Raises:
        ValueError if the input table is not a CSV or Excel file.
        ValueError if any column header is missing from the table.
        ValueError if any input format is not correct.
        ValueError if if any input file does not have a projection.

    """
    required_column_headers = ['NAME', 'PATH', 'TYPE', _BUFFER_HEADER]
    required_types = [_HABITAT_TYPE, _STRESSOR_TYPE]
    required_buffer_type = _STRESSOR_TYPE

    # Read file with pandas based on its type
    file_ext = os.path.splitext(base_info_table_path)[1].lower()
    if file_ext == '.csv':
        # use sep=None, engine='python' to infer what the separator is
        info_df = pandas.read_csv(base_info_table_path, sep=None,
                                  engine='python')
    elif file_ext in ['.xlsx', '.xls']:
        info_df = pandas.read_excel(base_info_table_path)
    else:
        raise ValueError('Info table %s is not a CSV nor an Excel file.' %
                         base_info_table_path)

    # Convert column names to upper case and strip whitespace
    info_df.columns = [col.strip().upper() for col in info_df.columns]

    missing_columns = list(
        set(required_column_headers) - set(info_df.columns.values))

    if missing_columns:
        raise ValueError(
            'Missing column header(s) from the Info CSV file: %s' %
            missing_columns)

    # Drop columns that have all NA values
    info_df.dropna(axis=1, how='all', inplace=True)

    # Convert the values in TYPE column to lowercase first
    info_df.TYPE = [
        val.lower() if isinstance(val, str) else val.encode('utf-8').lower()
        for val in info_df.TYPE]

    unknown_types = list(set(info_df.TYPE) - set(required_types))
    if unknown_types:
        raise ValueError(
            'The ``TYPE`` attribute in Info table could only have either %s '
            'or %s as its value, but is having %s' % (
                required_types[0], required_types[1], unknown_types))

    buffer_column_dtype = info_df[info_df.TYPE == required_buffer_type][
        _BUFFER_HEADER].dtype
    if not numpy.issubdtype(buffer_column_dtype, numpy.number):
        raise ValueError(
            'The %s attribute in Info table should be a number for stressors, '
            'and empty for habitats.' % _BUFFER_HEADER)

    # Convert all relative paths to absolute paths
    info_df['PATH'] = info_df.apply(
        lambda row: _to_abspath(
            row['PATH'], os.path.dirname(base_info_table_path)), axis=1)
    # Check if the file on the path is a raster or vector
    info_df['IS_RASTER'] = info_df.apply(
        lambda row: _label_raster(row['PATH']), axis=1)
    # Get raster's linear unit, and raise an exception if projection is missing
    info_df['LINEAR_UNIT'] = info_df.apply(
        lambda row: _label_linear_unit(row), axis=1)
    # Generate simplified vector path if the file is a vector
    info_df['SIMPLE_VECTOR_PATH'] = info_df.apply(
        lambda row: _generate_vector_path(
            row, file_preprocessing_dir, 'simplified_', suffix_end), axis=1)
    # Generate raster paths which vectors will be rasterized onto
    info_df['BASE_RASTER_PATH'] = info_df.apply(
        lambda row: _generate_raster_path(
            row, file_preprocessing_dir, 'base_', suffix_end), axis=1)
    # Generate raster paths which will be aligned & resized from base rasters
    info_df['ALIGN_RASTER_PATH'] = info_df.apply(
        lambda row: _generate_raster_path(
            row, file_preprocessing_dir, 'aligned_', suffix_end), axis=1)
    # Generate distance raster paths which is transformed from aligned rasters
    info_df['DIST_RASTER_PATH'] = info_df.apply(
        lambda row: _generate_raster_path(
            row, file_preprocessing_dir, 'dist_', suffix_end), axis=1)

    for column_name, criteria_type in {
            'TOT_E_RASTER_PATH': '_E_',
            'TOT_C_RASTER_PATH': '_C_'}.items():
        suffix_front = 'TOTAL' + criteria_type  # front suffix for file names
        # Generate raster paths with exposure and consequence suffixes.
        info_df[column_name] = info_df.apply(
            lambda row: _generate_raster_path(
                row, intermediate_dir, suffix_front, suffix_end), axis=1)
        # Generate pickle path to storing criteria score stats
        info_df['TOT'+criteria_type+'PICKLE_STATS_PATH'] = info_df.apply(
            lambda row: _generate_pickle_path(
                row, file_preprocessing_dir, suffix_front, suffix_end), axis=1)

    # Generate cumulative risk raster paths with risk suffix.
    info_df['TOT_RISK_RASTER_PATH'] = info_df.apply(
        lambda row: _generate_raster_path(
            row, output_dir, 'TOTAL_RISK_', suffix_end), axis=1)
    # Generate pickled statistics for habitat risks
    info_df['TOT_RISK_PICKLE_STATS_PATH'] = info_df.apply(
        lambda row: _generate_pickle_path(
            row, file_preprocessing_dir, 'TOTAL_RISK_', suffix_end), axis=1)

    # Generate reclassified risk raster paths with risk suffix.
    info_df['RECLASS_RISK_RASTER_PATH'] = info_df.apply(
        lambda row: _generate_raster_path(
            row, output_dir, 'RECLASS_RISK_', suffix_end), axis=1)

    # Get lists of habitat and stressor names
    habitat_names = info_df[info_df.TYPE == _HABITAT_TYPE].NAME.tolist()
    stressor_names = info_df[info_df.TYPE == _STRESSOR_TYPE].NAME.tolist()

    return info_df, habitat_names, stressor_names


def _get_criteria_dataframe(base_criteria_table_path):
    """Get validated criteria dataframe from a criteria table.

    Args:
        base_criteria_table_path (str): a path to the CSV or Excel file with
            habitat and stressor criteria ratings.

    Returns:
        criteria_df (dataframe): a validated dataframe converted from the table
            where None represents an empty cell.

    Raises:
        ValueError if the input table is not a CSV or Excel file.
        ValueError if any required index or column header is missing from the
            table.

    """
    # Read the table into dataframe based on its type, with first column as
    # index. Column names are auto-generated ordinal values
    file_ext = os.path.splitext(base_criteria_table_path)[1].lower()
    if file_ext == '.csv':
        # use sep=None, engine='python' to infer what the separator is
        criteria_df = pandas.read_csv(
            base_criteria_table_path, index_col=0, header=None, sep=None,
            engine='python')
    elif file_ext in ['.xlsx', '.xls']:
        criteria_df = pandas.read_excel(base_criteria_table_path,
                                        index_col=0, header=None)
    else:
        raise ValueError('Criteria table %s is not a CSV or an Excel file.' %
                         base_criteria_table_path)

    # Drop columns that have all NA values
    criteria_df.dropna(axis=1, how='all', inplace=True)

    # Convert empty cells (those that are not in string or unicode format)
    # to None
    criteria_df.index = [x if isinstance(x, str) else None
                         for x in criteria_df.index]

    # Verify the values in the index column, and append to error message if
    # there's any missing index
    required_indexes = [_HABITAT_NAME_HEADER, _HABITAT_RESILIENCE_HEADER,
                        _HABITAT_STRESSOR_OVERLAP_HEADER]

    missing_indexes = set(required_indexes) - set(criteria_df.index.values)

    if missing_indexes:
        raise ValueError('The Criteria table is missing the following '
                         'value(s) in the first column: %s.\n' %
                         list(missing_indexes))

    # Validate the column header, which should have 'criteria type'
    criteria_df.columns = [
        x.strip() if isinstance(x, str) else None for x in
        criteria_df.loc[_HABITAT_NAME_HEADER].values]
    if _CRITERIA_TYPE_HEADER not in criteria_df.columns.values:
        raise ValueError('The Criteria table is missing the column header'
                         ' "%s".' % _CRITERIA_TYPE_HEADER)

    LOGGER.info('Criteria dataframe was created successfully.')

    return criteria_df


def _get_attributes_from_df(criteria_df, habitat_names, stressor_names):
    """Get habitat names, resilience attributes, stressor attributes info.

    Get the info from the criteria dataframe.

    Args:
        criteria_df (dataframe): a validated dataframe with required
            fields in it.
        habitat_names (list): a list of habitat names obtained from info table.
        stressor_names (list): a list of stressor names obtained from info
            table.

    Returns:
        resilience_attributes (list): a list of resilience attributes used for
            getting rating, dq, and weight for each attribute.
        stressor_attributes (dict): a dictionary with stressor names as keys,
            and a list of overlap properties (strings) as values.

    Raises:
        ValueError if criteria_df does not have names from habitat_names and
            stressor_names.
        ValueError if a stressor criteria shows up before any stressor.

    """
    # Get habitat names from the first row
    missing_habitat_names = list(
        set(habitat_names) - set(criteria_df.columns.values))

    missing_stressor_names = list(
        set(stressor_names) - set(criteria_df.index.values))

    missing_names_error_message = ''
    if missing_habitat_names:
        missing_names_error_message += (
            'The following Habitats in the info table are missing from the '
            'criteria table: %s. ' % missing_habitat_names)
    if missing_stressor_names:
        missing_names_error_message += (
            'The following Stresors in the info table are missing from the '
            'criteria table: %s' % missing_stressor_names)
    if missing_names_error_message:
        raise ValueError(missing_names_error_message)

    # Get habitat resilience attributes
    resilience_attributes = []
    found_resilience_header = False

    for idx, field in enumerate(criteria_df.index.values):
        # Make an empty list when the habitat resilience header shows up
        if field == _HABITAT_RESILIENCE_HEADER:
            found_resilience_header = True
            continue

        # Add the field to the list if it's after the resilience header and
        # before the overlap header
        if found_resilience_header:
            if field != _HABITAT_STRESSOR_OVERLAP_HEADER:
                # Append the field if the cell is not empty
                if field is not None:
                    resilience_attributes.append(field)
            else:
                # Get the index of overlap header when it's reached
                last_idx = idx
                break

    LOGGER.debug('resilience_attributes: %s' % resilience_attributes)

    # Make a dictionary of stressor (key) with its attributes (value)
    stressor_attributes = {}
    # Enumerate from the overlap header
    stressor_overlap_indexes = criteria_df.index.values[(last_idx+1):]
    current_stressor = None
    for field in stressor_overlap_indexes:
        if field is not None:
            if field in stressor_names:
                # Set the current stressor to the encountered stressor and
                # add it to the attributes table
                current_stressor = field
                stressor_attributes[current_stressor] = []
                continue

            # Append the field as a stressor attribute if it's not a stressor
            elif current_stressor:
                stressor_attributes[current_stressor].append(field)

            # Raise an exception if a criteria shows up before a stressor
            else:
                raise ValueError('The "%s" criteria does not belong to any '
                                 'stressors. Please check your criteria table.'
                                 % field)

    LOGGER.debug('stressor_attributes: %s' % stressor_attributes)

    return resilience_attributes, stressor_attributes


def _validate_rating(
        rating, max_rating, criteria_name, habitat, stressor=None):
    """Validate rating value, which should range between 1 to maximum rating.

    Args:
        rating (str): a string of either digit or file path. If it's a digit,
            it should range between 1 to maximum rating. It could be a filepath
            if the user is using spatially-explicit ratings.
        max_rating (float): a number representing the highest value that
            is represented in criteria rating.
        criteria_name (str): the name of the criteria attribute where rating
            is from.
        habitat (str): the name of the habitat where rating is from.
        stressor (str): the name of the stressor where rating is from. Can be
            None when we're checking the habitat-only attributes. (optional)

    Returns:
        True for values from 1 to max_rating.
        False for values less than 1.

    Raises:
        ValueError if the rating score is larger than the maximum rating
            or if the rating is numpy.nan, indicating a missing rating.

    """
    try:
        num_rating = float(rating)
    except ValueError:
        # assume it's a path to a file, which is validated elsewhere
        return True

    message_prefix = '"%s" for habitat %s' % (criteria_name, habitat)
    if stressor:
        message_prefix += (' and stressor %s' % stressor)

    if num_rating < 1:
        warning_message = message_prefix + (
            ' has a rating %s less than 1, so this criteria attribute is '
            'ignored in calculation.' % num_rating)
        LOGGER.warning(warning_message)
        return False

    if num_rating > float(max_rating):
        error_message = message_prefix + (
            ' has a rating %s larger than the maximum rating %s. '
            'Please check your criteria table.' % (rating, max_rating))
        raise ValueError(error_message)

    if numpy.isnan(num_rating):
        raise ValueError(
            f'{message_prefix} has no rating. Please check the criteria table.')

    return True


def _validate_dq_weight(dq, weight, habitat, stressor=None):
    """Check if DQ and Weight column values are numbers and not 0.

    Args:
        dq (str): a string representing the value of data quality score.
        weight (str): a string representing the value of weight score.
        habitat (str): the name of the habitat where the score is from.
        stressor (str): the name of the stressor where the score is from. Can
            be None when we're checking the habitat-only attributes. (optional)

    Returns:
        None

    Raises:
        ValueError if the value of the DQ or weight is 0 or not a number.

    """
    for key, value in {
            _DQ_KEY: dq,
            _WEIGHT_KEY: weight}.items():

        # The value might be NaN or a string of non-digit, therefore check for
        # both cases
        error_message = (
            'Values in the %s column for habitat "%s" ' % (key, habitat))
        if stressor:
            error_message += 'and stressor "%s"' % stressor
        error_message += ' should be a number, but is "%s".' % value

        try:
            num_value = float(value)
        except ValueError:
            raise ValueError(error_message)

        if numpy.isnan(num_value) or num_value == 0:
            raise ValueError(error_message)


def _get_overlap_dataframe(criteria_df, habitat_names, stressor_attributes,
                           max_rating, inter_dir, output_dir, suffix):
    """Return a dataframe based on habitat-stressor overlap properties.

    Calculation on exposure or consequence score will need or build information
    on numerator, denominator, spatially explicit criteria dict, final score
    raster path, numerator raster path, and mean score statistics dict. The
    spatially explicit criteria scores will be added to the score calculation
    later on.

    Args:
        criteria_df (dataframe): a validated dataframe with required fields.
        habitat_names (list): a list of habitat names used as dataframe index.
        stressor_attributes (dict): a dictionary with stressor names as keys,
            and a list of overlap criteria (strings) as values.
        max_rating (float): a number representing the highest value that
            is represented in criteria rating.
        inter_dir (str): a path to the folder where numerator/denominator E/C
            paths will be created in.
        output_dir (str): a path to the folder where E/C raster paths will be
            created in.
        suffix (str): a file suffix to append to the end of filenames.

    Returns:
        overlap_df (dataframe): a multi-index dataframe with E/C scores for
            each habitat and stressor pair.

    Raises:
        ValueError if the value of the criteria type column from criteria_df
            is not either E or C.
        ValueError if the value of the rating column from criteria_df is less
            than 1 or larger than the maximum rating.
        ValueError if the value of the DQ or weight column from criteria_df is
            not a number or is a number less than 1.
        ValueError if any stressor-habitat does not have at least one E and C
            criteria rating.

    """
    # Create column headers and initialize default values in numerator,
    # denominator, and spatial columns
    overlap_column_headers = [
        'E_NUM', 'E_DENOM', 'E_SPATIAL', 'C_NUM', 'C_DENOM', 'C_SPATIAL']

    # Create an empty dataframe, indexed by habitat-stressor pairs.
    stressor_names = stressor_attributes.keys()
    multi_index = pandas.MultiIndex.from_product(
        iterables=[habitat_names, stressor_names],
        names=[_HABITAT_HEADER, _STRESSOR_HEADER])
    LOGGER.debug('multi_index: %s' % multi_index)

    # Create a multi-index dataframe and fill in default cell values
    overlap_df = pandas.DataFrame(
        # Data values on each row corresponds to each column header
        data=[[0, 0, {}, 0, 0, {}]
              for _ in range(len(habitat_names)*len(stressor_names))],
        columns=overlap_column_headers, index=multi_index)

    # Start iterating from row indicating the beginning of habitat and stressor
    # overlap criteria
    stressor = None
    for row_idx, row_data in criteria_df.loc[
            _HABITAT_STRESSOR_OVERLAP_HEADER:].iterrows():
        if row_idx in stressor_attributes:
            # Start keeping track of the number of overlap criteria used
            # for a stressor found from the row
            stressor = row_idx

        # If stressor exists and the row index is not None
        elif stressor and row_idx:
            criteria_name = row_idx
            criteria_type = row_data.pop(_CRITERIA_TYPE_HEADER)
            criteria_type = criteria_type.upper()
            if criteria_type not in ['E', 'C']:
                raise ValueError('Criteria Type in the criteria scores table '
                                 'should be either E or C.')

            # Values are always grouped in threes (rating, dq, weight)
            for idx in range(0, row_data.size, 3):
                habitat = row_data.keys()[idx]
                if habitat not in habitat_names:
                    # This is how we ignore extra columns in the csv
                    # like we have in the sample data for "Rating Instruction".
                    break
                rating = row_data[idx]
                dq = row_data[idx + 1]
                weight = row_data[idx + 2]

                # Create E or C raster paths on habitat-stressor pair
                overlap_df.loc[
                    (habitat, stressor),
                    criteria_type + '_RASTER_PATH'] = os.path.join(
                        output_dir, criteria_type + '_' + habitat
                        + '_' + stressor + suffix + '.tif')
                overlap_df.loc[
                    (habitat, stressor),
                    criteria_type + '_NUM_RASTER_PATH'] = os.path.join(
                        inter_dir, criteria_type + '_num_' +
                        habitat + '_' + stressor + suffix + '.tif')

                # Create individual habitat-stressor risk raster path
                overlap_df.loc[
                    (habitat, stressor),
                    'PAIR_RISK_RASTER_PATH'] = os.path.join(
                        output_dir, 'RISK_' +
                        habitat + '_' + stressor + suffix + '.tif')

                # Create pickle file path that stores zonal stats dict
                overlap_df.loc[
                    (habitat, stressor), criteria_type +
                    '_PICKLE_STATS_PATH'] = os.path.join(
                        inter_dir, criteria_type + '_' +
                        habitat + '_' + stressor + suffix + '_.pickle')
                overlap_df.loc[
                    (habitat, stressor),
                    'PAIR_RISK_PICKLE_STATS_PATH'] = os.path.join(
                        inter_dir, 'risk_' +
                        habitat + '_' + stressor + suffix + '_.pickle')

                _ = _validate_rating(
                    rating, max_rating, criteria_name, habitat, stressor)

                # Check the DQ and weight values when we have collected
                # both of them
                _validate_dq_weight(dq, weight, habitat, stressor)
                # Calculate cumulative numerator score if rating is a digit
                if (isinstance(rating, str) and rating.isdigit()) or (
                        isinstance(rating, (int, float))):
                    overlap_df.loc[(habitat, stressor),
                                   criteria_type + '_NUM'] += \
                        float(rating)/float(dq)/float(weight)

                # Save the rating, dq, and weight to the spatial criteria
                # dictionary in the dataframe if rating is not a number
                else:
                    overlap_df.loc[
                        (habitat, stressor),
                        criteria_type + '_SPATIAL']['_'.join(
                            [habitat, stressor, criteria_name])] = [
                                rating, dq, weight]

                # Calculate the cumulative denominator score
                overlap_df.loc[
                    (habitat, stressor), criteria_type + '_DENOM'] += \
                    1/float(dq)/float(weight)

    # If any stressor-habitat doesn't have at least one E or C criteria rating,
    # raise an exception
    for criteria_type, criteria_type_long in {
            'E': 'exposure', 'C': 'consequence'}.items():
        if (overlap_df[criteria_type + '_DENOM'] == 0).any():
            raise ValueError(
                'The following stressor-habitat pair(s) do not have at least '
                'one %s rating: %s' % (criteria_type_long, overlap_df[
                    overlap_df[criteria_type + '_DENOM'] == 0].index.tolist()))

    LOGGER.info('Overlap dataframe was created successfully.')
    return overlap_df


def _get_recovery_dataframe(criteria_df, habitat_names, resilience_attributes,
                            max_rating, inter_dir, output_dir, suffix):
    """Return a dataframe with calculated habitat resilience scores.

    The calculation of recovery score will need or build information on
    numerator, denominator, spatially explicit criteria dict, score raster
    path, and numerator raster path.

    Args:
        criteria_df (dataframe): a validated dataframe with required
            fields.
        habitat_names (list): a list of habitat names used as dataframe index.
        resilience_attributes (list): a list of resilience attributes used for
            getting rating, dq, and weight for each attribute.
        max_rating (float): a number representing the highest value that
            is represented in criteria rating.
        inter_dir (str): a path to the folder where recovery numerator score
            paths will be created.
        output_dir (str): a path to the folder where recovery raster paths will
            be created in.
        suffix (str): a file suffix to append to the end of filenames.

    Returns:
        recovery_df (dataframe): the dataframe with recovery information for
            each habitat.

    Raises:
        ValueError if the value of the rating column from criteria_df is less
            than 1 or larger than the maximum rating.
        ValueError if the value of the DQ or weight column from criteria_df is
            not a number or is a number less than 1.

    """
    # Create column headers to keep track of data needed to calculate recovery
    # scores for each habitat
    recovery_column_headers = [
        'R_NUM', 'R_DENOM', 'R_SPATIAL', 'R_RASTER_PATH', 'R_NUM_RASTER_PATH']

    # Create the dataframe whose data is 0 for numerators and denominators,
    # None for raster paths, and an empty dict for spatially explicit criteria.
    recovery_df = pandas.DataFrame(
        data=[[0, 0, {}, None, None] for _ in range(len(habitat_names))],
        index=habitat_names, columns=recovery_column_headers)

    i = 0
    # The loop through the column headers that has habitat names in itself
    while i < len(criteria_df.columns.values):
        # If the column header is in the habitat list, get the habitat name,
        # which will be used as index in recovery_df
        if criteria_df.columns.values[i] in habitat_names:
            habitat = criteria_df.columns.values[i]
            # Create recovery raster paths for later calculation
            recovery_df.loc[habitat, 'R_NUM_RASTER_PATH'] = os.path.join(
                inter_dir, 'RECOV_num_' + habitat + suffix + '.tif')

            recovery_df.loc[habitat, 'R_RASTER_PATH'] = os.path.join(
                output_dir, 'RECOVERY_' + habitat + suffix + '.tif')

            # Calculate cumulative numerator and denominator scores based on
            # each habitat's resilience rating, dq, and weight
            for resilience_attr in resilience_attributes:
                rating = criteria_df.loc[resilience_attr, habitat]
                dq = criteria_df.loc[resilience_attr][i+1]
                weight = criteria_df.loc[resilience_attr][i+2]

                # Check the DQ and weight values
                _validate_dq_weight(dq, weight, habitat)

                # If rating is less than 1, skip this criteria row
                if not _validate_rating(
                        rating, max_rating, resilience_attr, habitat):
                    continue

                # If rating is a number, calculate the numerator score
                if (isinstance(rating, str) and rating.isdigit()) or (
                        isinstance(rating, (int, float))):
                    recovery_df.loc[habitat, 'R_NUM'] += \
                        float(rating)/float(dq)/float(weight)
                else:
                    # If rating is not a number, store the file path, dq &
                    # weight in the dictionary
                    recovery_df.loc[habitat, 'R_SPATIAL'][
                        habitat + '_' + resilience_attr] = [rating, dq, weight]

                # Add 1/(dq*w) to the denominator
                recovery_df.loc[habitat, 'R_DENOM'] += (
                    1/float(dq)/float(weight))

            i += 3  # Jump to next habitat
        else:
            i += 1  # Keep finding the next habitat from the habitat list

    LOGGER.info('Recovery dataframe was created successfully.')
    return recovery_df


def _simplify_geometry(
        base_vector_path, tolerance, target_simplified_vector_path,
        preserved_field=None):
    """Simplify all the geometry in the vector.

    See https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
    for the math prove.

    Args:
        base_vector_path (string): path to base vector.
        tolerance (float): all new vertices in the geometry will be within
            this distance (in units of the vector's projection).
        target_simplified_vector_path (string): path to desired simplified
            vector.
        preserved_field (tuple): a tuple of field name (string) and field type
            (OGRFieldType) that will remain in the target simplified vector.
            Field name will be converted to lowercased.

    Returns:
        None

    """
    base_vector = ogr.Open(base_vector_path)
    base_layer = base_vector.GetLayer()
    target_field_name = None
    if preserved_field:
        # Convert the field name to lowercase
        preserved_field_name = preserved_field[0].lower()
        for base_field in base_layer.schema:
            base_field_name = base_field.GetName().lower()
            # Find the first field name, case-insensitive
            if base_field_name == preserved_field_name:
                # Create a target field definition with lowercased field name
                target_field_name = str(preserved_field_name)
                target_field = ogr.FieldDefn(
                    target_field_name, preserved_field[1])
                break

    target_layer_name = os.path.splitext(
        os.path.basename(target_simplified_vector_path))[0]

    if os.path.exists(target_simplified_vector_path):
        os.remove(target_simplified_vector_path)

    gpkg_driver = gdal.GetDriverByName('GPKG')

    target_simplified_vector = gpkg_driver.Create(
        target_simplified_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    target_simplified_layer = target_simplified_vector.CreateLayer(
        target_layer_name,
        base_layer.GetSpatialRef(), base_layer.GetGeomType())

    target_simplified_vector.StartTransaction()

    if target_field_name:
        target_simplified_layer.CreateField(target_field)

    for base_feature in base_layer:
        target_feature = ogr.Feature(target_simplified_layer.GetLayerDefn())
        base_geometry = base_feature.GetGeometryRef()

        # Use SimplifyPreserveTopology to prevent features from missing
        simplified_geometry = base_geometry.SimplifyPreserveTopology(tolerance)
        if (simplified_geometry is not None and
                simplified_geometry.GetArea() > 0):
            target_feature.SetGeometry(simplified_geometry)
            # Set field value to the field name that needs to be preserved
            if target_field_name:
                field_value = base_feature.GetField(target_field_name)
                target_feature.SetField(target_field_name, field_value)
            target_simplified_layer.CreateFeature(target_feature)

        # If simplify doesn't work, fall back to the original geometry
        else:
            # Still using the target_feature here because the preserve_field
            # option altered the layer defn between base and target.
            target_feature.SetGeometry(base_geometry)
            target_simplified_layer.CreateFeature(target_feature)
        base_geometry = None

    target_simplified_layer.SyncToDisk()
    target_simplified_vector.CommitTransaction()

    target_simplified_layer = None
    target_simplified_vector = None


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
