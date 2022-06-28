"""InVEST Scenic Quality Model."""
import os
import math
import logging
import tempfile
import shutil
import time

import numpy
from osgeo import gdal
from osgeo import osr
import taskgraph
import pygeoprocessing
import rtree
import shapely.geometry

from natcap.invest.scenic_quality.viewshed import viewshed
from .. import utils
from .. import spec_utils
from ..spec_utils import u
from .. import validation
from ..model_metadata import MODEL_METADATA
from .. import gettext


LOGGER = logging.getLogger(__name__)
_VALUATION_NODATA = -99999  # largish negative nodata value.
_BYTE_NODATA = 255  # Largest value a byte can hold
BYTE_GTIFF_CREATION_OPTIONS = (
    'GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
              'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
FLOAT_GTIFF_CREATION_OPTIONS = (
    'GTIFF', ('PREDICTOR=3',) + BYTE_GTIFF_CREATION_OPTIONS[1])

_OUTPUT_BASE_FILES = {
    'viewshed_value': 'vshed_value.tif',
    'n_visible_structures': 'vshed.tif',
    'viewshed_quality': 'vshed_qual.tif',
}

_INTERMEDIATE_BASE_FILES = {
    'aoi_reprojected': 'aoi_reprojected.shp',
    'clipped_dem': 'dem_clipped.tif',
    'structures_clipped': 'structures_clipped.shp',
    'structures_reprojected': 'structures_reprojected.shp',
    'visibility_pattern': 'visibility_{id}.tif',
    'auxiliary_pattern': 'auxiliary_{id}.tif',  # Retained for debugging.
    'value_pattern': 'value_{id}.tif',
}


ARGS_SPEC = {
    "model_name": MODEL_METADATA["scenic_quality"].model_title,
    "pyname": MODEL_METADATA["scenic_quality"].pyname,
    "userguide": MODEL_METADATA["scenic_quality"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["aoi_path", "structure_path", "dem_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "aoi_path": {
            **spec_utils.AOI,
        },
        "structure_path": {
            "name": gettext("features impacting scenic quality"),
            "type": "vector",
            "geometries": spec_utils.POINT,
            "fields": {
                "radius": {
                    "type": "number",
                    "units": u.meter,
                    "required": False,
                    "about": gettext(
                        "Maximum length of the line of sight originating from "
                        "a viewpoint. The value can either be positive "
                        "(preferred) or negative (kept for backwards "
                        "compatibility), but is converted to a positive "
                        "number. If this field is not provided, the model "
                        "will include all pixels in the DEM in the visibility "
                        "analysis. RADIUS preferred, but may also be called "
                        "RADIUS2 for backwards compatibility.")},
                "weight": {
                    "type": "number",
                    "units": u.none,
                    "required": False,
                    "about": gettext(
                        "Viewshed importance coefficient. If this field is "
                        "provided, the values are used to weight each "
                        "feature's viewshed impacts. If not provided, all "
                        "viewsheds are equally weighted with a weight of 1.")},
                "height": {
                    "type": "number",
                    "units": u.meter,
                    "required": False,
                    "about": gettext(
                        "Viewpoint height, the elevation above the ground of "
                        "each feature. If this field is not provided, "
                        "defaults to 0.")}
            },
            "about": gettext(
                "Map of locations of objects that negatively affect scenic "
                "quality. This must have the same projection as the DEM.")
        },
        "dem_path": {
            **spec_utils.DEM,
            "projected": True,
            "projection_units": u.meter
        },
        "refraction": {
            "name": gettext("refractivity coefficient"),
            "type": "ratio",
            "about": gettext(
                "The refractivity coefficient corrects for the curvature of "
                "the earth and refraction of visible light in air.")
        },
        "do_valuation": {
            "name": gettext("run valuation"),
            "type": "boolean",
            "required": False,
            "about": gettext("Run the valuation model.")
        },
        "valuation_function": {
            "name": gettext("Valuation function"),
            "type": "option_string",
            "required": "do_valuation",
            "options": {
                "linear": {"display_name": gettext("linear: a + bx")},
                "logarithmic": {"display_name": gettext(
                    "logarithmic: a + b log(x+1)")},
                "exponential": {"display_name": gettext("exponential: a * e^(-bx)")}
            },
            "about": gettext(
                "Valuation function used to calculate the visual impact of "
                "each feature, given distance from the feature 'x' and "
                "parameters 'a' and 'b'."),
        },
        "a_coef": {
            "name": gettext("coefficient a"),
            "type": "number",
            "units": u.none,
            "required": "do_valuation",
            "about": gettext("First coefficient ('a') used by the valuation function"),
        },
        "b_coef": {
            "name": gettext("coefficient b"),
            "type": "number",
            "units": u.none,
            "required": "do_valuation",
            "about": gettext("Second coefficient ('b') used by the valuation function"),
        },
        "max_valuation_radius": {
            "name": gettext("maximum valuation radius"),
            "type": "number",
            "units": u.meter,
            "required": False,
            "expression": "value > 0",
            "about": gettext(
                "Valuation will only be computed for cells that fall within "
                "this radius of a feature impacting scenic quality."),
        },
    }
}


def execute(args):
    """Scenic Quality.

    Args:
        args['workspace_dir'] (string): (required) output directory for
            intermediate, temporary, and final files.
        args['results_suffix'] (string): (optional) string to append to any
            output file.
        args['aoi_path'] (string): (required) path to a vector that
            indicates the area over which the model should be run.
        args['structure_path'] (string): (required) path to a point vector
            that has the features for the viewpoints. Optional fields:
            'WEIGHT', 'RADIUS' / 'RADIUS2', 'HEIGHT'
        args['dem_path'] (string): (required) path to a digital elevation model
            raster.
        args['refraction'] (float): (required) number indicating the refraction
            coefficient to use for calculating curvature of the earth.
        args['do_valuation'] (bool): (optional) indicates whether to compute
            valuation. If ``False``, per-viewpoint value will not be computed,
            and the summation of valuation rasters (vshed_value.tif) will not
            be created. Additionally, the Viewshed Quality raster will
            represent the weighted sum of viewsheds. Default: ``False``.
        args['valuation_function'] (string): The type of economic
            function to use for valuation. One of "linear", "logarithmic",
            or "exponential".
        args['a_coef'] (float): The "a" coefficient for valuation. Required
            if ``args['do_valuation']`` is ``True``.
        args['b_coef'] (float): The "b" coefficient for valuation. Required
            if ``args['do_valuation']`` is ``True``.
        args['max_valuation_radius'] (float): Past this distance
            from the viewpoint, the valuation raster's pixel values will be set
            to 0. Required if ``args['do_valuation']`` is ``True``.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model. If omitted, computation will take
            place in the current process.

    Returns:
        ``None``

    """
    LOGGER.info("Starting Scenic Quality Model")
    dem_raster_info = pygeoprocessing.get_raster_info(args['dem_path'])

    try:
        do_valuation = bool(args['do_valuation'])
    except KeyError:
        do_valuation = False

    if do_valuation:
        valuation_coefficients = {
            'a': float(args['a_coef']),
            'b': float(args['b_coef']),
        }
        if (args['valuation_function'] not in
                ARGS_SPEC['args']['valuation_function']['options']):
            raise ValueError('Valuation function type %s not recognized' %
                             args['valuation_function'])
        max_valuation_radius = float(args['max_valuation_radius'])

    # Create output and intermediate directory
    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    utils.make_directories([output_dir, intermediate_dir])

    file_suffix = utils.make_suffix_string(
        args, 'results_suffix')

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir)],
        file_suffix)

    work_token_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous execution
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    reprojected_aoi_task = graph.add_task(
        pygeoprocessing.reproject_vector,
        args=(args['aoi_path'],
              dem_raster_info['projection_wkt'],
              file_registry['aoi_reprojected']),
        target_path_list=[file_registry['aoi_reprojected']],
        task_name='reproject_aoi_to_dem')

    reprojected_viewpoints_task = graph.add_task(
        pygeoprocessing.reproject_vector,
        args=(args['structure_path'],
              dem_raster_info['projection_wkt'],
              file_registry['structures_reprojected']),
        target_path_list=[file_registry['structures_reprojected']],
        task_name='reproject_structures_to_dem')

    clipped_viewpoints_task = graph.add_task(
        _clip_vector,
        args=(file_registry['structures_reprojected'],
              file_registry['aoi_reprojected'],
              file_registry['structures_clipped']),
        target_path_list=[file_registry['structures_clipped']],
        dependent_task_list=[reprojected_aoi_task,
                             reprojected_viewpoints_task],
        task_name='clip_reprojected_structures_to_aoi')

    clipped_dem_task = graph.add_task(
        _clip_and_mask_dem,
        args=(args['dem_path'],
              file_registry['aoi_reprojected'],
              file_registry['clipped_dem'],
              intermediate_dir),
        target_path_list=[file_registry['clipped_dem']],
        dependent_task_list=[reprojected_aoi_task],
        task_name='clip_dem_to_aoi')

    # viewshed calculation requires that the DEM and structures are all
    # finished.
    LOGGER.info('Waiting for clipping to finish')
    clipped_dem_task.join()
    clipped_viewpoints_task.join()

    # phase 2: calculate viewsheds.
    valid_viewpoints_task = graph.add_task(
        _determine_valid_viewpoints,
        args=(file_registry['clipped_dem'],
              file_registry['structures_clipped']),
        store_result=True,
        dependent_task_list=[clipped_viewpoints_task, clipped_dem_task],
        task_name='determine_valid_viewpoints')

    viewpoint_tuples = valid_viewpoints_task.get()
    if not viewpoint_tuples:
        raise ValueError('No valid viewpoints found. This may happen if '
                         'viewpoints are beyond the edge of the DEM or are '
                         'over nodata pixels.')

    # These are sorted outside the vector to ensure consistent ordering. This
    # helps avoid unnecessary recomputation in taskgraph for when an ESRI
    # Shapefile, for example, returns a different order of points because
    # someone decided to repack it.
    viewshed_files = []
    viewshed_tasks = []
    valuation_tasks = []
    valuation_filepaths = []
    weights = []
    feature_index = 0
    for viewpoint, max_radius, weight, viewpoint_height in sorted(
            viewpoint_tuples, key=lambda x: x[0]):
        weights.append(weight)
        visibility_filepath = file_registry['visibility_pattern'].format(
            id=feature_index)
        viewshed_files.append(visibility_filepath)
        viewshed_task = graph.add_task(
            viewshed,
            args=((file_registry['clipped_dem'], 1),  # DEM
                  viewpoint,
                  visibility_filepath),
            kwargs={'curved_earth': True,  # SQ model always assumes this.
                    'refraction_coeff': float(args['refraction']),
                    'max_distance': max_radius,
                    'viewpoint_height': viewpoint_height,
                    'aux_filepath': None},  # Remove aux filepath after run
            target_path_list=[visibility_filepath],
            dependent_task_list=[clipped_dem_task,
                                 clipped_viewpoints_task],
            task_name='calculate_visibility_%s' % feature_index)
        viewshed_tasks.append(viewshed_task)

        if do_valuation:
            # calculate valuation
            viewshed_valuation_path = file_registry['value_pattern'].format(
                id=feature_index)
            valuation_task = graph.add_task(
                _calculate_valuation,
                args=(visibility_filepath,
                      viewpoint,
                      weight,  # user defined, from WEIGHT field in vector
                      args['valuation_function'],
                      valuation_coefficients,  # a, b from args, a dict.
                      max_valuation_radius,
                      viewshed_valuation_path),
                target_path_list=[viewshed_valuation_path],
                dependent_task_list=[viewshed_task],
                task_name=f'calculate_valuation_for_viewshed_{feature_index}')
            valuation_tasks.append(valuation_task)
            valuation_filepaths.append(viewshed_valuation_path)

        feature_index += 1

    # The weighted visible structures raster is a leaf node
    weighted_visible_structures_task = graph.add_task(
        _count_and_weight_visible_structures,
        args=(viewshed_files,
              weights,
              file_registry['clipped_dem'],
              file_registry['n_visible_structures']),
        target_path_list=[file_registry['n_visible_structures']],
        dependent_task_list=sorted(viewshed_tasks),
        task_name='sum_visibility_for_all_structures')

    # If we're not doing valuation, we can still compute visual quality,
    # we'll just use the weighted visible structures raster instead of the
    # sum of the valuation rasters.
    if not do_valuation:
        parent_visual_quality_task = weighted_visible_structures_task
        parent_visual_quality_raster_path = (
            file_registry['n_visible_structures'])
    else:
        parent_visual_quality_task = graph.add_task(
            _sum_valuation_rasters,
            args=(file_registry['clipped_dem'],
                  valuation_filepaths,
                  file_registry['viewshed_value']),
            target_path_list=[file_registry['viewshed_value']],
            dependent_task_list=sorted(valuation_tasks),
            task_name='add_up_valuation_rasters')
        parent_visual_quality_raster_path = file_registry['viewshed_value']

    # visual quality is one of the leaf nodes on the task graph.
    graph.add_task(
        _calculate_visual_quality,
        args=(parent_visual_quality_raster_path,
              intermediate_dir,
              file_registry['viewshed_quality']),
        dependent_task_list=[parent_visual_quality_task],
        target_path_list=[file_registry['viewshed_quality']],
        task_name='calculate_visual_quality'
    )

    LOGGER.info('Waiting for Scenic Quality tasks to complete.')
    graph.join()


def _determine_valid_viewpoints(dem_path, structures_path):
    """Determine which viewpoints are valid and return them.

    A point is considered valid when it meets all of these conditions:

        1. The point must be within the bounding box of the DEM
        2. The point must not overlap a DEM pixel that is nodata
        3. The point must not have the same coordinates as another point

    All invalid points are skipped, and a logger message is written for the
    feature.

    Args:
        dem_path (str): The path to a GDAL-compatible digital elevation model
            raster on disk. The projection must match the projection of the
            ``structures_path`` vector.
        structures_path (str): The path to a GDAL-compatible vector containing
            point geometries and, optionally, a few fields describing
            parameters to the viewshed:

                * 'RADIUS' or 'RADIUS2': How far out from the viewpoint (in m)
                    the viewshed operation is permitted to extend. Default: no
                    limit.
                * 'HEIGHT': The height of the structure (in m). Default: 0.0
                * 'WEIGHT': The numeric weight that this viewshed should be
                    assigned when calculating visual quality. Default: 1.0

    Returns:
        An unsorted list of the valid viewpoints and their metadata. The
        tuples themselves are in the order::

            (viewpoint, radius, weight, height)

        Where

            * ``viewpoint``: a tuple of
                ``(projected x coord, projected y coord``)
            * ``radius``: the maximum radius of the viewshed
            * ``weight``: the weight of the viewshed (for calculating visual
                quality)
            * ``height``: The height of the structure at this point.
    """
    dem_raster_info = pygeoprocessing.get_raster_info(dem_path)
    dem_nodata = dem_raster_info['nodata'][0]
    dem_gt = dem_raster_info['geotransform']
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = (
        dem_raster_info['bounding_box'])

    # Use interleaved coordinates (xmin, ymin, xmax, ymax)
    spatial_index = rtree.index.Index(interleaved=True)

    structures_vector = gdal.OpenEx(structures_path, gdal.OF_VECTOR)
    for structures_layer_index in range(structures_vector.GetLayerCount()):
        structures_layer = structures_vector.GetLayer(structures_layer_index)
        layer_name = structures_layer.GetName()
        LOGGER.info('Layer %s has %s features', layer_name,
                    structures_layer.GetFeatureCount())

        radius_fieldname = None
        fieldnames = set(
            column.GetName() for column in structures_layer.schema)
        possible_radius_fieldnames = set(
            ['RADIUS', 'RADIUS2']).intersection(fieldnames)
        if possible_radius_fieldnames:
            radius_fieldname = possible_radius_fieldnames.pop()

        height_present = False
        height_fieldname = 'HEIGHT'
        if height_fieldname in fieldnames:
            height_present = True

        weight_present = False
        weight_fieldname = 'WEIGHT'
        if weight_fieldname in fieldnames:
            weight_present = True

        last_log_time = time.time()
        n_features_touched = -1
        for point in structures_layer:
            n_features_touched += 1
            if time.time() - last_log_time > 5.0:
                LOGGER.info(
                    ("Checking structures in layer %s, approx. "
                     "%.2f%%complete."), layer_name,
                    100.0 * (n_features_touched /
                             structures_layer.GetFeatureCount()))
                last_log_time = time.time()

            # Coordinates in map units to pass to viewshed algorithm
            geometry = point.GetGeometryRef()
            viewpoint = (geometry.GetX(), geometry.GetY())

            if (not bbox_minx <= viewpoint[0] <= bbox_maxx or
                    not bbox_miny <= viewpoint[1] <= bbox_maxy):
                LOGGER.info(
                    ('Feature %s in layer %s is outside of the DEM bounding '
                     'box. Skipping.'), point.GetFID(), layer_name)
                continue

            max_radius = None
            if radius_fieldname:
                max_radius = math.fabs(point.GetField(radius_fieldname))

            height = 0.0
            if height_present:
                height = math.fabs(point.GetField(height_fieldname))

            weight = 1.0
            if weight_present:
                weight = float(point.GetField(weight_fieldname))

            spatial_index.insert(
                point.GetFID(),
                (viewpoint[0], viewpoint[1], viewpoint[0], viewpoint[1]),
                {'max_radius': max_radius,
                 'weight': weight,
                 'height': height})

    # Now check that the viewpoint isn't over nodata in the DEM.
    valid_structures = {}

    dem_origin_x = dem_gt[0]
    dem_origin_y = dem_gt[3]
    dem_pixelsize_x = dem_raster_info['pixel_size'][0]
    dem_pixelsize_y = dem_raster_info['pixel_size'][1]
    dem_raster = gdal.OpenEx(dem_path, gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(1)
    for block_data in pygeoprocessing.iterblocks((dem_path, 1),
                                                 offset_only=True):
        # Using shapely.geometry.box here so that it'll handle the min/max for
        # us and all we need to define here are the correct coordinates.
        block_geom = shapely.geometry.box(
            dem_origin_x + dem_pixelsize_x * block_data['xoff'],
            dem_origin_y + dem_pixelsize_y * block_data['yoff'],
            dem_origin_x + dem_pixelsize_x * (
                block_data['xoff'] + block_data['win_xsize']),
            dem_origin_y + dem_pixelsize_y * (
                block_data['yoff'] + block_data['win_ysize']))

        intersecting_points = list(spatial_index.intersection(
            block_geom.bounds, objects=True))
        if len(intersecting_points) == 0:
            continue

        dem_block = dem_band.ReadAsArray(**block_data)
        for item in intersecting_points:
            viewpoint = (item.bounds[0], item.bounds[2])
            metadata = item.object
            ix_viewpoint = int(
                (viewpoint[0] - dem_gt[0]) // dem_gt[1]) - block_data['xoff']
            iy_viewpoint = int(
                (viewpoint[1] - dem_gt[3]) // dem_gt[5]) - block_data['yoff']
            if utils.array_equals_nodata(
                    numpy.array(dem_block[iy_viewpoint][ix_viewpoint]),
                    dem_nodata).any():
                LOGGER.info(
                    'Feature %s in layer %s is over nodata; skipping.',
                    point.GetFID(), layer_name)
                continue

            if viewpoint in valid_structures:
                LOGGER.info(
                    ('Feature %s in layer %s is a duplicate viewpoint. '
                     'Skipping.'), point.GetFID(), layer_name)
                continue

            # if we've made it here, the viewpoint is valid.
            valid_structures[viewpoint] = metadata

    # Casting to a list so that taskgraph can pickle the result.
    return list(
        (point, meta['max_radius'], meta['weight'], meta['height'])
        for (point, meta) in valid_structures.items())


def _clip_vector(shape_to_clip_path, binding_shape_path, output_path):
    """Clip one vector by another.

    Uses gdal.Layer.Clip() to clip a vector, where the output Layer
    inherits the projection and fields from the original.

    Args:
        shape_to_clip_path (string): a path to a vector on disk. This is
            the Layer to clip. Must have same spatial reference as
            'binding_shape_path'.
        binding_shape_path (string): a path to a vector on disk. This is
            the Layer to clip to. Must have same spatial reference as
            'shape_to_clip_path'
        output_path (string): a path on disk to write the clipped ESRI
            Shapefile to. Should end with a '.shp' extension.

    Returns:
        ``None``

    """
    driver = gdal.GetDriverByName('ESRI Shapefile')
    if os.path.isfile(output_path):
        driver.Delete(output_path)

    shape_to_clip = gdal.OpenEx(shape_to_clip_path, gdal.OF_VECTOR)
    binding_shape = gdal.OpenEx(binding_shape_path, gdal.OF_VECTOR)

    input_layer = shape_to_clip.GetLayer()
    binding_layer = binding_shape.GetLayer()

    vector = driver.Create(output_path, 0, 0, 0, gdal.GDT_Unknown)
    input_layer_defn = input_layer.GetLayerDefn()
    out_layer = vector.CreateLayer(
        input_layer_defn.GetName(), input_layer.GetSpatialRef())

    input_layer.Clip(binding_layer, out_layer)

    # Add in a check to make sure the intersection didn't come back
    # empty
    if out_layer.GetFeatureCount() == 0:
        raise ValueError(
            'Intersection ERROR: _clip_vector '
            'found no intersection between: file - %s and file - %s.' %
            (shape_to_clip_path, binding_shape_path))

    input_layer = None
    binding_layer = None
    shape_to_clip.FlushCache()
    binding_shape.FlushCache()
    shape_to_clip = None
    binding_shape = None


def _sum_valuation_rasters(dem_path, valuation_filepaths, target_path):
    """Sum up all valuation rasters.

    Args:
        dem_path (string): A path to the DEM. Must perfectly overlap all of
            the rasters in ``valuation_filepaths``.
        valuation_filepaths (list of strings): A list of paths to individual
            valuation rasters. All rasters in this list must overlap
            perfectly.
        target_path (string): The path on disk where the output raster will be
            written. If a file exists at this path, it will be overwritten.

    Returns:
        ``None``

    """
    dem_nodata = pygeoprocessing.get_raster_info(dem_path)['nodata'][0]

    def _sum_rasters(dem, *valuation_rasters):
        valid_dem_pixels = ~utils.array_equals_nodata(dem, dem_nodata)
        raster_sum = numpy.empty(dem.shape, dtype=numpy.float64)
        raster_sum[:] = _VALUATION_NODATA
        raster_sum[valid_dem_pixels] = 0

        for valuation_matrix in valuation_rasters:
            valid_pixels = (
                ~utils.array_equals_nodata(valuation_matrix, _VALUATION_NODATA)
                & valid_dem_pixels)
            raster_sum[valid_pixels] += valuation_matrix[valid_pixels]
        return raster_sum

    pygeoprocessing.raster_calculator(
        [(dem_path, 1)] + [(path, 1) for path in valuation_filepaths],
        _sum_rasters, target_path, gdal.GDT_Float64, _VALUATION_NODATA,
        raster_driver_creation_tuple=FLOAT_GTIFF_CREATION_OPTIONS)


def _calculate_valuation(visibility_path, viewpoint, weight,
                         valuation_method, valuation_coefficients,
                         max_valuation_radius,
                         valuation_raster_path):
    """Calculate valuation with one of the defined methods.

    Args:
        visibility_path (string): The path to a visibility raster for a single
            point. The visibility raster has pixel values of 0, 1, or nodata.
            This raster must be projected in meters.
        viewpoint (tuple): The viewpoint in projected coordinates (x, y) of the
            visibility raster.
        weight (number): The numeric weight of the visibility.
        valuation_method (string): The valuation method to use, one of
            ('linear', 'logarithmic', 'exponential').
        valuation_coefficients (dict): A dictionary mapping string coefficient
            letters to numeric coefficient values. Keys 'a' and 'b' are
            required.
        max_valuation_radius (number): Past this distance (in meters),
            valuation values will be set to 0.
        valuation_raster_path (string): The path to where the valuation raster
            will be saved.

    Returns:
        ``None``

    """
    LOGGER.info('Calculating valuation with %s method. Coefficients: %s',
                valuation_method,
                ' '.join(['%s=%g' % (k, v) for (k, v) in
                          sorted(valuation_coefficients.items())]))

    # All valuation functions use coefficients a, b
    a = valuation_coefficients['a']
    b = valuation_coefficients['b']

    if valuation_method == 'linear':

        def _valuation(distance, visibility):
            valid_pixels = (visibility == 1)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = _VALUATION_NODATA
            valuation[(visibility == 0) | valid_pixels] = 0

            x = distance[valid_pixels]
            valuation[valid_pixels] = (
                (a + b * x) * (weight * visibility[valid_pixels]))
            return valuation

    elif valuation_method == 'logarithmic':

        def _valuation(distance, visibility):
            valid_pixels = (visibility == 1)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = _VALUATION_NODATA
            valuation[(visibility == 0) | valid_pixels] = 0

            # Per Rob, this is the natural log.
            # Also per Rob (and Rich), we'll use log(x+1) because log of values
            # where 0 < x < 1 yields strange results indeed.
            valuation[valid_pixels] = (
                (a + b * numpy.log(distance[valid_pixels] + 1)) * (
                    weight * visibility[valid_pixels]))
            return valuation

    elif valuation_method == 'exponential':

        def _valuation(distance, visibility):
            valid_pixels = (visibility == 1)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = _VALUATION_NODATA
            valuation[(visibility == 0) | valid_pixels] = 0

            valuation[valid_pixels] = (
                (a * numpy.exp(-b * distance[valid_pixels])) * (
                    weight * visibility[valid_pixels]))
            return valuation

    pygeoprocessing.new_raster_from_base(
        visibility_path, valuation_raster_path, gdal.GDT_Float64,
        [_VALUATION_NODATA])

    vis_raster_info = pygeoprocessing.get_raster_info(visibility_path)
    vis_gt = vis_raster_info['geotransform']
    iy_viewpoint = int((viewpoint[1] - vis_gt[3]) / vis_gt[5])
    ix_viewpoint = int((viewpoint[0] - vis_gt[0]) / vis_gt[1])

    # convert the distance transform to meters
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(vis_raster_info['projection_wkt'])
    linear_units = spatial_reference.GetLinearUnits()
    pixel_size_in_m = utils.mean_pixel_size_and_area(
        vis_raster_info['pixel_size'])[0] * linear_units

    valuation_raster = gdal.OpenEx(valuation_raster_path,
                                   gdal.OF_RASTER | gdal.GA_Update)
    valuation_band = valuation_raster.GetRasterBand(1)
    vis_nodata = vis_raster_info['nodata'][0]

    for block_info, vis_block in pygeoprocessing.iterblocks(
            (visibility_path, 1)):
        visibility_value = numpy.empty(vis_block.shape, dtype=numpy.float64)
        visibility_value[:] = _VALUATION_NODATA

        x_coord = numpy.linspace(
            block_info['xoff'],
            block_info['xoff'] + block_info['win_xsize'] - 1,
            block_info['win_xsize'])
        y_coord = numpy.linspace(
            block_info['yoff'],
            block_info['yoff'] + block_info['win_ysize'] - 1,
            block_info['win_ysize'])
        ix_matrix, iy_matrix = numpy.meshgrid(x_coord, y_coord)
        dist_in_m = numpy.hypot(numpy.absolute(ix_matrix - ix_viewpoint),
                                numpy.absolute(iy_matrix - iy_viewpoint),
                                dtype=numpy.float64) * pixel_size_in_m

        valid_distances = (dist_in_m <= max_valuation_radius)
        nodata = utils.array_equals_nodata(vis_block, vis_nodata)
        valid_indexes = (valid_distances & (~nodata))

        visibility_value[valid_indexes] = _valuation(dist_in_m[valid_indexes],
                                                     vis_block[valid_indexes])
        visibility_value[~valid_distances & ~nodata] = 0

        valuation_band.WriteArray(visibility_value,
                                  xoff=block_info['xoff'],
                                  yoff=block_info['yoff'])

    # the 0 means approximate stats are not okay
    valuation_band.ComputeStatistics(0)
    valuation_band = None
    valuation_raster.FlushCache()
    valuation_raster = None


def _clip_and_mask_dem(dem_path, aoi_path, target_path, working_dir):
    """Clip and mask the DEM to the AOI.

    Args:
        dem_path (string): The path to the DEM to use. Must have the same
            projection as the AOI.
        aoi_path (string): The path to the AOI to use. Must have the same
            projection as the DEM.
        target_path (string): The path on disk to where the clipped and masked
            raster will be saved. If a file exists at this location it will be
            overwritten. The raster will have a bounding box matching the
            intersection of the AOI and the DEM's bounding box and a spatial
            reference matching the AOI and the DEM.
        working_dir (string): A path to a directory on disk. A new temporary
            directory will be created within this directory for the storage of
            several working files. This temporary directory will be removed at
            the end of this function.

    Returns:
        ``None``

    """
    temp_dir = tempfile.mkdtemp(dir=working_dir,
                                prefix='clip_dem')

    LOGGER.info('Clipping the DEM to its intersection with the AOI.')
    aoi_vector_info = pygeoprocessing.get_vector_info(aoi_path)
    dem_raster_info = pygeoprocessing.get_raster_info(dem_path)
    mean_pixel_size = (
        abs(dem_raster_info['pixel_size'][0]) +
        abs(dem_raster_info['pixel_size'][1])) / 2.0
    pixel_size = (mean_pixel_size, -mean_pixel_size)

    intersection_bbox = [op(aoi_dim, dem_dim) for (aoi_dim, dem_dim, op) in
                         zip(aoi_vector_info['bounding_box'],
                             dem_raster_info['bounding_box'],
                             [max, max, min, min])]

    clipped_dem_path = os.path.join(temp_dir, 'clipped_dem.tif')
    pygeoprocessing.warp_raster(
        dem_path, pixel_size, clipped_dem_path, 'near',
        target_bb=intersection_bbox)

    LOGGER.info('Masking DEM pixels outside the AOI to nodata')
    aoi_mask_raster_path = os.path.join(temp_dir, 'aoi_mask.tif')
    pygeoprocessing.new_raster_from_base(
        clipped_dem_path, aoi_mask_raster_path, gdal.GDT_Byte,
        [_BYTE_NODATA], [0],
        raster_driver_creation_tuple=BYTE_GTIFF_CREATION_OPTIONS)
    pygeoprocessing.rasterize(aoi_path, aoi_mask_raster_path, [1], None)

    dem_nodata = dem_raster_info['nodata'][0]

    def _mask_op(dem, aoi_mask):
        valid_pixels = (~utils.array_equals_nodata(dem, dem_nodata) &
                        (aoi_mask == 1))
        masked_dem = numpy.empty(dem.shape)
        masked_dem[:] = dem_nodata
        masked_dem[valid_pixels] = dem[valid_pixels]
        return masked_dem

    pygeoprocessing.raster_calculator(
        [(clipped_dem_path, 1), (aoi_mask_raster_path, 1)],
        _mask_op, target_path, gdal.GDT_Float32, dem_nodata,
        raster_driver_creation_tuple=FLOAT_GTIFF_CREATION_OPTIONS)

    shutil.rmtree(temp_dir, ignore_errors=True)


def _count_and_weight_visible_structures(visibility_raster_path_list, weights,
                                         clipped_dem_path, target_path):
    """Count (and weight) the number of visible structures for each pixel.

    Args:
        visibility_raster_path_list (list of strings): A list of strings to
            perfectly overlapping visibility rasters.
        weights (list of numbers): A list of numeric weights to apply to each
            visibility raster. There must be the same number of weights in
            this list as there are elements in visibility_rasters.
        clipped_dem_path (string): String path to the DEM.
        target_path (string): The path to where the output raster is stored.

    Returns:
        ``None``

    """
    LOGGER.info('Summing and weighting %d visibility rasters',
                len(visibility_raster_path_list))
    target_nodata = -1
    dem_raster_info = pygeoprocessing.get_raster_info(clipped_dem_path)
    dem_nodata = dem_raster_info['nodata'][0]

    pygeoprocessing.new_raster_from_base(
        clipped_dem_path, target_path, gdal.GDT_Float32, [target_nodata],
        raster_driver_creation_tuple=FLOAT_GTIFF_CREATION_OPTIONS)

    weighted_sum_visibility_raster = gdal.OpenEx(
        target_path, gdal.OF_RASTER | gdal.GA_Update)
    weighted_sum_visibility_band = (
        weighted_sum_visibility_raster.GetRasterBand(1))

    dem_raster = gdal.OpenEx(clipped_dem_path, gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(1)
    last_log_time = time.time()
    n_visibility_pixels = (
        dem_raster_info['raster_size'][0] * dem_raster_info['raster_size'][1] *
        len(visibility_raster_path_list))
    n_visibility_pixels_touched = 0
    for block_data in pygeoprocessing.iterblocks((clipped_dem_path, 1),
                                                 offset_only=True):
        dem_block = dem_band.ReadAsArray(**block_data)
        valid_mask = ~utils.array_equals_nodata(dem_block, dem_nodata)

        visibility_sum = numpy.empty(dem_block.shape, dtype=numpy.float32)
        visibility_sum[:] = target_nodata
        visibility_sum[valid_mask] = 0

        # Weight and sum the outputs, only opening one raster at a time.
        # Opening rasters one at a time avoids errors about having too many
        # files open at once and also avoids possible out-of-memory errors
        # relative to if we were to open all the incoming rasters at once.
        for vis_raster_path, weight in zip(visibility_raster_path_list,
                                           weights):
            if time.time() - last_log_time > 5.0:
                LOGGER.info(
                    'Weighting and summing approx. %.2f%% complete.',
                    100.0 * (n_visibility_pixels_touched / n_visibility_pixels))
                last_log_time = time.time()

            visibility_raster = gdal.OpenEx(vis_raster_path, gdal.OF_RASTER)
            visibility_band = visibility_raster.GetRasterBand(1)
            visibility_block = visibility_band.ReadAsArray(**block_data)

            visible_mask = (valid_mask & (visibility_block == 1))
            visibility_sum[visible_mask] += (
                visibility_block[visible_mask] * weight)

            visibility_band = None
            visibility_raster = None
            n_visibility_pixels_touched += dem_block.size

        weighted_sum_visibility_band.WriteArray(
            visibility_sum, xoff=block_data['xoff'], yoff=block_data['yoff'])

    weighted_sum_visibility_band.ComputeStatistics(0)
    weighted_sum_visibility_band = None
    weighted_sum_visibility_raster = None

    dem_band = None
    dem_raster = None


def _calculate_visual_quality(source_raster_path, working_dir, target_path):
    """Calculate visual quality based on a raster.

    Visual quality is based on the nearest-rank method for breaking pixel
    values from the source raster into percentiles.

    Args:
        source_raster_path (string): The path to a raster from which
            percentiles should be calculated. Nodata values and pixel values
            of 0 are ignored.
        working_dir (string): A directory where working files can be saved.
            A new temporary directory will be created within. This new
            temporary directory will be removed at the end of the function.
        target_path (string): The path to where the output raster will be
            written.

    Returns:
        ``None``

    """
    # Using the nearest-rank method.
    LOGGER.info('Calculating visual quality')

    raster_info = pygeoprocessing.get_raster_info(source_raster_path)
    raster_nodata = raster_info['nodata'][0]

    temp_dir = tempfile.mkdtemp(dir=working_dir,
                                prefix='visual_quality')

    # phase 1: calculate percentiles from the visible_structures raster
    LOGGER.info('Determining percentiles for %s',
                os.path.basename(source_raster_path))

    def _mask_zeros(valuation_matrix):
        """Assign zeros to nodata, excluding them from percentile calc."""
        valid_mask = ~numpy.isclose(valuation_matrix, 0.0)
        if raster_nodata is not None:
            valid_mask &= ~utils.array_equals_nodata(
                valuation_matrix, raster_nodata)
        visual_quality = numpy.empty(valuation_matrix.shape,
                                     dtype=numpy.float64)
        visual_quality[:] = _VALUATION_NODATA
        visual_quality[valid_mask] = valuation_matrix[valid_mask]
        return visual_quality

    masked_raster_path = os.path.join(temp_dir, 'zeros_masked.tif')
    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _mask_zeros, masked_raster_path,
        gdal.GDT_Float64, _VALUATION_NODATA,
        raster_driver_creation_tuple=FLOAT_GTIFF_CREATION_OPTIONS)

    percentile_values = pygeoprocessing.raster_band_percentile(
        (masked_raster_path, 1), temp_dir, [0., 25., 50., 75.])

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Phase 2: map values to their bins to indicate visual quality.
    percentile_bins = numpy.array(percentile_values)
    LOGGER.info('Mapping percentile breaks %s', percentile_bins)

    def _map_percentiles(valuation_matrix):
        nonzero = (valuation_matrix != 0)
        nodata = utils.array_equals_nodata(valuation_matrix, raster_nodata)
        valid_indexes = (~nodata & nonzero)
        visual_quality = numpy.empty(valuation_matrix.shape,
                                     dtype=numpy.int8)
        visual_quality[:] = _BYTE_NODATA
        visual_quality[~nonzero & ~nodata] = 0
        visual_quality[valid_indexes] = numpy.digitize(
            valuation_matrix[valid_indexes], percentile_bins)
        return visual_quality

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _map_percentiles, target_path,
        gdal.GDT_Byte, _BYTE_NODATA,
        raster_driver_creation_tuple=BYTE_GTIFF_CREATION_OPTIONS)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the ``args[limit_to]`` value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire ``args`` dictionary.

    Returns:
        list of ([invalid key_a, invalid_key_b, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    return validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])
