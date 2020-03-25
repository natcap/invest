"""InVEST Scenic Quality Model."""
import os
import math
import logging
import tempfile
import shutil
import itertools
import heapq
import struct

import numpy
from osgeo import gdal
from osgeo import osr
import taskgraph
import pygeoprocessing

from natcap.invest.scenic_quality.viewshed import viewshed
from .. import utils
from .. import validation

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
    "model_name": "Unobstructed Views: Scenic Quality Provision",
    "module": __name__,
    "userguide_html": "scenic_quality.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["aoi_path", "structure_path", "dem_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "aoi_path": {
            "name": "Area of Interest",
            "type": "vector",
            "required": True,
            "about": (
                "A GDAL-supported vector file.  This AOI instructs "
                "the model where to clip the input data and the extent "
                "of analysis.  Users will create a polygon feature "
                "layer that defines their area of interest.  The AOI "
                "must intersect the Digital Elevation Model (DEM)."),
        },
        "structure_path": {
            "name": "Features Impacted Scenic Quality",
            "type": "vector",
            "required": True,
            "about": (
                "A GDAL-supported vector file.  The user must specify "
                "a point feature layer that indicates locations of "
                "objects that contribute to negative scenic quality, "
                "such as aquaculture netpens or wave energy "
                "facilities.  In order for the viewshed analysis to "
                "run correctly, the projection of this input must be "
                "consistent with the project of the DEM input."),
        },
        "dem_path": {
            "name": "Digital Elevation Model",
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
                "projection_units": "meters",
            },
            "about": (
                "A GDAL-supported raster file.  An elevation raster "
                "layer is required to conduct viewshed analysis. "
                "Elevation data allows the model to determine areas "
                "within the AOI's land-seascape where point features "
                "contributing to negative scenic quality are visible."),
        },
        "refraction": {
            "name": "Refractivity Coefficient",
            "type": "number",
            "required": True,
            "validation_options": {
                "expression": "(value >= 0) & (value <= 1)",
            },
            "about": (
                "The earth curvature correction option corrects for "
                "the curvature of the earth and refraction of visible "
                "light in air.  Changes in air density curve the light "
                "downward causing an observer to see further and the "
                "earth to appear less curved.  While the magnitude of "
                "this effect varies with atmospheric conditions, a "
                "standard rule of thumb is that refraction of visible "
                "light reduces the apparent curvature of the earth by "
                "one-seventh.  By default, this model corrects for the "
                "curvature of the earth and sets the refractivity "
                "coefficient to 0.13."),
        },
        "do_valuation": {
            "name": "Valuation",
            "type": "boolean",
            "required": False,
            "about": "Enable or disable valuation."
        },
        "valuation_function": {
            "name": "Valuation function",
            "type": "option_string",
            "required": "do_valuation",
            "validation_options": {
                "options": [
                    'linear: a + bx',
                    'logarithmic: a + b log(x+1)',
                    'exponential: a * e^(-bx)'],
            },
            "about": (
                "This field indicates the functional form f(x) the "
                "model will use to value the visual impact for each "
                "viewpoint."),
        },
        "a_coef": {
            "name": "'a' Coefficient",
            "type": "number",
            "required": "do_valuation",
            "about": ("First coefficient used by the valuation function"),
        },
        "b_coef": {
            "name": "'a' Coefficient",
            "type": "number",
            "required": "do_valuation",
            "about": ("Second coefficient used by the valuation function"),
        },
        "max_valuation_radius": {
            "name": "Maximum Valuation Radius",
            "type": "number",
            "required": False,
            "validation_options": {
                "expression": "value > 0",
            },
            "about": (
                "Radius beyond which the valuation is set to zero. "
                "The valuation function 'f' cannot be negative at the "
                "radius 'r' (f(r)>=0)."),
        },
    }
}


def execute(args):
    """Run the Scenic Quality Model.

    Parameters:
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
            valuation.  If ``False``, per-viewpoint value will not be computed,
            and the summation of valuation rasters (vshed_value.tif) will not
            be created.  Additionally, the Viewshed Quality raster will
            represent the weighted sum of viewsheds. Default: ``False``.
        args['valuation_function'] (string): The type of economic
            function to use for valuation.  One of "linear", "logarithmic",
            or "exponential".
        args['a_coef'] (float): The "a" coefficient for valuation.  Required
            if ``args['do_valuation']`` is ``True``.
        args['b_coef'] (float): The "b" coefficient for valuation.  Required
            if ``args['do_valuation']`` is ``True``.
        args['max_valuation_radius'] (float): Past this distance
            from the viewpoint, the valuation raster's pixel values will be set
            to 0.  Required if ``args['do_valuation']`` is ``True``.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
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
        if args['valuation_function'].startswith('linear'):
            valuation_method = 'linear'
        elif args['valuation_function'].startswith('logarithmic'):
            valuation_method = 'logarithmic'
        elif args['valuation_function'].startswith('exponential'):
            valuation_method = 'exponential'
        else:
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
              dem_raster_info['projection'],
              file_registry['aoi_reprojected']),
        target_path_list=[file_registry['aoi_reprojected']],
        task_name='reproject_aoi_to_dem')

    reprojected_viewpoints_task = graph.add_task(
        pygeoprocessing.reproject_vector,
        args=(args['structure_path'],
              dem_raster_info['projection'],
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
    LOGGER.info('Setting up viewshed tasks')
    viewpoint_tuples = []
    structures_vector = gdal.OpenEx(file_registry['structures_reprojected'],
                                    gdal.OF_VECTOR)
    for structures_layer_index in range(structures_vector.GetLayerCount()):
        structures_layer = structures_vector.GetLayer(structures_layer_index)
        layer_name = structures_layer.GetName()
        LOGGER.info('Layer %s has %s features', layer_name,
                    structures_layer.GetFeatureCount())

        for point in structures_layer:
            # Coordinates in map units to pass to viewshed algorithm
            geometry = point.GetGeometryRef()
            viewpoint = (geometry.GetX(), geometry.GetY())

            if not _viewpoint_within_raster(viewpoint, file_registry['clipped_dem']):
                LOGGER.info(
                    ('Feature %s in layer %s is outside of the DEM bounding '
                     'box. Skipping.'), layer_name, point.GetFID())
                continue

            if _viewpoint_over_nodata(viewpoint, file_registry['clipped_dem']):
                LOGGER.info(
                    'Feature %s in layer %s is over nodata; skipping.',
                    point.GetFID(), layer_name)
                continue

            # RADIUS is the suggested value for InVEST Scenic Quality
            # RADIUS2 is for users coming from ArcGIS's viewshed.
            # Assume positive infinity if neither field is provided.
            # Positive infinity is represented in our viewshed by None.
            max_radius = None
            for fieldname in ('RADIUS', 'RADIUS2'):
                try:
                    max_radius = math.fabs(point.GetField(fieldname))
                    break
                except (ValueError, KeyError):
                    # When this field is not present.
                    # ValueError was changed to KeyError between GDAL 2.2 and
                    # 2.4.
                    pass

            try:
                viewpoint_height = math.fabs(point.GetField('HEIGHT'))
            except (ValueError, KeyError):
                # When height field is not present, assume height of 0.0
                # ValueError was changed to KeyError between GDAL 2.2 and 2.4.
                viewpoint_height = 0.0

            try:
                weight = float(point.GetField('WEIGHT'))
            except (ValueError, KeyError):
                # When no weight provided, set scale to 1
                # ValueError was changed to KeyError between GDAL 2.2 and 2.4.
                weight = 1.0

            viewpoint_tuples.append((viewpoint, max_radius, weight,
                                     viewpoint_height))
    structures_vector = None

    if not viewpoint_tuples:
        raise ValueError('No valid viewpoints found. This may happen if '
                         'viewpoints are beyond the edge of the DEM or are '
                         'over nodata pixels.')

    # These are sorted outside the vector to ensure consistent ordering.  This
    # helps avoid unnecesary recomputation in taskgraph for when an ESRI
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
                      valuation_method,
                      valuation_coefficients,  # a, b from args, a dict.
                      max_valuation_radius,
                      viewshed_valuation_path),
                target_path_list=[viewshed_valuation_path],
                dependent_task_list=[viewshed_task],
                task_name='calculate_valuation_for_viewshed_%s' % feature_index)
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


def _clip_vector(shape_to_clip_path, binding_shape_path, output_path):
    """Clip one vector by another.

    Uses gdal.Layer.Clip() to clip a vector, where the output Layer
    inherits the projection and fields from the original.

    Parameters:
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

    Parameters:
        dem_path (string): A path to the DEM.  Must perfectly overlap all of
            the rasters in ``valuation_filepaths``.
        valuation_filepaths (list of strings): A list of paths to individual
            valuation rasters.  All rasters in this list must overlap
            perfectly.
        target_path (string): The path on disk where the output raster will be
            written.  If a file exists at this path, it will be overwritten.

    Returns:
        ``None``

    """
    dem_nodata = pygeoprocessing.get_raster_info(dem_path)['nodata'][0]

    def _sum_rasters(dem, *valuation_rasters):
        valid_dem_pixels = (dem != dem_nodata)
        raster_sum = numpy.empty(dem.shape, dtype=numpy.float64)
        raster_sum[:] = _VALUATION_NODATA
        raster_sum[valid_dem_pixels] = 0

        for valuation_matrix in valuation_rasters:
            valid_pixels = ((valuation_matrix != _VALUATION_NODATA) &
                            valid_dem_pixels)
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

    Parameters:
        visibility_path (string): The path to a visibility raster for a single
            point.  The visibility raster has pixel values of 0, 1, or nodata.
            This raster must be projected in meters.
        viewpoint (tuple): The viewpoint in projected coordinates (x, y) of the
            visibility raster.
        weight (number): The numeric weight of the visibility.
        valuation_method (string): The valuation method to use, one of
            ('linear', 'logarithmic', 'exponential').
        valuation_coefficients (dict): A dictionary mapping string coefficient
            letters to numeric coefficient values.  Keys 'a' and 'b' are
            required.
        max_valuation_radius (number): Past this distance (in meters),
            valuation values will be set to 0.
        valuation_raster_path (string): The path to where the valuation raster
            will be saved.

    Returns:
        ``None``

    """
    valuation_method = valuation_method.lower()
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
                (a+b*x)*(weight*visibility[valid_pixels]))
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
                (a+b*numpy.log(distance[valid_pixels] + 1))*(
                    weight*visibility[valid_pixels]))
            return valuation

    elif valuation_method == 'exponential':

        def _valuation(distance, visibility):
            valid_pixels = (visibility == 1)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = _VALUATION_NODATA
            valuation[(visibility == 0) | valid_pixels] = 0

            valuation[valid_pixels] = (
                (a*numpy.exp(-b*distance[valid_pixels])) * (
                    weight*visibility[valid_pixels]))
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
    spatial_reference.ImportFromWkt(vis_raster_info['projection'])
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
        nodata = (vis_block == vis_nodata)
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


def _viewpoint_within_raster(viewpoint, dem_path):
    """Determine if a viewpoint overlaps a DEM.

    Parameters:
        viewpoint (tuple): A coordinate pair indicating the (x, y) coordinates
            projected in the DEM's coordinate system.
        dem_path (string): The path to a DEM raster on disk.

    Returns:
        ``True`` if the viewpoint overlaps the DEM, ``False`` if not.

    """
    dem_raster_info = pygeoprocessing.get_raster_info(dem_path)

    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = (
        dem_raster_info['bounding_box'])
    if (not bbox_minx <= viewpoint[0] <= bbox_maxx or
            not bbox_miny <= viewpoint[1] <= bbox_maxy):
        return False
    return True


def _viewpoint_over_nodata(viewpoint, dem_path):
    """Determine if a viewpoint overlaps a nodata value within the DEM.

    Parameters:
        viewpoint (tuple): A coordinate pair indicating the (x, y) coordinates
            projected in the DEM's coordinate system.
        dem_path (string): The path to a DEM raster on disk.

    Returns:
        ``True`` if the viewpoint overlaps a nodata value within the DEM,
        ``False`` if not.  If the DEM does not have a nodata value defined,
        returns ``False``.

    """
    raster = gdal.OpenEx(dem_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem_gt = raster.GetGeoTransform()

    ix_viewpoint = int((viewpoint[0] - dem_gt[0]) / dem_gt[1])
    iy_viewpoint = int((viewpoint[1] - dem_gt[3]) / dem_gt[5])

    value_under_viewpoint = band.ReadAsArray(
        xoff=ix_viewpoint, yoff=iy_viewpoint, win_xsize=1, win_ysize=1)

    band = None
    raster = None

    # If the nodata value is not set, ``nodata`` will be None and this should
    # always return False.
    if value_under_viewpoint == nodata:
        return True
    return False


def _clip_and_mask_dem(dem_path, aoi_path, target_path, working_dir):
    """Clip and mask the DEM to the AOI.

    Parameters:
        dem_path (string): The path to the DEM to use.  Must have the same
            projection as the AOI.
        aoi_path (string): The path to the AOI to use.  Must have the same
            projection as the DEM.
        target_path (string): The path on disk to where the clipped and masked
            raster will be saved.  If a file exists at this location it will be
            overwritten.  The raster will have a bounding box matching the
            intersection of the AOI and the DEM's bounding box and a spatial
            reference matching the AOI and the DEM.
        working_dir (string): A path to a directory on disk.  A new temporary
            directory will be created within this directory for the storage of
            several working files.  This temporary directory will be removed at
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
        valid_pixels = ((dem != dem_nodata) &
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

    Parameters:
        visibility_raster_path_list (list of strings): A list of strings to
            perfectly overlapping visibility rasters.
        weights (list of numbers): A list of numeric weights to apply to each
            visibility raster.  There must be the same number of weights in
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

    def _sum_and_weight(*args):
        """Sum and weight the input matrices, masking the output to the DEM.

        Parameters:
            args (list): A list of 2n+1 items, where n is the number of
                visibility rasters to sum and weight.  Item 0 in this list must
                be the DEM array.  Items 1 through n of this list must be the
                visibility arrays.  Items n+1 through 2n of this list must be
                the weights that correspond with the visibility arrays, in
                corresponding order.

        Returns:
            A 2D numpy array for the weighted sum of the visibility rasters.

        """
        dem = args[0]
        n_visibility_arrays = (len(args) - 1) // 2
        visibility_rasters = args[1: n_visibility_arrays + 1]
        weights = args[n_visibility_arrays + 1:]

        valid_mask = (dem != dem_nodata)

        visibility_sum = numpy.empty(dem.shape, dtype=numpy.float32)
        visibility_sum[:] = target_nodata
        visibility_sum[valid_mask] = 0

        # Weight and sum the outputs.
        for visibility_matrix, weight in zip(visibility_rasters, weights):
            visible_mask = (valid_mask & (visibility_matrix == 1))
            visibility_sum[visible_mask] += (visibility_matrix[visible_mask] *
                                             weight)
        return visibility_sum

    pygeoprocessing.raster_calculator(
        ([(clipped_dem_path, 1)] +
         [(vis_path, 1) for vis_path in visibility_raster_path_list] +
         [(weight, 'raw') for weight in weights]),
        _sum_and_weight, target_path, gdal.GDT_Float32, target_nodata,
        raster_driver_creation_tuple=FLOAT_GTIFF_CREATION_OPTIONS)


def _calculate_visual_quality(source_raster_path, working_dir, target_path):
    """Calculate visual quality based on a raster.

    Visual quality is based on the nearest-rank method for breaking pixel
    values from the source raster into percentiles.

    Parameters:
        source_raster_path (string): The path to a raster from which
            percentiles should be calculated.  Nodata values and pixel values
            of 0 are ignored.
        working_dir (string): A directory where working files can be saved.
            A new temporary directory will be created within.  This new
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
        nonzero = ~numpy.isclose(valuation_matrix, 0.0)
        nodata = numpy.isclose(valuation_matrix, raster_nodata)
        valid_indexes = (~nodata & nonzero)
        visual_quality = numpy.empty(valuation_matrix.shape,
                                     dtype=numpy.float64)
        visual_quality[:] = _VALUATION_NODATA
        visual_quality[valid_indexes] = valuation_matrix[valid_indexes]
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
        nodata = (valuation_matrix == raster_nodata)
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

    Parameters:
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
