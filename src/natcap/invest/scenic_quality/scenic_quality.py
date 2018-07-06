"""InVEST Scenic Quality Model."""
import os
import math
import operator
import logging
import time
import tempfile

import numpy
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import taskgraph
import pygeoprocessing

from natcap.invest.scenic_quality.viewshed import viewshed
from .. import utils
from .. import validation

LOGGER = logging.getLogger(__name__)
_NODATA = -99999  # largish negative nodata value.


_OUTPUT_BASE_FILES = {
    'viewshed_value': 'visibility_value.tif',
    'n_visible_structures': 'n_visible_structures.tif',
    'viewshed_quality': 'viewshed_qual.tif',
    'overlap_path': 'vp_overlap.shp',
}

_INTERMEDIATE_BASE_FILES = {
    'aligned_dem_path': 'aligned_dem.tif',
    'aoi_reprojected': 'aoi_reprojected.shp',
    'clipped_dem': 'dem_clipped.tif',
    'structures_clipped': 'structures_clipped.shp',
    'structures_reprojected': 'structures_reprojected.shp',
    'visibility_pattern': 'visibility_{id}.tif',
    'auxilliary_pattern': 'auxilliary_{id}.tif',
    'value_pattern': 'value_{id}.tif',
}


def execute(args):
    """Run the Scenic Quality Model.

    Parameters:
        args['workspace_dir'] (string): (required) output directory for
            intermediate, temporary, and final files.
        args['results_suffix] (string): (optional) string to append to any
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
        args['valuation_function'] (string): (required) The type of economic
            function to use for valuation.  One of "polynomial", "logarithmic",
            or "exponential".
        args['a_coef'] (float): (required) The "a" coefficient for valuation.
        args['b_coef'] (float): (required) The "b" coefficient for valuation.
        args['c_coef'] (float): (optional) The "c" coefficient for valuation.
            Required only for polynomial valuation.
        args['d_coef'] (float): (optional) The "d" coefficient for valuation.
            Required only for polynomial valuation.
        args['max_valuation_radius'] (float): (required) Past this distance
            from the viewpoint, the valuation raster's pixel values will be set
            to 0.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.
    """
    LOGGER.info("Starting Scenic Quality Model")

    valuation_coefficients = {
        'a': float(args['a_coef']),
        'b': float(args['b_coef']),
    }
    if args['valuation_function'].startswith('polynomial'):
        valuation_method = 'polynomial'
        valuation_coefficients['c'] = float(args['c_coef'])
        valuation_coefficients['d'] = float(args['d_coef'])
    elif args['valuation_function'].startswith('logarithmic'):
        valuation_method = 'logarithmic'
    elif args['valuation_function'].startswith('exponential'):
        valuation_method = 'exponential'
    else:
        raise ValueError('Valuation function type %s not recognized' %
                         args['valuation_function'])

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

    max_valuation_radius = float(args['max_valuation_radius'])

    dem_raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
    work_token_dir = os.path.join(intermediate_dir, '_tmp_work_tokens')
    try:
        n_workers = int(args['n_workers'])
    except KeyError:
        n_workers = 0  # Threaded queue management, but same process.
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
    clipped_dem_task.join()
    clipped_viewpoints_task.join()

    # phase 2: calculate viewsheds.
    viewshed_files = []
    viewshed_tasks = []
    valuation_tasks = []
    valuation_filepaths = []
    structures_vector = ogr.Open(file_registry['structures_reprojected'])
    for lyr_index, structures_layer in enumerate(structures_vector):
        layer_name = structures_layer.GetName()
        LOGGER.info('Layer %s has %s features', layer_name,
                    structures_layer.GetFeatureCount())

        for point in structures_layer:
            feature_id = "%s-%s" % (lyr_index, point.GetFID())

            # Coordinates in map units to pass to viewshed algorithm
            geometry = point.GetGeometryRef()
            viewpoint = (geometry.GetX(), geometry.GetY())

            if not _viewpoint_within_raster(viewpoint, args['dem_path']):
                LOGGER.info(
                    ('Feature %s in layer %s is outside of the DEM bounding '
                     'box. Skipping.'), layer_name, point.GetFID())
                continue

            if _viewpoint_over_nodata(viewpoint, args['dem_path']):
                LOGGER.info(
                    'Feature %s in layer %s is over nodata; skipping.',
                    point.GetFID(), layer_name)
                continue

            max_radius = None
            # RADIUS is the suggested value for InVEST Scenic Quality
            # RADIUS2 is for users coming from ArcGIS's viewshed.
            # Assume positive infinity if neither field is provided.
            # Positive infinity is represented in our viewshed by None.
            for fieldname in ('RADIUS', 'RADIUS2'):
                try:
                    max_radius = math.fabs(point.GetField(fieldname))
                    break
                except ValueError:
                    # When this field is not present.
                    pass

            try:
                viewpoint_height = math.fabs(point.GetField('HEIGHT'))
            except ValueError:
                # When height field is not present, assume height of 0.0
                viewpoint_height = 0.0

            try:
                weight = float(point.GetField('WEIGHT'))
            except ValueError:
                # When no weight provided, set scale to 1
                weight = 1.0

            visibility_filepath = file_registry['visibility_pattern'].format(
                id=feature_id)
            viewshed_files.append(visibility_filepath)
            auxilliary_filepath = file_registry['auxilliary_pattern'].format(
                id=feature_id)
            viewshed_task = graph.add_task(
                viewshed,
                args=((file_registry['clipped_dem'], 1),  # DEM
                      viewpoint,
                      visibility_filepath),
                kwargs={'curved_earth': True,  # model always assumes this.
                        'refraction_coeff': float(args['refraction']),
                        'max_distance': max_radius,
                        'viewpoint_height': viewpoint_height,
                        'aux_filepath': auxilliary_filepath},
                target_path_list=[auxilliary_filepath, visibility_filepath],
                dependent_task_list=[clipped_dem_task,
                                     clipped_viewpoints_task],
                task_name='calculate_visibility_%s_%s' % (layer_name,
                                                          point.GetFID()))
            viewshed_tasks.append(viewshed_task)

            # calculate valuation
            viewshed_valuation_path = file_registry['value_pattern'].format(
                id=feature_id)
            valuation_task = graph.add_task(
                _calculate_valuation,
                args=(visibility_filepath,
                      viewpoint,
                      weight,  # user defined, from WEIGHT field in vector
                      valuation_method,
                      valuation_coefficients,  # a, b, c, d from args, a dict
                      max_valuation_radius,
                      viewshed_valuation_path),
                target_path_list=[viewshed_valuation_path],
                dependent_task_list=[viewshed_task],
                task_name='calculate_valuation_for_viewshed_%s' % feature_id)
            valuation_tasks.append(valuation_task)
            valuation_filepaths.append(viewshed_valuation_path)

    # The valuation sum is a leaf node on the graph
    graph.add_task(
        pygeoprocessing.raster_calculator,
        args=([(path, 1) for path in valuation_filepaths],
              _sum_valuation_rasters,
              file_registry['viewshed_value'],
              gdal.GDT_Float32,
              _NODATA),
        target_path_list=[file_registry['viewshed_value']],
        dependent_task_list=sorted(valuation_tasks),
        task_name='add_up_valuation_rasters')

    viewshed_sum_task = graph.add_task(
        _count_visible_structures,
        args=(viewshed_files,
              file_registry['clipped_dem'],
              file_registry['n_visible_structures']),
        target_path_list=[file_registry['n_visible_structures']],
        dependent_task_list=sorted(viewshed_tasks),
        task_name='sum_visibility_for_all_structures')

    # visual quality is one of the leaf nodes on the task graph.
    graph.add_task(
        _calculate_visual_quality,
        args=(file_registry['n_visible_structures'],
              file_registry['viewshed_quality']),
        dependent_task_list=[viewshed_sum_task],
        target_path_list=[file_registry['viewshed_quality']],
        task_name='calculate_visual_quality'
    )

    LOGGER.info('Waiting for Scenic Quality tasks to complete.')
    graph.join()


def _clip_vector(shape_to_clip_path, binding_shape_path, output_path):
    """Clip one vector by another.

    Uses ogr.Layer.Clip() to clip a vector, where the output Layer
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
        Nothing
    """
    if os.path.isfile(output_path):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.DeleteDataSource(output_path)

    shape_to_clip = ogr.Open(shape_to_clip_path)
    binding_shape = ogr.Open(binding_shape_path)

    input_layer = shape_to_clip.GetLayer()
    binding_layer = binding_shape.GetLayer()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector = driver.CreateDataSource(output_path)
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


def _sum_valuation_rasters(*valuation_rasters):
    """Sum up all valuation rasters.

    This is a ``raster_calculator`` local operation.

    Parameters:
        valuation_rasters (list of ``numpy.array``s): The valuation rasters
            to sum.

    Returns:
        A ``numpy.array`` representing the per-pixel sum of the valuation
        matrices.
    """
    return numpy.sum(numpy.stack(valuation_rasters), axis=0)


def _calculate_valuation(visibility_path, viewpoint, weight,
                         valuation_method, valuation_coefficients,
                         max_valuation_radius,
                         valuation_raster_path):
    """Calculate valuation with one of the defined methods.

    Parameters:
        visibility_path (string): The path to a visibility raster for a single
            point.  The visibility raster has pixel values of 0, 1, or nodata.
            This raster must be projected in meters.
        viewpoint (tuple): The viewpoint in projected coordinates of the
            visibility raster.
        weight (number): The numeric weight of the visibility.
        valuation_method (string): The valuation method to use, one of
            ('polynomial', 'logarithmic', 'exponential').
        valuation_coefficients (dict): A dictionary mapping string coefficient
            letters to numeric coefficient values.  For 'logarithmic' and
            'exponential' valuation methods, keys 'a' and 'b' are required.
            For 'polynomial' valuation method, keys 'a', 'b', 'c' and 'd' are
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

    if valuation_method == 'polynomial':
        c = valuation_coefficients['c']
        d = valuation_coefficients['d']

        def _valuation(distance, visibility):
            valid_pixels = (visibility > 0)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = 0

            x = distance[valid_pixels]
            valuation[valid_pixels] = (
                (a+b*x+c*x**2+d*x**3)*(weight*visibility[valid_pixels]))
            return valuation

    elif valuation_method == 'logarithmic':

        def _valuation(distance, visibility):
            valid_pixels = (visibility > 0)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = 0

            # Per Rob, this is the natural log.
            valuation[valid_pixels] = (
                (a+b*numpy.log(distance[valid_pixels]))*(
                    weight*visibility[valid_pixels]))
            return valuation

    elif valuation_method == 'exponential':

        def _valuation(distance, visibility):
            valid_pixels = (visibility > 0)
            valuation = numpy.empty(distance.shape, dtype=numpy.float64)
            valuation[:] = 0

            valuation[valid_pixels] = (
                (a*numpy.exp(-b*distance[valid_pixels])) * (
                    weight*visibility[valid_pixels]))
            return valuation

    pygeoprocessing.new_raster_from_base(
        visibility_path, valuation_raster_path, gdal.GDT_Float64, [_NODATA])

    vis_raster_info = pygeoprocessing.get_raster_info(visibility_path)
    vis_gt = vis_raster_info['geotransform']
    iy_viewpoint = int((viewpoint[1] - vis_gt[3]) / vis_gt[5])
    ix_viewpoint = int((viewpoint[0] - vis_gt[0]) / vis_gt[1])

    # convert the distance transform to meters
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(vis_raster_info['projection'])
    linear_units = spatial_reference.GetLinearUnits()
    pixel_size_in_m = vis_raster_info['mean_pixel_size'] * linear_units

    valuation_raster = gdal.OpenEx(valuation_raster_path,
                                   gdal.OF_RASTER | gdal.GA_Update)
    valuation_band = valuation_raster.GetRasterBand(1)
    vis_nodata = vis_raster_info['nodata'][0]

    for block_info, vis_block in pygeoprocessing.iterblocks(visibility_path):
        valid_pixels = (vis_block != vis_nodata)
        visibility_value = numpy.empty(vis_block.shape, dtype=numpy.float64)
        visibility_value[:] = _NODATA

        x_coord = numpy.linspace(
            block_info['xoff'],
            block_info['xoff'] + block_info['win_xsize'] - 1,
            block_info['win_xsize'])
        y_coord = numpy.linspace(
            block_info['yoff'],
            block_info['yoff'] + block_info['win_ysize'] - 1,
            block_info['win_ysize'])
        ix, iy = numpy.meshgrid(x_coord, y_coord)
        dx = numpy.absolute(ix - ix_viewpoint)
        dy = numpy.absolute(iy - iy_viewpoint)
        dist_in_m = numpy.hypot(dx, dy, dtype=numpy.float64) * pixel_size_in_m

        valid_distances = (dist_in_m <= max_valuation_radius)
        nodata = (vis_block == vis_nodata)
        valid_indexes = (valid_distances & (~nodata))

        visibility_value[valid_indexes] = _valuation(dist_in_m[valid_indexes],
                                                     vis_block[valid_indexes])
        visibility_value[~valid_distances & ~nodata] = 0

        valuation_band.WriteArray(visibility_value,
                                  xoff=block_info['xoff'],
                                  yoff=block_info['yoff'])

    valuation_band = None
    valuation_raster.FlushCache()
    valuation_raster = None

    pygeoprocessing.calculate_raster_stats(valuation_raster_path)


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

    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = dem_raster_info['bounding_box']
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

    if nodata is None:
        return False

    ix_viewpoint = int((viewpoint[0] - dem_gt[0]) / dem_gt[1])
    iy_viewpoint = int((viewpoint[1] - dem_gt[3]) / dem_gt[5])

    value_under_viewpoint = band.ReadAsArray(
        xoff=ix_viewpoint, yoff=iy_viewpoint, win_xsize=1, win_ysize=1)

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
    pixel_size = (dem_raster_info['mean_pixel_size'],
                  dem_raster_info['mean_pixel_size'])

    intersection_bbox = [op(aoi_dim, dem_dim) for (aoi_dim, dem_dim) in
                         zip(aoi_vector_info['bounding_box'],
                             dem_raster_info['bounding_box'],
                             [max, max, min, min])]

    clipped_dem_path = os.path.join(temp_dir, 'clipped_dem.tif')
    pygeoprocessing.warp_raster(
        dem_path, pixel_size, clipped_dem_path, 'nearest',
        target_bb=intersection_bbox)

    aoi_mask_raster_path = os.path.join(temp_dir, 'aoi_mask.tif')
    pygeoprocessing.new_raster_from_base(
        clipped_dem_path, aoi_mask_raster_path, gdal.GDT_Byte, [255], [0])
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
        mask_op, target_path, gdal.GDT_Float32, dem_nodata)

    try:
        shutil.rmtree(temp_dir)
    except OSError:
        LOGGER.exception('Could not remove temp directory %s' % temp_dir)


def _count_visible_structures(visibility_rasters, clipped_dem, target_path):
    LOGGER.info('Summing %d visibility rasters', len(visibility_rasters))
    target_nodata = -1
    pygeoprocessing.new_raster_from_base(clipped_dem, target_path,
                                         gdal.GDT_Int32,
                                         [target_nodata])
    dem_raster_info = pygeoprocessing.get_raster_info(clipped_dem)
    dem_nodata = dem_raster_info['nodata'][0]
    pixels_in_dem = operator.mul(*dem_raster_info['raster_size'])
    pixels_processed = 0.0

    vis_raster_bands = [gdal.OpenEx(vis_path, gdal.OF_RASTER).GetRasterBand(1)
                        for vis_path in visibility_rasters]

    target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    last_log_time = time.time()
    for block_info, dem_matrix in pygeoprocessing.iterblocks(clipped_dem):
        current_time = time.time()
        if current_time - last_log_time > 5.0:
            last_log_time = current_time
            LOGGER.info('Counting visible structures approx. %.2f%% complete',
                        (pixels_processed / pixels_in_dem) * 100.0)
        visibility_sum = numpy.empty((block_info['win_ysize'],
                                      block_info['win_xsize']),
                                     dtype=numpy.int32)

        visibility_sum[:] = target_nodata
        valid_mask = (dem_matrix != dem_nodata)
        visibility_sum[~valid_mask] = target_nodata
        for visibility_band in vis_raster_bands:
            visibility_matrix = visibility_band.ReadAsArray(**block_info)
            visible_mask = ((visibility_matrix == 1) & valid_mask)
            visibility_sum[visible_mask] += visibility_matrix[visible_mask]

        target_band.WriteArray(visibility_sum,
                               xoff=block_info['xoff'],
                               yoff=block_info['yoff'])
        pixels_processed += dem_matrix.size

    target_band = None
    target_raster.FlushCache()
    target_raster = None

    pygeoprocessing.calculate_raster_stats(target_path)


def _calculate_visual_quality(visible_structures_raster, target_path):
    """Calculate the visual quality of a q"""
    LOGGER.info('Calculating visual quality')
    # Using the nearest-rank method.
    n_elements = 0
    value_counts = {}

    raster_nodata = pygeoprocessing.get_raster_info(
        visible_structures_raster)['nodata'][0]

    # phase 1: calculate percentiles from the visible_structures raster
    for _, block in pygeoprocessing.iterblocks(visible_structures_raster):
        valid_pixels = block[block != raster_nodata]
        n_elements += len(valid_pixels)

        for index, counted_values in enumerate(numpy.bincount(valid_pixels)):
            try:
                value_counts[index] += counted_values
            except KeyError:
                value_counts[index] = 0

    # Rather than iterate over a loop of all the elements, we can locate the
    # values of the individual ranks in a more abbreviated fashion to minimize
    # looping (which is slow in python).
    rank_ordinals = [math.ceil(n*n_elements) for n in
                     (0.25, 0.50, 0.75, 1.0)]
    percentile_ranks = [0]
    for rank in rank_ordinals:
        pixels_touched = 0
        for n_structures_visible, n_pixels in sorted(value_counts.items()):
            if pixels_touched < rank <= pixels_touched + n_pixels:
                percentile_ranks.append(n_structures_visible)
                break
            pixels_touched += n_pixels

    # phase 2: use the calculated percentiles to write a new raster
    percentile_ranks = numpy.array(percentile_ranks)
    pygeoprocessing.raster_calculator(
        [(visible_structures_raster, 1)],
        lambda matrix: numpy.digitize(matrix, percentile_ranks, right=True),
        target_path, gdal.GDT_Byte, 255)


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
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        'aoi_path',
        'structure_path',
        'dem_path',
        'refraction',
        'max_valuation_radius',
        'valuation_function',
        'a_coef',
        'b_coef']

    if ('valuation_function' in args and
            args['valuation_function'].lower().startswith('polynomial')):
        required_keys.append('c_coef')
        required_keys.append('d_coef')

    for key in required_keys:
        if limit_to in (None, key):
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ('', None):
                no_value_list.append(key)

    if missing_key_list:
        raise KeyError(*missing_key_list)

    if no_value_list:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    if limit_to in ('valuation_function', None):
        if not args['valuation_function'].startswith(
                ('polynomial', 'logarithmic', 'exponential')):
            validation_error_list.append(
                (['valuation_function'], 'Invalid function'))

    spatial_files = (
        ('dem_path', gdal.OF_RASTER, 'raster',),
        ('aoi_path', gdal.OF_VECTOR, 'vector'),
        ('structure_path', gdal.OF_VECTOR, 'vector'))
    with utils.capture_gdal_logging():
        for key, filetype, filetype_string in spatial_files:
            if key not in args:
                continue
            if args[key] in (None, ''):
                continue

            spatial_file = gdal.OpenEx(args[key], filetype)
            if spatial_file is None:
                validation_error_list.append(
                    ([key], 'Must be a %s' % filetype_string))

    # Verify that the DEM projection is in meters.
    # We don't care about the other spatial inputs, as they'll all be
    # reprojected to the DEM's projection.
    if limit_to in ('dem_path', None):
        # only do this check if we can open the raster.
        with utils.capture_gdal_logging():
            do_spatial_check = False
            if gdal.OpenEx(args['dem_path'], gdal.OF_RASTER) != None:
                do_spatial_check = True
        if do_spatial_check:
            dem_srs = osr.SpatialReference()
            dem_srs.ImportFromWkt(
                pygeoprocessing.get_raster_info(args['dem_path'])['projection'])
            if (abs(dem_srs.GetLinearUnits() - 1.0) > 0.5e7 or
                    not bool(dem_srs.IsProjected())):
                validation_error_list.append(
                    (['dem_path'], 'Must be projected in meters'))

    numeric_keys = [
        'refraction',
        'max_valuation_radius',
        'a_coef',
        'b_coef',
        'c_coef',
        'd_coef',
    ]
    for key in numeric_keys:
        if key not in args or args[key] in ('', None):
            continue

        try:
            float(args[key])
        except (ValueError, TypeError):
            validation_error_list.append(
                ([key], "Must be a number"))
        except Exception:
            LOGGER.exception('Unexpected error when testing for float value')

    return validation_error_list
