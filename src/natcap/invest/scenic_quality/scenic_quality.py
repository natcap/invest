"""InVEST Scenic Quality Model."""
import os
import sys
import math
import heapq
import bisect
import itertools

import numpy

import shutil
import logging

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import taskgraph
import pygeoprocessing

from natcap.invest.scenic_quality.viewshed import viewshed
from .. import utils
import natcap.invest.reporting
from .. import validation

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(__name__)
_N_WORKERS = 0


class ValuationContainerError(Exception):
    """A custom error message for missing Valuation parameters."""

    pass

_OUTPUT_BASE_FILES = {
    'viewshed_valuation_path': 'vshed.tif',
    'viewshed_path': 'viewshed_counts.tif',
    'viewshed_quality_path': 'vshed_qual.tif',
    'pop_stats_path': 'populationStats.html',
    'pop_stats_table': 'population_stats.csv',
    'overlap_path': 'vp_overlap_final.shp',
    }

_INTERMEDIATE_BASE_FILES = {
    'pop_affected_path': 'affected_population.tif',
    'pop_unaffected_path': 'unaffected_population.tif',
    'aligned_pop_path': 'aligned_pop.tif',
    'aligned_viewshed_path': 'aligned_viewshed.tif',
    'viewshed_no_zeros_path': 'view_no_zeros.tif'
    }

_TMP_BASE_FILES = {
    'aoi_proj_dem_path': 'aoi_proj_to_dem.shp',
    'aoi_proj_pop_path': 'aoi_proj_to_pop.shp',
    'aoi_proj_struct_path': 'aoi_proj_to_struct.shp',
    'structures_clipped_path': 'structures_clipped.shp',
    'structures_projected_path': 'structures_projected.shp',
    'aoi_proj_overlap_path': 'aoi_proj_to_overlap.shp',
    'overlap_clipped_path': 'overlap_clipped.shp',
    'clipped_dem_path': 'dem_clipped.tif',
    'dem_proj_to_aoi_path': 'dem_proj_to_aoi.tif',
    'clipped_pop_path': 'pop_clipped.tif',
    'pop_proj_to_aoi_path': 'pop_proj_to_aoi.tif',
    'single_point_path': 'tmp_viewpoint_path.shp',
    'population_projected': 'population_projected.shp',
    'overlap_projected_path': 'vp_overlap.shp'
    }


def execute(args):
    """Run the Scenic Quality Model.

    Parameters:
        args['workspace_dir'] (string): (required) output directory for
            intermediate, temporary, and final files.
        args['aoi_path'] (string): (required) path to a vector that
            indicates the area over which the model should be run.
        args['structure_path'] (string): (required) path to a point vector
            that has the features for the viewpoints. Optional fields:
            'WEIGHT', 'RADIUS' / 'RADIUS2', 'HEIGHT'
        args['keep_feat_viewsheds'] : a Boolean for whether individual feature
            viewsheds should be saved to disk.
        args['keep_val_viewsheds'] : a Boolean for whether individual feature
            viewsheds that have been adjusted for valuation should be saved
            to disk.
        args['dem_path'] (string): (required) path to a digital elevation model
            raster.
        args['refraction'] (float): (optional) number indicating the refraction
            coefficient to use for calculating curvature of the earth.
        args['population_path'] (string): (optional) path to a raster for
            population data.
        args['overlap_path'] (string): (optional) path to a polygon shapefile.
        args['results_suffix] (string): (optional) string to append to any
            output file.
        args['valuation_function'] (int): type of economic function to use
            for valuation. Either 3rd degree polynomial (0), logarithmic (1) or
            exponential decay (2).
        args['poly_a_coef'] (float): 1st coefficient for polynomial function.
        args['poly_b_coef'] (float): 2nd coefficient for polynomial function.
        args['poly_c_coef'] (float): 3rd coefficient for polynomial function.
        args['poly_d_coef'] (float): 4th coefficient for polynomial function.
        args['log_a_coef'] (float): 1st coefficient for logarithmic function.
        args['log_b_coef'] (float): 2nd coefficient for logarithmic function.
        args['exp_a_coef'] (float): 1st coefficient for exponential function.
        args['exp_b_coef'] (float): 2nd coefficient for exponential function.
        args['max_valuation_radius'] (float): (required) distance in
            meters for maximum radius of valuation calculations.
    """
    LOGGER.info("Start Scenic Quality Model")

    if args['valuation_function'].startswith('polynomial'):
        valuation_method = 'polynomial'
        valuation_coefficients = {
            'a': args['a_coef'],
            'b': args['b_coef'],
            'c': args['c_coef'],
            'd': args['d_coef'],
        }
    elif args['valuation_function'].startswith('logarithmic'):
        valuation_method = 'logarithmic'
        valuation_coefficients = {
            'a': args['a_coef'],
            'b': args['b_coef'],
        }
    elif args['valuation_function'].startswith('exponential'):
        valuation_method = 'exponential'
        valuation_coefficients = {
            'a': args['a_coef'],
            'b': args['b_coef'],
        }
    else:
        raise ValueError('Valuation function type %s not recognized' %
                         args['valuation_function'])

    # Create output and intermediate directory
    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    viewshed_dir = os.path.join(intermediate_dir, 'viewpoint_rasters')
    utils.make_directories([output_dir, intermediate_dir, viewshed_dir])

    file_suffix = utils.make_suffix_string(
        args, 'results_suffix')

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    dem_raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
    aoi_vector_info = pygeoprocessing.get_vector_info(args['aoi_path'])
    work_token_dir = os.path.join(intermediate_dir, '_tmp_work_tokens')
    graph = taskgraph.TaskGraph(work_token_dir, _N_WORKERS)

    reprojected_aoi_task = graph.add_task(
        pygeoprocessing.reproject_vector,
        args=(args['aoi_path'],
              dem_raster_info['projection'],
              file_registry['aoi_proj_dem_path']),
        target_path_list=[file_registry['aoi_proj_dem_path']],
        task_name='reproject_aoi_to_dem')

    # TODO: clip the AOI to the DEM bounding box.

    reprojected_viewpoints_task = graph.add_task(
        pygeoprocessing.reproject_vector,
        args=(args['structure_path'],
              dem_raster_info['projection'],
              file_registry['structures_projected_path']),
        target_path_list=[file_registry['structures_projected_path']],
        task_name='reproject_structures_to_dem')

    clipped_viewpoints_task = graph.add_task(
        clip_datasource_layer,
        args=(file_registry['structures_projected_path'],
              file_registry['aoi_proj_dem_path'],
              file_registry['structures_clipped_path']),
        target_path_list=[file_registry['structures_clipped_path']],
        dependent_task_list=[reprojected_aoi_task,
                             reprojected_viewpoints_task],
        task_name='clip_reprojected_structures_to_aoi')

    clipped_dem_task = graph.add_task(
        _clip_dem,
        args=(args['dem_path'],
              file_registry['aoi_proj_dem_path'],
              file_registry['clipped_dem_path']),
        target_path_list=[file_registry['clipped_dem_path']],
        dependent_task_list=[reprojected_aoi_task],
        task_name='clip_dem_to_aoi')

    # viewshed calculation requires that the DEM and structures are all
    # finished.
    graph.join()

    # phase 2: calculate viewsheds.
    viewshed_files = []
    viewshed_tasks = []
    valuation_tasks = []
    valuation_filepaths = []
    structures_vector = ogr.Open(file_registry['structures_projected_path'])
    for structures_layer in structures_vector:
        layer_name = structures_layer.GetName()

        for point in structures_layer:
            feature_id = "%s_%s" % (layer_name, point.GetFID())

            # Coordinates in map units to pass to viewshed algorithm
            geometry = point.GetGeometryRef()
            viewpoint = (geometry.GetX(), geometry.GetY())

            if _viewpoint_over_nodata(viewpoint, args['dem_path']):
                LOGGER.info(
                    'Feature %s in layer %s is over nodata; skipping.',
                    layer_name, point.GetFID())
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
                weight = point.GetField('WEIGHT')
            except ValueError:
                # When no weight provided, set scale to 1
                weight = 1.0

            visibility_filepath = os.path.join(
                intermediate_dir, 'visibility_%s_%s%s.tif' % (
                    layer_name, point.GetFID(), file_suffix))
            viewshed_files.append(visibility_filepath)
            auxilliary_filepath = os.path.join(
                intermediate_dir, 'auxilliary_%s_%s%s.tif' % (
                    layer_name, point.GetFID(), file_suffix))

            viewshed_task = graph.add_task(
                viewshed,
                args=(file_registry['clipped_dem_path'],  # DEM
                      viewpoint,
                      visibility_filepath),
                kwargs={'curved_earth': True,  # model always assumes this.
                        'refraction_coeff': args['refraction'],
                        'max_distance': max_radius,
                        'aux_filepath': auxilliary_filepath},
                target_path_list=[auxilliary_filepath, visibility_filepath],
                dependent_task_list=[clipped_dem_task,
                                     clipped_viewpoints_task],
                task_name='calculate_visibility_%s_%s' % (layer_name,
                                                          point.GetFID()))
            viewshed_tasks.append(viewshed_task)

            # create the distance transform
            # TODO: write this path to the registry
            distance_transform_base_path = os.path.join(
                intermediate_dir,
                'distance_transform_base_%s%s.tif' % (feature_id, file_suffix))
            distance_transform_path = os.path.join(
                intermediate_dir,
                'distance_transform_%s%s.tif' % (feature_id, file_suffix))
            distance_to_viewpoint_path = os.path.join(
                intermediate_dir,
                'distance_to_viewpoint_%s%s.tif' % (feature_id, file_suffix))
            distance_transform_task = graph.add_task(
                _calculate_distance_from_viewpoint,
                args=(file_registry['clipped_dem_path'],
                      viewpoint,
                      distance_transform_base_path,
                      distance_transform_path,
                      distance_to_viewpoint_path),
                target_path_list=[distance_transform_base_path,
                                  distance_to_viewpoint_path],
                dependent_task_list=[clipped_dem_task],
                task_name='calculate_distance_to_viewpoint_%s' % feature_id)

            # calculate valuation
            viewshed_valuation_path = os.path.join(
                intermediate_dir,
                'val_viewshed_%s%s.tif' % (feature_id, file_suffix))
            valuation_task = graph.add_task(
                _calculate_valuation,
                args=(distance_to_viewpoint_path,
                      visibility_filepath,
                      weight,  # user defined, from WEIGHT field in vector
                      valuation_method,
                      valuation_coefficients,  # a, b, c, d from args, a dict
                      viewshed_valuation_path),
                target_path_list=[viewshed_valuation_path],
                dependent_task_list=[distance_transform_task,
                                     viewshed_task],
                task_name='calculate_valuation_for_viewshed_%s' % feature_id)
            valuation_tasks.append(valuation_task)
            valuation_filepaths.append(viewshed_valuation_path)

    # The valuation sum is a leaf node on the graph
    graph.add_task(
        pygeoprocessing.raster_calculator,
        args=([(path, 1) for path in valuation_filepaths],
              _sum_valuation_rasters,
              file_registry['viewshed_valuation_path'],
              gdal.GDT_Float32,
              -9999),  # TODO: make this a module-level variable?
        target_path_list=[file_registry['viewshed_valuation_path']],
        dependent_task_list=valuation_tasks,
        task_name='add_up_valuation_rasters')

    viewshed_sum_task = graph.add_task(
        _count_visible_structures,
        args=(viewshed_files,
              file_registry['clipped_dem_path'],
              file_registry['viewshed_counts']),
        target_path_list=[file_registry['viewshed_counts']],
        dependent_task_list=viewshed_tasks,
        task_name='sum_visibility_for_all_structures')

    # visual quality is one of the leaf nodes on the task graph.
    graph.add_task(
        _calculate_visual_quality,
        args=(file_registry['viewshed_counts'],
              file_registry['viewshed_quality_path']),
        dependent_task_list=[viewshed_sum_task],
        target_path_list=[file_registry['viewshed_quality_path']],
        task_name='calculate_visual_quality'
    )

    if 'population_path' in args and args['population_path'] not in (None, ''):
        population_raster_info = pygeoprocessing.get_raster_info(
            args['population_path'])
        target_bbox = pygeoprocessing.transform_bounding_box(
            population_raster_info['bounding_box'],
            population_raster_info['projection'],
            aoi_vector_info['projection'])
        reprojected_clipped_population_task = graph.add_task(
            pygeoprocessing.warp_raster,
            args=(args['population_path'],
                  population_raster_info['pixel_size'],
                  file_registry['population_projected'],
                  'nearest',
                  target_bbox,
                  aoi_vector_info['projection']),
            target_path_list=[file_registry['population_projected']],
            task_name='reprojected_clipped_population_task')

        affected_population_summary_task = graph.add_task(
            _summarize_affected_populations,
            args=(file_registry['population_projected'],
                  file_registry['viewshed_counts'],
                  file_registry['pop_stats_table']),
            target_path_list=[file_registry['pop_stats_table']],
            task_name='affected_population_summary_task',
            dependent_task_list=[reprojected_clipped_population_task,
                                 viewshed_sum_task])

    if 'overlap_path' in args and args['overlap_path'] not in (None, ''):
        # reproject overlap layer to DEM
        # clip by overlap vector by AOI vector
        # count the number of pixels greater than zero under the vector.
        # Create a vector (copied from the overlap vector) with a new field
        # called "%_overlap" that contains for each polygon:
        #    n_pixels under the polygon*(pixel_size**2)/geometry_area*100
        reprojected_overlap_vector_task = graph.add_task(
            pygeoprocessing.reproject_vector,
            args=(args['overlap_path'],
                  dem_raster_info['projection'],
                  file_registry['overlap_projected_path']),
            target_path_list=[file_registry['overlap_projected_path']],
            dependent_task_list=[viewshed_sum_task],
            task_name='reprojected_overlap_vector_task')

        clipped_overlap_vector_task = graph.add_task(
            clip_datasource_layer,
            args=(file_registry['overlap_projected_path'],
                  file_registry['aoi_proj_dem_path'],
                  file_registry['overlap_clipped_path']),
            target_path_list=[file_registry['overlap_clipped_path']],
            dependent_task_list=[reprojected_overlap_vector_task],
            task_name='clipped_overlap_vector_task')

        # convert zero-values to nodata for correct summing.
        mask_out_zero_values_task = graph.add_task(
            _mask_out_zero_values,
            args=(file_registry['viewshed_counts'],
                  file_registry['viewshed_no_zeros_path']),
            target_path_list=[file_registry['viewshed_no_zeros_path']],
            dependent_task_list=[viewshed_sum_task],
            task_name='mask_out_zero_values_task')

        # Calculating percent overlap is a leaf node on the graph.
        graph.add_task(
            _calculate_percent_overlap,
            args=(file_registry['overlap_clipped_path'],
                  file_registry['viewshed_no_zeros_path'],
                  file_registry['overlap_path']),
            target_path_list=[file_registry['overlap_path']],
            dependent_task_list=[mask_out_zero_values_task,
                                 clipped_overlap_vector_task],
            task_name='calculate_percent_overlap_task')


def clip_datasource_layer(shape_to_clip_path, binding_shape_path, output_path):
    """Clip Shapefile Layer by second Shapefile Layer.

    Uses ogr.Layer.Clip() to clip a Shapefile, where the output Layer
    inherits the projection and fields from the original Shapefile.

    Parameters:
        shape_to_clip_path (string): a path to a Shapefile on disk. This is
            the Layer to clip. Must have same spatial reference as
            'binding_shape_path'.
        binding_shape_path (string): a path to a Shapefile on disk. This is
            the Layer to clip to. Must have same spatial reference as
            'shape_to_clip_path'
        output_path (string): a path on disk to write the clipped Shapefile
            to. Should end with a '.shp' extension.

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
    ds = driver.CreateDataSource(output_path)
    input_layer_defn = input_layer.GetLayerDefn()
    out_layer = ds.CreateLayer(
        input_layer_defn.GetName(), input_layer.GetSpatialRef())

    input_layer.Clip(binding_layer, out_layer)

    # Add in a check to make sure the intersection didn't come back
    # empty
    if(out_layer.GetFeatureCount() == 0):
        raise ValueError(
            'Intersection ERROR: clip_datasource_layer '
            'found no intersection between: file - %s and file - %s.' %
            (shape_to_clip_path, binding_shape_path))


def _sum_valuation_rasters(*valuation_rasters):
    return numpy.sum(numpy.stack(valuation_rasters))


def _calculate_distance_from_viewpoint(dem_path, viewpoint,
                                       distance_transform_base_path,
                                       distance_transform_path,
                                       distance_to_viewpoint_path):
    dem_raster_info = pygeoprocessing.get_raster_info(dem_path)
    dem_gt = dem_raster_info['geotransform']

    # Create a new raster with the viewpoint as the one pixel to calculate
    # distance from in the Euclidean Distance Transform.
    pygeoprocessing.new_raster_from_base(
        dem_path, distance_transform_base_path, gdal.GDT_Byte, [255], [0])

    iy_viewpoint = int((viewpoint[1] - dem_gt[3]) / dem_gt[5])
    ix_viewpoint = int((viewpoint[0] - dem_gt[0]) / dem_gt[1])

    edt_raster = gdal.OpenEx(distance_transform_base_path,
                             gdal.OF_RASTER | gdal.GA_Update)
    edt_band = edt_raster.GetRasterBand(1)
    edt_band.WriteArray(numpy.array([[1]]),
                        xoff=ix_viewpoint,
                        yoff=iy_viewpoint)

    edt_band = None
    edt_raster = None

    pygeoprocessing.distance_transform_edt((distance_transform_base_path, 1),
                                           distance_transform_path)

    # convert the distance transform to meters
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(dem_raster_info['projection'])
    linear_units = spatial_reference.GetLinearUnits()
    pixel_size = dem_raster_info['mean_pixel_size']
    dem_nodata = dem_raster_info['nodata'][0]
    distance_nodata = -9999

    def _convert_to_meters(dem, distance):
        valid_pixels = (dem != dem_nodata)
        converted_matrix = numpy.empty(dem.shape, dtype=numpy.float32)
        converted_matrix[:] = distance_nodata

        converted_matrix[valid_pixels] = (
            distance[valid_pixels]*linear_units*pixel_size)
        return converted_matrix

    pygeoprocessing.raster_calculator(
        [(dem_path, 1), (distance_transform_path, 1)],
        _convert_to_meters,
        distance_to_viewpoint_path,
        gdal.GDT_Float32,
        distance_nodata)


def _calculate_valuation(distance_to_viewpoint_path, visibility_path, weight,
                         valuation_method, valuation_coefficients,
                         valuation_raster_path):
    valuation_method = valuation_method.lower()
    # TODO: make these operations nodata-aware (based on the DEM)
    valuation_nodata = -99999

    # All valuation functions use coefficients a, b
    a = valuation_coefficients['a']
    b = valuation_coefficients['b']

    if valuation_method == 'polynomial':
        c = valuation_coefficients['c']
        d = valuation_coefficients['d']

        def _valuation(distance, visibility):
            valid_pixels = (visibility > 0)
            valuation = numpy.empty(distance.shape, dtype=numpy.float32)
            valuation[:] = 0

            valuation[valid_pixels] = (
                (a+b*distance+c*distance**2+d*distance**3)*(weight*visibility))
            return valuation

    elif valuation_method == 'logarithmic':

        def _valuation(distance, visibility):
            valid_pixels = (visibility > 0)
            valuation = numpy.empty(distance.shape, dtype=numpy.float32)
            valuation[:] = 0

            valuation[valid_pixels] = (
                (a+b*numpy.log(distance))*(weight*visibility))
            return valuation

    elif valuation_method == 'exponential':

        def _valuation(distance, visibility):
            valid_pixels = (visibility > 0)
            valuation = numpy.empty(distance.shape, dtype=numpy.float32)
            valuation[:] = 0

            valuation[valid_pixels] = (
                (a*numpy.exp(-b*distance)) * weight*visibility)
            return valuation

    pygeoprocessing.raster_calculator(
        [(distance_to_viewpoint_path, 1), (visibility_path, 1)],
        _valuation,
        valuation_raster_path,
        gdal.GDT_Float32,
        valuation_nodata)




def _viewpoint_over_nodata(viewpoint, dem_path):
    raster = gdal.OpenEx(dem_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem_gt = raster.GetGeoTransform()

    ix_viewpoint = int((viewpoint[0] - dem_gt[0]) / dem_gt[1])
    iy_viewpoint = int((viewpoint[1] - dem_gt[3]) / dem_gt[5])

    value_under_viewpoint = band.ReadAsArray(
        xoff=ix_viewpoint, yoff=iy_viewpoint, win_xsize=1, win_ysize=1)

    if value_under_viewpoint == nodata:
        return True
    return False


def _calculate_percent_overlap(overlap_vector, viewshed_raster, target_path):
    # viewshed_raster must not have any zero-pixel values.

    in_vector = gdal.Open(overlap_vector, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    out_vector = driver.CreateCopy(target_path, in_vector)

    # for now, let's assume that the InVESTID field doesn't exist.
    field_name = 'InVESTID'
    id_field = ogr.FieldDefn(field_name, ogr.OFTInteger)
    layer = out_vector.GetLayer()
    layer.CreateField(id_field)

    for index, feature in enumerate(layer):
        feature.setField(id_field, index)
        layer.setFeature(feature)

    layer = None
    out_vector = None

    raster_stats = pygeoprocessing.zonal_statistics(
        (viewshed_raster, 1), target_path, field_name)

    # having calculated the raster stats, create a new column and add the
    # raster stats.
    viewshed_info = pygeoprocessing.get_raster_info(viewshed_raster)
    pixel_area = viewshed_info['mean_pixel_size']**2
    vector = gdal.Open(target_path, gdal.OF_VECTOR | gdal.GA_Update)
    layer = vector.GetLayer()
    perc_overlap_fieldname = '%_overlap'
    layer.CreateField(ogr.FieldDefn(perc_overlap_fieldname, ogr.OFTReal))
    for feature in layer:
        feature_id = feature.GetField(id_field)
        geometry = feature.GetGeometryRef()
        geom_area = geometry.GetArea()
        n_pixels_overlapping_polygon = raster_stats[feature_id]['count']
        percent_overlap = (
            ((pixel_area*n_pixels_overlapping_polygon)/geom_area)*100.0)
        feature.SetField(perc_overlap_fieldname, percent_overlap)
        layer.SetFeature(feature)


def _mask_out_zero_values(viewshed_sum, target_raster_path):
    viewshed_nodata = pygeoprocessing.get_raster_info(viewshed_sum)

    def _mask_out_zeros(viewshed):
        viewshed[viewshed==0] = viewshed_nodata
        return viewshed

    pygeoprocessing.raster_calculator(
        [viewshed_sum], _mask_out_zeros, target_raster_path,
        gdal.GDT_Int32, viewshed_nodata)


def _clip_dem(dem_path, aoi_path, target_path):
    # invariate: dem and aoi have the same projection.
    aoi_vector_info = pygeoprocessing.get_vector_info(aoi_path)
    dem_raster_info = pygeoprocessing.get_raster_info(dem_path)
    pixel_size = (dem_raster_info['mean_pixel_size'],
                  dem_raster_info['mean_pixel_size'])
    pygeoprocessing.warp_raster(
        dem_path, pixel_size, target_path, 'nearest',
        target_bb=aoi_vector_info['bounding_box'])


def _count_visible_structures(visibility_rasters, clipped_dem, target_path):
    target_nodata = -1
    pygeoprocessing.new_raster_from_base(clipped_dem, target_path,
                                         [target_nodata])
    dem_nodata = pygeoprocessing.get_raster_info(clipped_dem)['nodata'][0]

    target_raster = gdal.Open(target_path)
    target_band = target_raster.GetRasterBand(1)
    for block_info, dem_matrix in pygeoprocessing.iterblocks(clipped_dem,
                                                             offset_only=True):
        visibility_sum = numpy.empty((block_info['win_ysize'],
                                      block_info['win_xsize']),
                                     dtype=numpy.int32)
        valid_mask = (dem_matrix != dem_nodata)
        visibility_sum[~valid_mask] = target_nodata
        for visibility_path in visibility_rasters:
            visibility_raster = gdal.Open(visibility_path)
            visibility_band = visibility_raster.GetRasterBand(1)
            visibility_matrix = visibility_band.ReadAsArray(**block_info)
            visibility_sum[valid_mask] += visibility_matrix[valid_mask]

        target_band.WriteArray(buffer, xoff=block_info['xoff'],
                               yoff=block_info['yoff'])
    target_band = None
    target_raster = None


def _calculate_visual_quality(visible_structures_raster, target_path):
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
    pygeoprocessing.raster_calculator(
        [visible_structures_raster],
        lambda matrix: bisect.bisect(percentile_ranks, matrix),
        target_path, gdal.GDT_Byte, 255)


def _summarize_affected_populations(population_path, viewshed_sum_path,
                                    target_table_path):
    population_nodata = pygeoprocessing.get_raster_info(
        population_path)['nodata']
    n_visible_nodata = pygeoprocessing.get_raster_info(
        viewshed_sum_path)['nodata']

    unaffected_sum = 0
    unaffected_count = 0
    affected_sum = 0
    affected_count = 0
    for (_, population), (_, n_visible) in itertools.izip(
            pygeoprocessing.iterblocks(population_path),
            pygeoprocessing.iterblocks(viewshed_sum_path)):
        valid_mask = ((population != population_nodata) &
                      (n_visible != n_visible_nodata))
        affected_pixels = population[n_visible[valid_mask] > 0]
        affected_sum += numpy.sum(affected_pixels)
        affected_count += len(affected_pixels)

        unaffected_pixels = population[n_visible[valid_mask] == 0]
        unaffected_sum += numpy.sum(unaffected_pixels)
        unaffected_count += len(unaffected_pixels)

    # TODO: adjust for when population is a density raster.
    # TODO: adjust by the cell size.

    with open(target_table_path, 'w') as table_file:
        table_file.write('"# of features visible","Population (estimate)"\n')
        table_file.write('"None visible",%s\n' % unaffected_sum)
        table_file.write('"1 or more visible",%s\n' % affected_sum)


@validation.invest_validator
def validate(args, limit_to=None):
    return []
