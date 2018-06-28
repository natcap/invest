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
import natcap.invest.utils
import natcap.invest.reporting

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

    # Check that the Validation container is selected since the UI can not
    val_ordinal = args['valuation_function']
    val_containers = {
        '0': 'polynomial_container', '1': 'log_container',
        '2': 'exponential_container'}

    if args[val_containers[val_ordinal]] == False:
        raise ValuationContainerError(
            'The container for the selected valuation functions '
            'coefficients was not selected.')

    # Create output and intermediate directory
    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    viewshed_dir = os.path.join(intermediate_dir, 'viewpoint_rasters')
    pygeoprocessing.create_directories(
        [output_dir, intermediate_dir, viewshed_dir])

    file_suffix = natcap.invest.utils.make_suffix_string(
        args, 'results_suffix')

    LOGGER.info('Building file registry')
    file_registry = natcap.invest.utils.build_file_registry(
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
                      (geometry.GetX(), geometry.GetY()),  # viewpoint
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


    # Reproject AOI to DEM to clip DEM by AOI.
    pygeoprocessing.reproject_datasource_uri(
        args['aoi_path'], dem_wkt, file_registry['aoi_proj_dem_path'])

    # Clip DEM by AOI
    pygeoprocessing.clip_dataset_uri(
        args['dem_path'], file_registry['aoi_proj_dem_path'],
        file_registry['clipped_dem_path'], assert_projections=False)

    # Reproject AOI to viewpoint structures in order to clip
    structures_srs = pygeoprocessing.get_spatial_ref_uri(
        args['structure_path'])
    structures_wkt = structures_srs.ExportToWkt()
    pygeoprocessing.reproject_datasource_uri(
        args['aoi_path'], structures_wkt,
        file_registry['aoi_proj_struct_path'])

    # Clip viewpoint structures by AOI
    clip_datasource_layer(
        args['structure_path'], file_registry['aoi_proj_struct_path'],
        file_registry['structures_clipped_path'])

    aoi_srs = pygeoprocessing.get_spatial_ref_uri(args['aoi_path'])
    aoi_wkt = aoi_srs.ExportToWkt()
    # Project viewpoint structures to AOI
    pygeoprocessing.reproject_datasource_uri(
        file_registry['structures_clipped_path'], aoi_wkt,
        file_registry['structures_projected_path'])

    pixel_size = projected_pixel_size(
        file_registry['clipped_dem_path'], aoi_srs)

    LOGGER.debug('Projected Pixel Size: %s', pixel_size)

    # Project DEM to AOI
    pygeoprocessing.reproject_dataset_uri(
        file_registry['clipped_dem_path'], pixel_size, aoi_wkt,
        'bilinear', file_registry['dem_proj_to_aoi_path'])

    # Read in valuation coefficients
    val_coeffs = {'0': 'poly', '1': 'log', '2': 'exp'}
    coef_a = float(args['%s_a_coeff' % val_coeffs[val_ordinal]])
    coef_b = float(args['%s_b_coeff' % val_coeffs[val_ordinal]])

    # If 3rd degree polynomial, get the other two coefficients
    if val_ordinal == '0':
        coef_c = float(args['poly_c_coeff'])
        coef_d = float(args['poly_d_coeff'])

    max_val_radius = float(args['max_valuation_radius'])

    def polynomial_val(dist, weight):
        """Third Degree Polynomial Valuation function.

        This represents equation 2 in the User Guide with the added weighted
        factor.

        Parameters:
            dist (numpy.array): pixels are distances to a viewpoint.
            weight (numpy.array): pixels are constant weight values used to
                scale the valuation output

        Returns:
            numpy.array
        """
        # Based off of Equation 2 in the Users Guide
        return numpy.where(
            (dist != nodata) & (weight != nodata),
            ((coef_a + coef_b * dist + coef_c * dist**2 +
                coef_d * dist**3) * weight),
            nodata)

    def log_val(dist, weight):
        """Logarithmic Valuation function.

        This represents equation 1 in the User Guide with the added weighted
        factor.

        Parameters:
            dist (numpy.array): pixels are distances to a viewpoint.
            weight (numpy.array): pixels are constant weight values used to
                scale the valuation output

        Returns:
            numpy.array
        """
        # Based off of Equation 1 in the Users Guide
        return numpy.where(
            (dist != nodata) & (weight != nodata),
            (coef_a + coef_b * numpy.log(dist)) * weight,
            nodata)

    def exp_val(dist, weight):
        """Exponential Decay Valuation function.

        This represents equation X in the User Guide with the added weighted
        factor.

        Parameters:
            dist (numpy.array): pixels are distances to a viewpoint.
            weight (numpy.array): pixels are constant weight values used to
                scale the valuation output

        Returns:
            numpy.array
        """
        return numpy.where(
            (dist != nodata) & (weight != nodata),
            (coef_a * numpy.exp(-coef_b * dist)) * weight,
            nodata)

    def add_op(raster_one, raster_two):
        """Aggregate valuation matrices.

        Sums all non-nodata values.

        Parameters:
            raster_one (numpy.array): values to aggregate with raster_two
            raster_two (numpy.array): values to aggregate with raster_one

        Returns:
            numpy.array where the pixel value represents the combined
                pixel values found across the two matrices.
        """
        raster_one[raster_one == nodata] = 0
        raster_two[raster_two == nodata] = 0
        return raster_one + raster_two

    # Determine which valuation operation to use based on user input
    val_functs = {'0': polynomial_val, '1': log_val, '2': exp_val}
    val_op = val_functs[args["valuation_function"]]

    viewpoints_vector = ogr.Open(file_registry['structures_projected_path'])

    # Get the 'yes' or 'no' responses for whether the individual viewpoint
    # rasters should be kept
    keep_viewsheds = args['keep_feat_viewsheds']
    keep_val_viewsheds = args['keep_val_viewsheds']

    include_curvature = True
    try:
        curvature_correction = float(args['refraction'])
    except ValueError:
        # If refraction is not present, set to 0 and set include curvature
        # to False
        curvature_correction = 0.0
        include_curvature = False

    # An indicator if any previous viewpoints have been calculated
    initial_viewpoint = True

    for layer in viewpoints_vector:
        num_features = layer.GetFeatureCount()

        # Create lists of temporary files for the number of viewpoint features
        # This will be used to aggregate the current viewpoint rasters to the
        # previous ones on the fly.
        feat_val_paths = []
        feat_views_paths = []
        for feat_num in xrange(num_features - 1):
            val_path = pygeoprocessing.temporary_filename()
            feat_val_paths.append(val_path)
            feat_path = pygeoprocessing.temporary_filename()
            feat_views_paths.append(feat_path)

        # The last filenames in the lists should be the aggregated output paths
        feat_val_paths.append(file_registry['viewshed_valuation_path'])
        feat_views_paths.append(file_registry['viewshed_path'])

        for index, point in enumerate(layer):
            geometry = point.GetGeometryRef()
            feature_id = point.GetFID()

            # Coordinates in map units to pass to viewshed algorithm
            geom_x, geom_y = geometry.GetX(), geometry.GetY()

            max_radius = float('inf')
            # RADIUS is the suggested value for InVEST Scenic Quality
            # RADIUS2 is for users coming from ArcGIS's viewshed.
            # Assume positive infinity if neither field is provided.
            for fieldname in ['RADIUS', 'RADIUS2']:
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

            LOGGER.debug(('Processing viewpoint %s of %s (FID %s). '
                          'Radius:%s, Height:%s, Weight:%s'),
                         index, num_features, feature_id, max_radius,
                         viewpoint_height, weight)

            viewshed_filepath = os.path.join(
                viewshed_dir, 'viewshed_%s.tif' % index)

            try:
                pygeoprocessing.viewshed(
                    file_registry['dem_proj_to_aoi_path'], (geom_x, geom_y),
                    viewshed_filepath, None, include_curvature,
                    curvature_correction, max_radius, viewpoint_height)
            except ValueError:
                # When pixel is over nodata and we told it to skip
                LOGGER.info('Viewpoint %s is over nodata, skipping.', index)
                # If this happens we want to continue and re-organize our
                # file lists a bit
                if not initial_viewpoint:
                    if num_features - 1 == index:
                        # Skipping last feature means our final outputs are
                        # the previous aggregated ones. Copy to final file path
                        shutil.copy(feat_val_paths[index - 1],
                                    feat_val_paths[index])
                        shutil.copy(feat_views_paths[index - 1],
                                    feat_views_paths[index])
                        os.remove(feat_vals_paths[index - 1])
                        os.remove(feat_views_paths[index - 1])
                    else:
                        # NOTE: not removing index - 1 files because they
                        # are empty and cleaned up on model exit
                        feat_val_paths[index] = feat_val_paths[index - 1]
                        feat_views_paths[index] = feat_views_paths[index - 1]
                continue

            # Create temporary point shapefile from geometry of feature
            if os.path.isfile(file_registry['single_point_path']):
                driver = ogr.GetDriverByName('ESRI Shapefile')
                driver.DeleteDataSource(file_registry['single_point_path'])

            output_driver = ogr.GetDriverByName('ESRI Shapefile')
            output_datasource = output_driver.CreateDataSource(
                file_registry['single_point_path'])
            layer_name = 'viewpoint'
            output_layer = output_datasource.CreateLayer(
                    layer_name, aoi_srs, ogr.wkbPoint)

            output_field = ogr.FieldDefn('view_ID', ogr.OFTReal)
            output_layer.CreateField(output_field)

            tmp_geom = ogr.Geometry(ogr.wkbPoint)
            tmp_geom.AddPoint_2D(geom_x, geom_y)

            output_feature = ogr.Feature(output_layer.GetLayerDefn())
            output_layer.CreateFeature(output_feature)

            field_index = output_feature.GetFieldIndex('view_ID')
            output_feature.SetField(field_index, 1)

            output_feature.SetGeometryDirectly(tmp_geom)
            output_layer.SetFeature(output_feature)
            output_feature = None
            output_layer = None
            output_datasource = None

            nodata = pygeoprocessing.get_nodata_from_uri(viewshed_filepath)
            cell_size = pygeoprocessing.get_cell_size_from_uri(
                viewshed_filepath)
            weighted_view_path = pygeoprocessing.temporary_filename()

            def weight_factor_op(view):
                """Scale raster by weight value.

                Parameters:
                    view (numpy.array): array to scale

                Returns:
                    numpy.array with view values multiplied by weight
                """
                return numpy.where(view != nodata, view * weight, nodata)

            if weight != 1.0:
                # Multiply viewpoints by the scalar weight
                pygeoprocessing.vectorize_datasets(
                    [viewshed_filepath], weight_factor_op, weighted_view_path,
                    gdal.GDT_Float32, nodata, cell_size, 'intersection',
                    vectorize_op=False)
            else:
                # Weight was not provided or set to 1.0
                shutil.copy(viewshed_filepath, weighted_view_path)

            # Create a new raster and burn viewpoint feature onto it
            burned_feat_path = pygeoprocessing.temporary_filename()
            dist_pixel_path = pygeoprocessing.temporary_filename()

            pygeoprocessing.new_raster_from_base_uri(
                viewshed_filepath, burned_feat_path, 'GTiff', nodata,
                gdal.GDT_Float32, fill_value=0.0)
            pygeoprocessing.rasterize_layer_uri(
                burned_feat_path, file_registry['single_point_path'],
                burn_values=[1.0], option_list=["ALL_TOUCHED=TRUE"])
            # Do a distance transform on viewpoint raster
            pygeoprocessing.distance_transform_edt(
                burned_feat_path, dist_pixel_path, process_pool=None)

            dist_nodata = pygeoprocessing.get_nodata_from_uri(dist_pixel_path)

            def dist_op(dist):
                """Convert pixel distances to meter distances.

                Parameters:
                    dist (numpy.array): array of distances in pixels

                Returns:
                    numpy.array with distances in meters
                """
                valid_mask = (dist != dist_nodata)
                # There will be a pixel of zero distance that represents the
                # viewpoint other distances are calculated from. Set this to
                # 1.0, to avoid valuation function calculations of 0.0
                # CONFIRM this is the correct behavior
                dist_cell_size = numpy.where(
                    dist[valid_mask] != 0.0, dist[valid_mask] * cell_size, 1.0)

                dist_final = numpy.empty(valid_mask.shape)
                dist_final[:] = nodata
                dist_final[valid_mask] = dist_cell_size
                return dist_final

            dist_meters_path = pygeoprocessing.temporary_filename()

            pygeoprocessing.vectorize_datasets(
                [dist_pixel_path], dist_op, dist_meters_path,
                gdal.GDT_Float32, nodata, cell_size, 'intersection',
                vectorize_op=False)

            vshed_val_path = os.path.join(
                viewshed_dir, 'val_viewshed_%s.tif' % index)
            # Run valuation equation on distance raster
            pygeoprocessing.vectorize_datasets(
                [dist_meters_path, weighted_view_path], val_op, vshed_val_path,
                gdal.GDT_Float32, nodata, cell_size, 'intersection',
                vectorize_op=False)

            if initial_viewpoint:
                # First time having computed on a viewpoint, nothing else
                # to aggregate with yet. Copy files into the aggregate list
                initial_viewpoint = False
                shutil.copy(vshed_val_path, feat_val_paths[index])
                shutil.copy(viewshed_filepath, feat_views_paths[index])

            else:
                for file_path, out_list in zip(
                        [vshed_val_path, viewshed_filepath],
                        [feat_val_paths, feat_views_paths]):

                    pygeoprocessing.vectorize_datasets(
                        [file_path, out_list[index - 1]], add_op,
                        out_list[index], gdal.GDT_Float32, nodata, cell_size,
                        'intersection', vectorize_op=False,
                        datasets_are_pre_aligned=True)

                    # No longer need the previous raster tracking accumalation.
                    os.remove(out_list[index - 1])

            tmp_files_remove = [
                dist_pixel_path, dist_meters_path, weighted_view_path,
                burned_feat_path]
            for tmp_file in tmp_files_remove:
                os.remove(tmp_file)
            # Remove temporary viewpoint feature shapefile
            driver = ogr.GetDriverByName('ESRI Shapefile')
            driver.DeleteDataSource(file_registry['single_point_path'])

            if keep_val_viewsheds == 'No':
                os.remove(vshed_val_path)
            if keep_viewsheds == 'No':
                os.remove(viewshed_filepath)

    layer = None
    viewpoints_vector = None


def setup_overlap_id_fields(shapefile_path, id_name):
    """Add field to shapefile with unique values.

    Parameters:
        shapefile_path (string): path to a shapefile on disk.
        id_name (string): string of new field to add.

    Returns:
        Nothing
    """
    shapefile = ogr.Open(shapefile_path, 1)
    layer = shapefile.GetLayer()
    id_field = ogr.FieldDefn(id_name, ogr.OFTInteger)
    layer.CreateField(id_field)

    index = 0

    for feat in layer:
        feat.SetField(id_name, index)
        layer.SetFeature(feat)
        index += 1


def add_percent_overlap(
        overlap_path, key_field, perc_name, pixel_counts, pixel_size):
    """Add overlap percentage of pixels on polygon in a new field.

    Parameters:
        overlap_path (string): path to polygon shapefile on disk.
        key_field (string): the field name for unique feature id's.
        perc_name (string): name of new field to hold percent overlap values.
        pixel_counts (dict): dictionary with keys mapping to 'key_field'
            and values being number of pixels.
        pixel_size (float): cell size for the pixels.

    Returns:
        Nothing
    """
    shapefile = ogr.Open(overlap_path, 1)
    layer = shapefile.GetLayer()
    perc_field = ogr.FieldDefn(perc_name, ogr.OFTReal)
    layer.CreateField(perc_field)

    for feat in layer:
        key = feat.GetFieldAsInteger(key_field)
        geom = feat.GetGeometryRef()
        geom_area = geom.GetArea()
        # Compute overlap by area of pixels and area of polygon
        pixel_area = pixel_size**2 * pixel_counts[key]
        feat.SetField(perc_name, (pixel_area / geom_area) * 100)
        layer.SetFeature(feat)



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
        raise IntersectionError(
            'Intersection ERROR: clip_datasource_layer '
            'found no intersection between: file - %s and file - %s.' %
            (shape_to_clip_path, binding_shape_path))


def _viewpoint_over_nodata(viewpoint, dem_path):
    raster = gdal.OpenEx(dem_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem_gt = raster.GetGeoTransform()

    ix_viewpoint = int((viewpoint[0] - dem_gt[0]) / dem_gt[1])
    iy_viewpoint = int((viewpoint[1] - dem_gt[3]) / dem_gt[5])

    value_under_viewpoint = band.ReadAsArray(
        xoff=ix_viewpoint, yoff=iy_viewpoint, xsize=1, ysize=1)

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
        target_bbox=aoi_vector_info['bounding_box'])


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
