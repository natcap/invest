"""InVEST Scenic Quality Model."""
import os
import sys
import math
import heapq

import numpy
from bisect import bisect

import shutil
import logging

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import pygeoprocessing
import natcap.invest.utils
import natcap.invest.reporting

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.scenic_quality.scenic_quality')


class ValuationContainerError(Exception):
    """A custom error message for missing Valuation parameters."""

    pass

_OUTPUT_BASE_FILES = {
    'viewshed_valuation_path': 'vshed.tif',
    'viewshed_path': 'viewshed_counts.tif',
    'viewshed_quality_path': 'vshed_qual.tif',
    'pop_stats_path': 'populationStats.html',
    'overlap_projected_path': 'vp_overlap.shp'
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
    'single_point_path': 'tmp_viewpoint_path.shp'
    }


def execute(args):
    """Run the Scenic Quality Model.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['aoi_path'] (string): path to a vector that indicates the area
            over which the model should be run.
        args['structure_path'] (string): path to a point vector that has
            the features for the viewpoints.
        args['keep_feat_viewsheds'] : a Boolean for whether individual feature
            viewsheds should be saved to disk.
        args['keep_val_viewsheds'] : a Boolean for whether individual feature
            viewsheds that have been adjusted for valuation should be saved
            to disk.
        args['dem_path'] (string): path to a digital elevation model raster.
        args['refraction'] (float): (optional) number indicating the refraction
            coefficient to use for calculating curvature of the earth.
        args['population_path'] (string): (optional) path to a raster for
            population
        args['overlap_path'] (string): (optional)
        args['results_suffix] (string): (optional) string to append to any
            output files
        args['valuation_function'] (int): type of economic function to use
            for valuation. Either 3rd degree polynomial or logarithmic.
        args['poly_a_coef'] (float):
        args['poly_b_coef'] (float):
        args['poly_c_coef'] (float):
        args['poly_d_coef'] (float):
        args['log_a_coef'] (float):
        args['log_b_coef'] (float):
        args['exp_a_coef'] (float):
        args['exp_b_coef'] (float):
        args['max_valuation_radius'] (float):

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

    dem_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(args['dem_path'])
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

    # Do quantiles on viewshed_uri
    percentile_list = [25, 50, 75, 100]

    # Set 0 values to nodata before calculating percentiles, since 0 values
    # indicate there was no viewpoint effects

    def zero_to_nodata(view):
        """Mask 0 values to nodata."""
        return numpy.where(view == 0., nodata, view)

    pygeoprocessing.vectorize_datasets(
        [file_registry['viewshed_valuation_path']], zero_to_nodata,
        file_registry['viewshed_no_zeros_path'], gdal.GDT_Int32, nodata,
        cell_size, 'intersection', assert_datasets_projected=False,
        vectorize_op=False)

    def raster_percentile(band):
        """Operation to use in vectorize_datasets.

        Takes the pixels of 'band' and groups them together based on
            their percentile ranges.
        Parameters:
            band (numpy.array): A gdal raster band
        Returns:
            An integer that places each pixel into a group
        """
        return bisect(percentiles, band)

    # Get the percentile values for each percentile
    percentiles = calculate_percentiles_from_raster(
        file_registry['viewshed_no_zeros_path'], percentile_list)

    LOGGER.debug('percentiles_list : %s', percentiles)

    # Add the start_value to the beginning of the percentiles so that any value
    # before the start value is set to nodata
    percentiles.insert(0, 0)

    # Set nodata to a very small negative number
    percentile_nodata = -9999919

    # Classify the pixels of raster_dataset into groups and write
    # them to output
    pygeoprocessing.vectorize_datasets(
        [file_registry['viewshed_no_zeros_path']], raster_percentile,
        file_registry['viewshed_quality_path'], gdal.GDT_Int32,
        percentile_nodata, cell_size, 'intersection',
        assert_datasets_projected=False)

    if 'pop_path' in args:
        # Project AOI to Population to clip Population raster
        pop_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(
            args['pop_path'])
        pygeoprocessing.reproject_datasource_uri(
            args['aoi_path'], pop_wkt, file_registry['aoi_proj_pop_path'])
        # Clip Population by AOI
        pygeoprocessing.clip_dataset_uri(
            args['pop_path'], file_registry['aoi_proj_pop_path'],
            file_registry['clipped_pop_path'], False)

        pop_cell_size = projected_pixel_size(
                file_registry['clipped_pop_path'], aoi_srs)

        # Project Population to AOI
        pygeoprocessing.reproject_dataset_uri(
            file_registry['clipped_pop_path'], pop_cell_size, aoi_wkt,
            'nearest', file_registry['pop_proj_to_aoi_path'])

        # Dataset lists for rasters to align and the aligned paths
        dataset_uri_list = [file_registry['pop_proj_to_aoi_path'],
                            file_registry['viewshed_path']]
        dataset_out_uri_list = [file_registry['aligned_pop_path'],
                                file_registry['aligned_viewshed_path']]
        resample_method_list = ['nearest', 'nearest']

        # Set a factor value which is a holder for the ratio between
        # pop_cell_size and viewshed cell size
        cell_size_factor = 1

        if cell_size >= pop_cell_size:
            # Viewshed cell size is bigger, so use pop cell size
            # to resample viewshed_count raster
            out_pixel_size = pop_cell_size
        else:
            # Viewshed cell size is smaller, so use it's cell size
            # to resample population raster
            out_pixel_size = cell_size
            # Set cell_size_factor to the ratio between cell sizes, so that
            # we can later maintain population data integrity.
            cell_size_factor = pop_cell_size**2 / viewshed_cell_size**2

        pygeoprocessing.align_dataset_list(
            dataset_uri_list, dataset_out_uri_list, resample_method_list,
            out_pixel_size, 'intersection', 1,
            dataset_to_bound_index=None, aoi_uri=args['aoi_path'],
            assert_datasets_projected=True, all_touched=False)

        pop_nodata = pygeoprocessing.get_nodata_from_uri(
            file_registry['aligned_pop_path'])

        def pop_affected_op(pop, view):
            """Compute affected population."""
            valid_mask = ((pop != pop_nodata) & (view != nodata))

            pop_places = numpy.where(view[valid_mask] > 0, pop[valid_mask], 0)
            pop_final = numpy.empty(valid_mask.shape)
            pop_final[:] = nodata
            pop_final[valid_mask] = pop_places
            return pop_final

        def pop_unaffected_op(pop, view):
            """Compute unaffected population."""
            valid_mask = ((pop != pop_nodata))

            pop_places = numpy.where(view[valid_mask] == 0, pop[valid_mask], 0)
            pop_final = numpy.empty(valid_mask.shape)
            pop_final[:] = nodata
            pop_final[valid_mask] = pop_places
            return pop_final

        pygeoprocessing.vectorize_datasets(
            [file_registry['aligned_pop_path'],
             file_registry['aligned_viewshed_path']],
            pop_affected_op, file_registry['pop_affected_path'],
            gdal.GDT_Float32, nodata, out_pixel_size,
            "intersection", vectorize_op=False)

        pygeoprocessing.vectorize_datasets(
            [file_registry['aligned_pop_path'],
             file_registry['aligned_viewshed_path']],
            pop_unaffected_op, file_registry['pop_unaffected_path'],
            gdal.GDT_Float32, nodata, out_pixel_size,
            "intersection", vectorize_op=False)

        # Count up the affected population values
        affected_sum = 0
        affected_count = 0
        for _, block in pygeoprocessing.iterblocks(
                file_registry['pop_affected_path']):

            valid_mask = (block != nodata)
            affected_count += numpy.sum(valid_mask)
            affected_sum += numpy.sum(block[valid_mask])
        # Count up the unaffected population values
        unaffected_sum = 0
        unaffected_count = 0
        for _, block in pygeoprocessing.iterblocks(
                file_registry['pop_unaffected_path']):

            valid_mask = (block != nodata)
            unaffected_count += numpy.sum(valid_mask)
            unaffected_sum += numpy.sum(block[valid_mask])

        if args['pop_type'] == "Density":
            # If population raster is population density per area then
            # adjust sums to reflect as much, to get in correct units
            cell_area = out_pixel_size**2
            affected_sum = affected_sum * (affected_count * cell_area)
            unaffected_sum = unaffected_sum * (unaffected_count * cell_area)
        else:
            # If population raster is population counts per cell then
            # we need to adjust counts by any resampling so we don't
            # double count
            affected_sum = affected_sum / cell_size_factor
            unaffected_sum = unaffected_sum / cell_size_factor

        # Create output HTML file for population stats
        header = ("<center><H1>Scenic Quality Model</H1>"
                  "<H2>(Visual Impact from Objects)</H2></center>"
                  "<br><br><HR><br><H2>Population Statistics</H2>")
        page_header = {'type': 'text', 'section': 'head', 'text': header}

        table_data = [
            {'Number of Features Visible': 'None Visible',
             'Population (estimate)': unaffected_sum},
            {'Number of Features Visible': '1 or more Visible',
             'Population (estimate)': affected_sum}]
        table_columns = [
            {'name': 'Number of Features Visible', 'total': False},
            {'name': 'Population (estimate)', 'total': False}]
        table_args = {
            'type': 'table', 'section': 'body', 'data_type': 'dictionary',
            'data': table_data, 'columns': table_columns, 'sortable': False}

        report_args = {}
        report_args['title'] = 'Marine InVEST'
        report_args['out_uri'] = file_registry['pop_stats_path']
        report_args['elements'] = [table_args]

        natcap.invest.reporting.generate_report(report_args)

    if "overlap_path" in args:
        # Project AOI to overlap vector to clip
        overlap_srs = pygeoprocessing.get_spatial_ref_uri(args['overlap_path'])
        overlap_wkt = overlap_srs.ExportToWkt()
        pygeoprocessing.reproject_datasource_uri(
            args['aoi_path'], overlap_wkt,
            file_registry['aoi_proj_overlap_path'])
        # Clip overlap vector by AOI
        clip_datasource_layer(
            args['overlap_path'], file_registry['aoi_proj_overlap_path'],
            file_registry['overlap_clipped_path'])

        # Project overlap to AOI
        pygeoprocessing.reproject_datasource_uri(
            file_registry['overlap_clipped_path'], aoi_wkt,
            file_registry['overlap_projected_path'])

        LOGGER.debug("Adding id field to overlap features.")
        id_name = 'investID'
        setup_overlap_id_fields(
            file_registry['overlap_projected_path'], id_name)

        LOGGER.debug("Count overlapping pixels per area.")
        pixel_counts = pygeoprocessing.aggregate_raster_values_uri(
            file_registry['viewshed_no_zeros_path'],
            file_registry['overlap_projected_path'], id_name,
            ignore_nodata=True, all_touched=True).n_pixels

        LOGGER.debug("Pixel Counts: %s", pixel_counts)
        LOGGER.debug("Add area field to overlap features.")
        perc_field = '%_overlap'
        add_percent_overlap(
            file_registry['overlap_projected_path'], id_name, perc_field,
            pixel_counts, cell_size)

    LOGGER.info('deleting temporary files')
    for file_id in _TMP_BASE_FILES:
        try:
            if isinstance(file_registry[file_id], basestring):
                if os.path.splitext(file_registry[file_id])[1] == '.shp':
                    driver = ogr.GetDriverByName('ESRI Shapefile')
                    driver.DeleteDataSource(file_registry[file_id])
                else:
                    os.remove(file_registry[file_id])
            elif isinstance(file_registry[file_id], list):
                for index in xrange(len(file_registry[file_id])):
                    os.remove(file_registry[file_id][index])
        except OSError:
            # Let it go.
            pass


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


def calculate_percentiles_from_raster(raster_path, percentiles):
    """A memory efficient sort to determine the percentiles of a raster.

    Percentile algorithm currently used is the nearest rank method.

    Parameters:
        raster_path (string): a path to a gdal raster on disk
        percentiles (list): a list of desired percentiles to lookup
            ex: [25,50,75,90]
    Returns:
        a list of values corresponding to the percentiles
            from the percentiles list
    """
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)

    def numbers_from_file(fle):
        """Generate an iterator.

        Iterator generated from a file by loading all the numbers
            and yielding

        Parameters:
            fle (file object): file object
        """
        arr = numpy.load(fle)
        for num in arr:
            yield num

    # List to hold the generated iterators
    iters = []

    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    n_rows = raster.RasterYSize
    n_cols = raster.RasterXSize

    # Variable to count the total number of elements to compute percentile
    # from. This leaves out nodata values
    n_elements = 0

    # Set the row strides to be something reasonable, like 256MB blocks
    row_strides = max(int(2**28 / (4 * n_cols)), 1)

    for row_index in xrange(0, n_rows, row_strides):
        # It's possible we're on the last set of rows and the stride
        # is too big, update if so
        if row_index + row_strides >= n_rows:
            row_strides = n_rows - row_index

        # Read in raster chunk as array
        arr = band.ReadAsArray(0, row_index, n_cols, row_strides)

        tmp_path = pygeoprocessing.temporary_filename()
        tmp_file = open(tmp_path, 'wb')
        # Make array one dimensional for sorting and saving
        arr = arr.flatten()
        # Remove nodata values from array and thus percentile calculation
        arr = numpy.delete(arr, numpy.where(arr == nodata))
        # Tally the number of values relevant for calculating percentiles
        n_elements += len(arr)
        # Sort array before saving
        arr = numpy.sort(arr)

        numpy.save(tmp_file, arr)
        tmp_file.close()
        tmp_file = open(tmp_path, 'rb')
        tmp_file.seek(0)
        iters.append(numbers_from_file(tmp_file))
        arr = None

    # List to store the rank/index where each percentile will be found
    rank_list = []
    # For each percentile calculate nearest rank
    for perc in percentiles:
        rank = math.ceil(perc/100.0 * n_elements)
        rank_list.append(int(rank))

    # Need to handle 0th percentile case. 0th percentile is first element
    if 0 in rank_list:
        rank_list[rank_list.index(0)] = 1

    # A variable to burn through when doing heapq merge sort over the
    # iterators. Variable is used to check if we've iterated to a
    # specified rank spot, to grab percentile value
    counter = 1
    # Setup a list of 'nans' to replace with percentile results, modeled
    # after scipy.stats.scoreatpercentile function
    results = [float('nan')] * len(rank_list)

    LOGGER.debug('Percentile Rank List: %s', rank_list)

    for num in heapq.merge(*iters):
        # If a percentile rank has been hit, grab percentile value
        if counter in rank_list:
            LOGGER.debug('percentile value is : %s', num)
            results[rank_list.index(counter)] = int(num)
        counter += 1

    LOGGER.debug("Percentile Counter : %s" % counter)

    band = None
    raster = None
    return results


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


def projected_pixel_size(raster_path, target_spat_ref):
    """Transform source cell size to target spatial reference.

    Determine what the pixel size for the raster would be if projected in
        the target spatial reference. This is common for trying to keep
        the same pixel size ration when reprojecting a raster.
        Calculated by doing a Coordinate Transformation on the upper left
        point of the raster and on the adjacent point. The difference is
        then taken to determine the new cell size.

    Raises an exception if the raster is not square since this'll break most of
        the pygeoprocessing algorithms.

    Parameters:
        raster_path (string): path to a gdal dataset on disk.
        target_spat_ref (string): target spatial reference for the pixel size.

    Returns:
        transformed pixel size
    """
    # Create two points from the raster
    raster_gt = pygeoprocessing.geoprocessing.get_geotransform_uri(
        raster_path)
    point_one = (raster_gt[0], raster_gt[3])
    # Get X and Y cell size
    pixel_size_x = raster_gt[1]
    pixel_size_y = raster_gt[5]
    point_two = (point_one[0] + pixel_size_x, point_one[1] + pixel_size_y)

    raster_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(raster_path)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)
    # A coordinate transformation to help get the proper pixel size
    coord_trans = osr.CoordinateTransformation(raster_srs, target_spat_ref)

    # Transform two points into new spatial reference
    point_one_proj = coord_trans.TransformPoint(point_one[0], point_one[1])
    point_two_proj = coord_trans.TransformPoint(point_two[0], point_two[1])
    # Calculate the x/y difference between two points
    # taking the absolute value because the direction doesn't matter for pixel
    # size in the case of most coordinate systems where y increases up and x
    # increases to the right (right handed coordinate system).
    pixel_diff_x = abs(point_two_proj[0] - point_one_proj[0])
    pixel_diff_y = abs(point_two_proj[1] - point_one_proj[1])

    try:
        numpy.testing.assert_approx_equal(
            abs(pixel_diff_x), abs(pixel_diff_y))
        resulting_size = abs(pixel_diff_x)
    except AssertionError as e:
        LOGGER.warn(e)
        resulting_size = (
            abs(pixel_diff_x) + abs(pixel_diff_y)) / 2.0

    return resulting_size
