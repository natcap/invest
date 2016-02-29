import os
import sys
import math
import heapq

import numpy
import scipy.stats
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

_OUTPUT_BASE_FILES = {
    'viewshed_valuation_path': 'vshed.tif',
    'viewshed_path': 'viewshed_counts.tif',
    'viewshed_quality_path': 'vshed_qual.tif',
    'pop_stats_path': 'populationStats.html',
    'overlap_path': 'vp_overlap.shp'
    }

_INTERMEDIATE_BASE_FILES = {
    'pop_affected_path': 'affected_population.tif',
    'pop_unaffected_path': 'unaffected_population.tif',
    'aligned_pop_path' : 'aligned_pop.tif',
    'aligned_viewshed_path' : 'aligned_viewshed.tif',
    'viewshed_no_zeros_path' : 'view_no_zeros.tif'
    }

_TMP_BASE_FILES = {
    'aoi_proj_dem_path' : 'aoi_proj_to_dem.shp',
    'aoi_proj_pop_path' : 'aoi_proj_to_pop.shp',
    'aoi_proj_struct_path' : 'aoi_proj_to_struct.shp',
    'structures_clipped_path' : 'structures_clipped.shp',
    'structures_projected_path' : 'structures_projected.shp',
    'clipped_dem_path' : 'dem_clipped.tif',
    'dem_proj_to_aoi_path' : 'dem_proj_to_aoi.tif',
    'clipped_pop_path' : 'pop_clipped.tif',
    'pop_proj_to_aoi_path' : 'pop_proj_to_aoi.tif'
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
        args['refraction'] (float): (optional) number indicating the refraction coefficient
            to use for calculating curvature of the earth.
        args['population_path'] (string): (optional) path to a raster for population
        args['overlap_path'] (string): (optional)
        args['results_suffix] (string): (optional) string to append to any
            output files
        args['valuation_function'] (string): type of economic function to use
            for valuation. Either 3rd degree polynomial or logarithmic.
        args['a_coef'] (string):
        args['b_coef'] (string):
        args['c_coef'] (string):
        args['d_coef'] (string):
        args['max_valuation_radius'] (string):

    """
    LOGGER.info("Start Scenic Quality Model")

    curvature_correction = float(args['refraction'])

    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    pygeoprocessing.create_directories([output_dir, intermediate_dir])

    file_suffix = natcap.invest.utils.make_suffix_string(
        args, 'results_suffix')

    LOGGER.info('Building file registry')
    file_registry = natcap.invest.utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    # Clip DEM by AOI and reclass
    dem_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(args['dem_path'])
    pygeoprocessing.reproject_datasource_uri(
        args['aoi_path'], dem_wkt, file_registry['aoi_proj_dem_path'])

    pygeoprocessing.clip_dataset_uri(
        args['dem_path'], file_registry['aoi_proj_dem_path'],
         file_registry['clipped_dem_path'], False)

    # Clip structures by AOI
    structures_srs = pygeoprocessing.get_spatial_ref_uri(args['structure_path'])
    structures_wkt = structures_srs.ExportToWkt()
    pygeoprocessing.reproject_datasource_uri(
        args['aoi_path'], structures_wkt, file_registry['aoi_proj_struct_path'])

    clip_datasource_layer(
        args['structure_path'], file_registry['aoi_proj_struct_path'],
        file_registry['structures_clipped_path'])

    # Project Structures and DEM to AOI
    aoi_srs = pygeoprocessing.get_spatial_ref_uri(args['aoi_path'])
    aoi_wkt = aoi_srs.ExportToWkt()
    pygeoprocessing.reproject_datasource_uri(
        file_registry['structures_clipped_path'], aoi_wkt,
        file_registry['structures_projected_path'])

    # Get a point from the clipped data object to use later in helping
    # determine proper pixel size
    raster_gt = pygeoprocessing.geoprocessing.get_geotransform_uri(
        file_registry['clipped_dem_path'])
    point_one = (raster_gt[0], raster_gt[3])

    # Create a Spatial Reference from the rasters WKT
    dem_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(
        file_registry['clipped_dem_path'])
    dem_srs = osr.SpatialReference()
    dem_srs.ImportFromWkt(dem_wkt)

    # A coordinate transformation to help get the proper pixel size of
    # the reprojected raster
    coord_trans = osr.CoordinateTransformation(dem_srs, aoi_srs)

    pixel_size = pixel_size_based_on_coordinate_transform_uri(
            file_registry['clipped_dem_path'], coord_trans, point_one)

    LOGGER.debug('Projected Pixel Size: %s', pixel_size[0])

    pygeoprocessing.reproject_dataset_uri(
        file_registry['clipped_dem_path'], pixel_size[0], aoi_wkt,
        'bilinear', file_registry['dem_proj_to_aoi_path'])

    # Read in valuation coefficients
    coef_a = float(args['a_coefficient'])
    coef_b = float(args['b_coefficient'])
    coef_c = float(args['c_coefficient'])
    coef_d = float(args['d_coefficient'])
    max_val_radius = float(args['max_valuation_radius'])

    def polynomial_val(dist, weight):
        """Third Degree Polynomial Valuation function."""

        valid_mask = ((dist != dist_nodata) & (weight != nodata))
        max_dist_mask = (dist > max_val_radius)
        # Based off of Equation 2 in the Users Guide
        dist_one = numpy.where(
            dist[valid_mask] >= 1000,
            (coef_a + coef_b * dist[valid_mask] +
             coef_c * dist[valid_mask]**2 + coef_d * dist[valid_mask]**3) * weight[valid_mask],
            ((coef_a + coef_b * 1000 + coef_c * 1000**2 + coef_d * 1000**3) -
             (coef_b + 2 * coef_c * 1000 + 3 * coef_d * 1000**2) *
             (1000 - dist[valid_mask])) * weight[valid_mask]
        )

        dist_final = numpy.empty(valid_mask.shape)
        dist_final[:] = dist_nodata
        dist_final[valid_mask] = dist_one
        dist_final[max_dist_mask] = 0.0
        return dist_final

    def log_val(dist, weight):
        """Logarithmic Valuation function."""

        valid_mask = ((dist != dist_nodata) & (weight != nodata))
        max_dist_mask = dist > max_valuation_radius
        # Based off of Equation 1 in the Users Guide
        dist_one = numpy.where(
            dist[valid_mask] >= 1000,
            (coef_a + coef_b * numpy.log(dist[valid_mask])) * weight[valid_mask],
            (coef_a + coef_b * numpy.log(1000) - (coef_b / 1000) *
             (1000 - dist[valid_mask])) * weight[valid_mask]
        )

        dist_final = numpy.empty(valid_mask.shape)
        dist_final[:] = dist_nodata
        dist_final[valid_mask] = dist_one
        dist_final[max_dist_mask] = 0.0
        return dist_final

    def add_op(raster_one, raster_two):
        """Aggregate valuation matrices.

        Sums all non-nodata values.

        Parameters:
            *raster (list of numpy arrays): List of numpy matrices.

        Returns:
            numpy.array where the pixel value represents the combined
                pixel values found across all matrices."""

        raster_one[raster_one == nodata] = 0
        raster_two[raster_two == nodata] = 0
        return raster_one + raster_two

    val_raster_list = []
    if args["valuation_function"] == "polynomial: a + bx + cx^2 + dx^3":
        val_op = polynomial_val
    else:
        val_op = log_val

    viewpoints_vector = ogr.Open(file_registry['structures_projected_path'])
    viewshed_dir = intermediate_dir
    rasters_dict = {}
    keep_viewsheds = args['keep_feat_viewsheds']
    keep_val_viewsheds = args['keep_val_viewsheds']

    for layer in viewpoints_vector:
        num_features = layer.GetFeatureCount()

        feat_val_paths = []
        feat_views_paths = []
        for feat_num in xrange(num_features - 1):
            val_path = pygeoprocessing.temporary_filename()
            feat_val_paths.append(val_path)
            feat_path = pygeoprocessing.temporary_filename()
            feat_views_paths.append(feat_path)

        feat_val_paths.append(file_registry['viewshed_valuation_path'])
        feat_views_paths.append(file_registry['viewshed_path'])

        for index, point in enumerate(layer):
            geometry = point.GetGeometryRef()
            feature_id = point.GetFID()

            # Coordinates in map units
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
                weight = point.GetField('coeff')
            except ValueError:
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
                    viewshed_filepath, None, True, curvature_correction, max_radius,
                    viewpoint_height)
            except ValueError:
                # When pixel is over nodata and we told it to skip
                LOGGER.info('Viewpoint %s is over nodata, skipping.', index)

            nodata = pygeoprocessing.get_nodata_from_uri(viewshed_filepath)
            cell_size = pygeoprocessing.get_cell_size_from_uri(viewshed_filepath)
            weighted_view_path = pygeoprocessing.temporary_filename()

            def weight_factor_op(view):
                """ """

                return numpy.where(view != nodata, view * weight, nodata)

            pygeoprocessing.vectorize_datasets(
                [viewshed_filepath], weight_factor_op, weighted_view_path,
                gdal.GDT_Float32, nodata, cell_size, 'intersection',
                vectorize_op=False)

            # do a distance transform on each viewpoint raster
            dist_path = pygeoprocessing.temporary_filename()
            pygeoprocessing.distance_transform_edt(
                viewshed_filepath, dist_path, process_pool=None)

            dist_nodata = pygeoprocessing.get_nodata_from_uri(dist_path)

            vshed_val_path = os.path.join(
                viewshed_dir, 'val_viewshed_%s.tif' % index)
            # run valuation equation on distance raster
            pygeoprocessing.vectorize_datasets(
                [dist_path, weighted_view_path], val_op, vshed_val_path,
                gdal.GDT_Float32, nodata,
                cell_size, 'intersection', vectorize_op=False)

            if index == 0:
                shutil.copy(vshed_val_path, feat_val_paths[index])
                shutil.copy(viewshed_filepath, feat_views_paths[index])

            else:
                for file_path, out_list in zip([vshed_val_path,
                                                viewshed_filepath],
                                                [feat_val_paths,
                                                feat_views_paths]):
                    pygeoprocessing.vectorize_datasets(
                        [file_path, out_list[index - 1]], add_op, out_list[index],
                        gdal.GDT_Float32, nodata, cell_size, 'intersection',
                        vectorize_op=False, datasets_are_pre_aligned=True)

                    os.remove(out_list[index - 1])

            os.remove(dist_path)
            os.remove(weighted_view_path)

            if keep_val_viewsheds == 'No':
                os.remove(vshed_val_path)
            if keep_viewsheds == 'No':
                os.remove(viewshed_filepath)

    # Do quantiles on viewshed_uri
    percentile_list = [25, 50, 75, 100]

    # Set 0 values to nodata before calculating percentiles, since 0 values
    # indicate there was no viewpoint effects

    def zero_to_nodata(view):
        """ """
        return numpy.where(view == 0., nodata, view)

    pygeoprocessing.vectorize_datasets(
        [file_registry['viewshed_valuation_path']], zero_to_nodata,
        file_registry['viewshed_no_zeros_path'], gdal.GDT_Int32, nodata,
        cell_size, 'intersection', assert_datasets_projected=False,
        vectorize_op=False)

    def raster_percentile(band):
        """Operation to use in vectorize_datasets that takes
            the pixels of 'band' and groups them together based on
            their percentile ranges.

            band - A gdal raster band

            returns - An integer that places each pixel into a group
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
    pixel_size = pygeoprocessing.get_cell_size_from_uri(
        file_registry['viewshed_valuation_path'])

    # Classify the pixels of raster_dataset into groups and write
    # them to output
    pygeoprocessing.vectorize_datasets(
        [file_registry['viewshed_no_zeros_path']], raster_percentile,
        file_registry['viewshed_quality_path'], gdal.GDT_Int32, percentile_nodata,
        pixel_size, 'intersection', assert_datasets_projected=False)

    # population html stuff
    if 'pop_path' in args:

        #clip population
        pop_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(
            args['pop_path'])
        pygeoprocessing.reproject_datasource_uri(
            args['aoi_path'], pop_wkt, file_registry['aoi_proj_pop_path'])

        pygeoprocessing.clip_dataset_uri(
            args['pop_path'], file_registry['aoi_proj_pop_path'],
            file_registry['clipped_pop_path'], False)

        #reproject clipped population
        LOGGER.debug("Reprojecting clipped population raster.")
        #vs_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(viewshed_path)
        #pop_cell_size = pygeoprocessing.get_cell_size_from_uri(pop_clip_path)


        # Get a point from the clipped data object to use later in helping
        # determine proper pixel size
        pop_gt = pygeoprocessing.geoprocessing.get_geotransform_uri(
            file_registry['clipped_pop_path'])
        point_one = (pop_gt[0], pop_gt[3])

        # Create a Spatial Reference from the rasters WKT
        pop_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(
            file_registry['clipped_pop_path'])
        pop_srs = osr.SpatialReference()
        pop_srs.ImportFromWkt(pop_wkt)

        # A coordinate transformation to help get the proper pixel size of
        # the reprojected raster
        coord_trans = osr.CoordinateTransformation(pop_srs, aoi_srs)

        pop_cell_size = pixel_size_based_on_coordinate_transform_uri(
                file_registry['clipped_pop_path'], coord_trans, point_one)

        pygeoprocessing.reproject_dataset_uri(
            file_registry['clipped_pop_path'], pop_cell_size[0], aoi_wkt,
            'bilinear', file_registry['pop_proj_to_aoi_path'])

        viewshed_cell_size = pygeoprocessing.get_cell_size_from_uri(
            file_registry['viewshed_path'])

        dataset_uri_list = [file_registry['pop_proj_to_aoi_path'], file_registry['viewshed_path']]
        dataset_out_uri_list = [file_registry['aligned_pop_path'], file_registry['aligned_viewshed_path']]
        resample_method_list = ['nearest', 'nearest']

        cell_size_factor = 1

        if viewshed_cell_size > pop_cell_size[0]:
            out_pixel_size = pop_cell_size[0]
            mode = 'intersection'
            dataset_to_align_index = 1

            pygeoprocessing.align_dataset_list(
                dataset_uri_list, dataset_out_uri_list, resample_method_list,
                out_pixel_size, mode, dataset_to_align_index,
                dataset_to_bound_index=None, aoi_uri=args['aoi_path'],
                assert_datasets_projected=True, all_touched=False)
        else:
            out_pixel_size = viewshed_cell_size
            mode = 'intersection'
            dataset_to_align_index = 1

            cell_size_factor = pop_cell_size[0]**2 / viewshed_cell_size**2

            pygeoprocessing.align_dataset_list(
                dataset_uri_list, dataset_out_uri_list, resample_method_list,
                out_pixel_size, mode, dataset_to_align_index,
                dataset_to_bound_index=None, aoi_uri=rgs['aoi_path'],
                assert_datasets_projected=True, all_touched=False)

        pop_nodata = pygeoprocessing.get_nodata_from_uri(
            file_registry['aligned_pop_path'])

        def pop_affected_op(pop, view):
            valid_mask = ((pop != pop_nodata) | (view != nodata))

            pop_places = numpy.where(view[valid_mask] > 0, pop[valid_mask], 0)
            pop_final = numpy.empty(valid_mask.shape)
            pop_final[:] = nodata
            pop_final[valid_mask] = pop_places
            return pop_final

        def pop_unaffected_op(pop, view):
            valid_mask = ((pop != pop_nodata))

            pop_places = numpy.where(view[valid_mask] == 0, pop[valid_mask], 0)
            pop_final = numpy.empty(valid_mask.shape)
            pop_final[:] = nodata
            pop_final[valid_mask] = pop_places
            return pop_final

        pygeoprocessing.vectorize_datasets(
            [file_registry['pop_proj_to_aoi_path'], file_registry['viewshed_path']],
            pop_affected_op, file_registry['pop_affected_path'],
            gdal.GDT_Float32, nodata, viewshed_cell_size,
            "intersection", vectorize_op=False)

        pygeoprocessing.vectorize_datasets(
            [file_registry['pop_proj_to_aoi_path'], file_registry['viewshed_path']],
            pop_unaffected_op, file_registry['pop_unaffected_path'],
            gdal.GDT_Float32, nodata, viewshed_cell_size,
            "intersection", vectorize_op=False)

        affected_sum = 0
        affected_count = 0
        for _, block in pygeoprocessing.iterblocks(
            file_registry['pop_affected_path']):

            valid_mask = (block != nodata)
            affected_count += numpy.sum(valid_mask)
            affected_sum += numpy.sum(block[valid_mask])

        unaffected_sum = 0
        unaffected_count = 0
        for _, block in pygeoprocessing.iterblocks(
            file_registry['pop_unaffected_path']):

            valid_mask = (block != nodata)
            unaffected_count += numpy.sum(valid_mask)
            unaffected_sum += numpy.sum(block[valid_mask])

        if args['pop_type'] == "Density":
            cell_area = out_pixel_size**2
            affected_sum = affected_sum * (affected_count * cell_area)
            unaffected_sum = unaffected_sum * (unaffected_count * cell_area)
        else:
            affected_sum = affected_sum / cell_size_factor
            unaffected_sum = unaffected_sum / cell_size_factor

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
        pygeoprocessing.copy_datasource_uri(
            args["overlap_path"], file_registry['overlap_path'])

        #def zero_to_nodata(view):
        #    """
        #    """
        #    return numpy.where(view <= 0, nodata, view)

        #pygeoprocessing.vectorize_datasets(
        #    [file_registry['viewshed_path']], zero_to_nodata,
        #    file_registry['viewshed_no_zeros_path'], gdal.GDT_Float32,
        #    nodata, pixel_size, 'intersection',
        #    assert_datasets_projected=False, vectorize_op=False)

        LOGGER.debug("Adding id field to overlap features.")
        id_name = 'investID'
        setup_overlap_id_fields(file_registry['overlap_path'], id_name)

        LOGGER.debug("Count overlapping pixels per area.")
        pixel_counts = pygeoprocessing.aggregate_raster_values_uri(
            file_registry['viewshed_no_zeros_path'],
            file_registry['overlap_path'], id_name,
            ignore_nodata=True).n_pixels

        LOGGER.debug("Add area field to overlap features.")
        perc_field = '%_overlap'
        add_percent_overlap(
            file_registry['overlap_path'], perc_field, pixel_counts,
            pixel_size)


    LOGGER.info('deleting temporary files')
    for file_id in _TMP_BASE_FILES:
        try:
            if isinstance(file_registry[file_id], basestring):
                os.remove(file_registry[file_id])
            elif isinstance(file_registry[file_id], list):
                for index in xrange(len(file_registry[file_id])):
                    os.remove(file_registry[file_id][index])
        except OSError:
            # Let it go.
            pass

def setup_overlap_id_fields(shapefile_path, id_name):
    """
    """
    shapefile = ogr.Open(shapefile_path, 1)
    layer = shapefile.GetLayer()
    id_field = ogr.FieldDefn(id_name, ogr.OFTInteger)
    layer.CreateField(id_field)

    for feature_id in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        feature.SetField(id_name, feature_id)
        layer.SetFeature(feature)

def add_percent_overlap(
    overlap_path, perc_name, pixel_counts, pixel_size):
    """
    """
    shapefile = ogr.Open(overlap_path, 1)
    layer = shapefile.GetLayer()
    perc_field = ogr.FieldDefn(perc_name, ogr.OFTReal)
    layer.CreateField(perc_field)

    for feature_id in pixel_counts.keys():
        feature = layer.GetFeature(feature_id)
        geom = feature.GetGeometryRef()
        geom_area = geom.GetArea()
        pixel_area = pixel_size**2 * pixel_counts[feature_id]
        feature.SetField(perc_name, pixel_area / geom_area)
        layer.SetFeature(feature)

def calculate_percentiles_from_raster(raster_path, percentiles):
    """Does a memory efficient sort to determine the percentiles
        of a raster. Percentile algorithm currently used is the
        nearest rank method.

        raster_path - a path to a gdal raster on disk
        percentiles - a list of desired percentiles to lookup
            ex: [25,50,75,90]

        returns - a list of values corresponding to the percentiles
            from the percentiles list
    """
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)

    def numbers_from_file(fle):
        """Generates an iterator from a file by loading all the numbers
            and yielding

            fle = file object
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

    #Set the row strides to be something reasonable, like 256MB blocks
    row_strides = max(int(2**28 / (4 * n_cols)), 1)

    for row_index in xrange(0, n_rows, row_strides):
        #It's possible we're on the last set of rows and the stride
        #is too big, update if so
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
    # Setup a list of zeros to replace with percentile results
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
        raise IntersectionError('Intersection ERROR: clip_datasource_layer '
            'found no intersection between: file - %s and file - %s.' %
            (shape_to_clip_path, binding_shape_path))


def pixel_size_based_on_coordinate_transform_uri(
        dataset_uri, coord_trans, point):
    """Get width and height of cell in meters.

    A wrapper for pixel_size_based_on_coordinate_transform that takes a dataset
    uri as an input and opens it before sending it along.

    Args:
        dataset_uri (string): a URI to a gdal dataset

        All other parameters pass along

    Returns:
        result (tuple): (pixel_width_meters, pixel_height_meters)
    """
    dataset = gdal.Open(dataset_uri)
    geo_tran = dataset.GetGeoTransform()
    pixel_size_x = geo_tran[1]
    pixel_size_y = geo_tran[5]
    top_left_x = point[0]
    top_left_y = point[1]
    # Create the second point by adding the pixel width/height
    new_x = top_left_x + pixel_size_x
    new_y = top_left_y + pixel_size_y
    # Transform two points into meters
    point_1 = coord_trans.TransformPoint(top_left_x, top_left_y)
    point_2 = coord_trans.TransformPoint(new_x, new_y)
    # Calculate the x/y difference between two points
    # taking the absolue value because the direction doesn't matter for pixel
    # size in the case of most coordinate systems where y increases up and x
    # increases to the right (right handed coordinate system).
    pixel_diff_x = abs(point_2[0] - point_1[0])
    pixel_diff_y = abs(point_2[1] - point_1[1])

    # Close and clean up dataset
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    return (pixel_diff_x, pixel_diff_y)
