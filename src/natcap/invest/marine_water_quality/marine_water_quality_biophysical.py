"""InVEST Marine Water Quality Biophysical model."""

import logging
import os
import csv

from osgeo import ogr
from osgeo import gdal
import numpy
import pygeoprocessing

from . import marine_water_quality_core
from .. import utils

LOGGER = logging.getLogger('natcap.invest.marine_water_quality.biophysical')


def execute(args):
    """Marine Water Quality.

    Main entry point for the InVEST 3.0 marine water quality
    biophysical model.

    Args:
        args['workspace_dir'] (string): Directory to place outputs
        args['results_suffix'] (string): a string to append to any output file
            name (optional)
        args['aoi_poly_uri'] (string): OGR polygon Datasource indicating region
            of interest to run the model.  Will define the grid.
        args['land_poly_uri'] (string): OGR polygon DataSource indicating areas
            where land is.
        args['pixel_size'] (float): float indicating pixel size in meters of
            output grid.
        args['layer_depth'] (float): float indicating the depth of the grid
            cells in meters.
        args['source_points_uri'] (string): OGR point Datasource indicating
            point sources of pollution.
        args['source_point_data_uri'] (string): csv file indicating the
            biophysical properties of the point sources.
        args['kps'] (float): float indicating decay rate of pollutant (kg/day)
        args['tide_e_points_uri'] (string): OGR point Datasource with spatial
            information about the E parameter
        args['adv_uv_points_uri'] (string): optional OGR point Datasource with
            spatial advection u and v vectors.

    Returns:
        nothing
    """
    LOGGER.info("Starting MWQ execute")

    # append a _ to the suffix if it's not empty and doens't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    output_directory = os.path.join(args['workspace_dir'], 'output')
    intermediate_directory = os.path.join(
        args['workspace_dir'], 'intermediate')
    pygeoprocessing.create_directories(
        [output_directory, intermediate_directory])

    # Create a grid based on the AOI
    LOGGER.info("Creating grid based on the AOI polygon")
    pixel_size = args['pixel_size']
    # the nodata value will be a min float
    nodata_out = -1.0
    raster_out_uri = os.path.join(
        intermediate_directory, 'concentration_grid%s.tif' % file_suffix)
    pygeoprocessing.create_raster_from_vector_extents_uri(
        args['aoi_poly_uri'], pixel_size, gdal.GDT_Float32,
        nodata_out, raster_out_uri)

    # create a temporary grid of interpolated points for tide_e and adv_uv
    LOGGER.info("Creating grids for the interpolated tide E and ADV uv points")
    tide_e_uri = os.path.join(
        intermediate_directory, 'tide_e%s.tif' % file_suffix)
    pygeoprocessing.new_raster_from_base_uri(
        raster_out_uri, tide_e_uri, 'GTiff', nodata_out, gdal.GDT_Float32)
    adv_u_uri = os.path.join(
        intermediate_directory, 'adv_u%s.tif' % file_suffix)
    pygeoprocessing.new_raster_from_base_uri(
        raster_out_uri, adv_u_uri, 'GTiff', nodata_out, gdal.GDT_Float32,
        fill_value=0)
    adv_v_uri = os.path.join(
        intermediate_directory, 'adv_v%s.tif' % file_suffix)
    pygeoprocessing.new_raster_from_base_uri(
        raster_out_uri, adv_v_uri, 'GTiff', nodata_out, gdal.GDT_Float32,
        fill_value=0)
    in_water_uri = os.path.join(
        intermediate_directory, 'in_water%s.tif' % file_suffix)
    pygeoprocessing.new_raster_from_base_uri(
        raster_out_uri, in_water_uri, 'GTiff', nodata_out, gdal.GDT_Byte,
        fill_value=1)

    # Set up the in_water_array
    LOGGER.info("Calculating the in_water array")
    in_water_raster = gdal.Open(in_water_uri, gdal.GA_Update)
    in_water_band = in_water_raster.GetRasterBand(1)
    land_poly = ogr.Open(args['land_poly_uri'])
    land_layer = land_poly.GetLayer()
    gdal.RasterizeLayer(in_water_raster, [1], land_layer, burn_values=[0])
    in_water_array = in_water_band.ReadAsArray()
    in_water_function = numpy.vectorize(lambda x: x == 1)
    in_water_array = in_water_function(in_water_array)
    in_water_band = None
    in_water_raster = None
    # Interpolate the datasource points onto a raster the same size as
    # raster_out
    LOGGER.info("Interpolating kh_km2_day onto raster")
    pygeoprocessing.vectorize_points_uri(
        args['tide_e_points_uri'], 'E_km2_day', tide_e_uri)
    if 'adv_uv_points_uri' in args and args['adv_uv_points_uri'] != '':
        # if adv_uv_points is not defined, then those are all 0 rasters
        LOGGER.info("Interpolating U_m_sec_ onto raster")
        pygeoprocessing.vectorize_points_uri(
            args['adv_uv_points_uri'], 'U_m_sec_', adv_u_uri)
        LOGGER.info("Interpolating V_m_sec_ onto raster")
        pygeoprocessing.vectorize_points_uri(
            args['adv_uv_points_uri'], 'V_m_sec_', adv_v_uri)

    # Mask the interpolated points to the land polygon
    LOGGER.info("Masking Tide E and ADV UV to the land polygon")
    for dataset_uri in [tide_e_uri, adv_u_uri, adv_v_uri]:
        dataset = gdal.Open(dataset_uri, gdal.GA_Update)
        band = dataset.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        gdal.RasterizeLayer(dataset, [1], land_layer, burn_values=[nodata])
        band = None
        dataset = None

    source_point_values = {}

    raster_out_gt = pygeoprocessing.get_geotransform_uri(raster_out_uri)

    def convert_to_grid_coords(point):
        """Helper to convert source points to numpy grid coordinates

           point - a list of the form [y0, x0]

           returns a projected point in the gridspace coordinates of
               raster_out"""

        x_grid = int((point[1] - raster_out_gt[0]) / raster_out_gt[1])
        y_grid = int((point[0] - raster_out_gt[3]) / raster_out_gt[5])

        return [y_grid, x_grid]

    LOGGER.info("Load the point sources")
    source_points = ogr.Open(args['source_points_uri'])
    source_layer = source_points.GetLayer()
    aoi_poly = ogr.Open(args['aoi_poly_uri'])
    aoi_layer = aoi_poly.GetLayer()
    aoi_polygon = aoi_layer.GetFeature(0)
    aoi_geometry = aoi_polygon.GetGeometryRef()
    for point_feature in source_layer:
        point_geometry = point_feature.GetGeometryRef()
        if aoi_geometry.Contains(point_geometry):
            point = point_geometry.GetPoint()
            point_id = point_feature.GetField('id')
            LOGGER.debug("point and id %s %s", point, point_id)
            # Appending point geometry with y first so it can be converted
            # to the numpy (row,col) 2D notation easily.
            source_point_values[point_id] = {
                'point': convert_to_grid_coords([point[1], point[0]])
            }

    csv_file = open(args['source_point_data_uri'])
    reader = csv.DictReader(csv_file)
    for row in reader:
        point_id = int(row['ID'])

        # Look up the concentration
        wps_concentration = float(row['WPS'])

        # This merges the current dictionary with a new one that includes WPS
        source_point_values[point_id] = dict(
            source_point_values[point_id].items() +
            {'WPS': wps_concentration}.items())

    LOGGER.info("Checking to see if all the points have WPS values")
    points_to_ignore = []
    for point_id in source_point_values:
        if 'WPS' not in source_point_values[point_id]:
            LOGGER.warn("point %s has no source parameters from the CSV.  "
                        "Ignoring that point.", point_id)
            # Can't delete out of the dictionary that we're iterating over
            points_to_ignore.append(point_id)
    # Deleting the points we don't have data for
    for point in points_to_ignore:
        del source_point_values[point]
    LOGGER.debug("these are the source points %s", source_point_values)
    # Convert the georeferenced source coordinates to grid coordinates
    LOGGER.info("Solving advection/diffusion equation")

    tide_e_memory_mapped_uri = pygeoprocessing.temporary_filename()
    #tide_e_memory_mapped_file = open(tide_e_memory_mapped_uri, 'wb')
    tide_e_array = pygeoprocessing.load_memory_mapped_array(
        tide_e_uri, tide_e_memory_mapped_uri)

    # convert E from km^2/day to m^2/day
    LOGGER.info("Convert tide E form km^2/day to m^2/day")
    tide_e_array[tide_e_array != nodata_out] *= 1000.0 ** 2

    # convert adv u from m/sec to m/day
    adv_u_memory_mapped_uri = pygeoprocessing.temporary_filename()
    adv_u_array = pygeoprocessing.load_memory_mapped_array(
        adv_u_uri, adv_u_memory_mapped_uri)
    adv_v_memory_mapped_uri = pygeoprocessing.temporary_filename()
    adv_v_array = pygeoprocessing.load_memory_mapped_array(
        adv_v_uri, adv_v_memory_mapped_uri)
    adv_u_array[adv_u_array != nodata_out] *= 86400.0
    adv_v_array[adv_v_array != nodata_out] *= 86400.0

    # If the cells are square then it doesn't matter if we look at x or y
    # but if different, we need just one value, so take the average.  Not the
    # best, but better than nothing.
    cell_size = pygeoprocessing.get_cell_size_from_uri(raster_out_uri)

    concentration_array = marine_water_quality_core.diffusion_advection_solver(
        source_point_values, args['kps'], in_water_array, tide_e_array,
        adv_u_array, adv_v_array, nodata_out, cell_size, args['layer_depth'])
    # The numerical solver might have slightly small negative values, this
    # sets them to 0.0
    concentration_array[
        ~numpy.isclose(concentration_array, nodata_out) &
        (concentration_array < 0.0)] = 0.0

    raster_out = gdal.Open(raster_out_uri, gdal.GA_Update)
    raster_out_band = raster_out.GetRasterBand(1)
    raster_out_band.WriteArray(concentration_array, 0, 0)
    raster_out_band = None
    raster_out = None

    # rasterize away anything outside of the AOI
    concentration_uri = os.path.join(
        output_directory, 'concentration%s.tif' % file_suffix)
    pygeoprocessing.vectorize_datasets(
        [raster_out_uri], lambda x: x, concentration_uri, gdal.GDT_Float32,
        nodata_out, cell_size, "intersection", aoi_uri=args['aoi_poly_uri'])

    pygeoprocessing.calculate_raster_stats_uri(
        concentration_uri)

    LOGGER.info("Done with marine water quality.")
    LOGGER.info("Intermediate rasters are located in %s",
                intermediate_directory)
    LOGGER.info("Output rasters are located in %s", output_directory)
