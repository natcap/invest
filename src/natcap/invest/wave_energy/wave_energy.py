"""InVEST Wave Energy Model Core Code"""
from __future__ import absolute_import

import heapq
import math
import os
import logging
import csv
import struct
import itertools

from bisect import bisect
import numpy
import pandas
import scipy
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shapely.wkb
import shapely.prepared
from shapely import speedups

import pygeoprocessing
import natcap.invest.pygeoprocessing_0_3_3.geoprocessing
from .. import validation
from .. import utils

import pdb

LOGGER = logging.getLogger('natcap.invest.wave_energy.wave_energy')

speedups.enable()

class IntersectionError(Exception):
    """A custom error message for when the AOI does not intersect any wave
        data points.
    """
    pass


def execute(args):
    """Wave Energy.

    Executes both the biophysical and valuation parts of the
    wave energy model (WEM). Files will be written on disk to the
    intermediate and output directories. The outputs computed for
    biophysical and valuation include: wave energy capacity raster,
    wave power raster, net present value raster, percentile rasters for the
    previous three, and a point shapefile of the wave points with
    attributes.

    Args:
        workspace_dir (str): Where the intermediate and output folder/files
            will be saved. (required)
        wave_base_data_path (str): Directory location of wave base data
            including WW3 data and analysis area shapefile. (required)
        analysis_area_path (str): A string identifying the analysis area of
            interest. Used to determine wave data shapefile, wave data text
            file, and analysis area boundary shape. (required)
        aoi_path (str): A polygon shapefile outlining a more detailed area
            within the analysis area. This shapefile should be projected with
            linear units being in meters. (required to run Valuation model)
        machine_perf_path (str): The path of a CSV file that holds the
            machine performance table. (required)
        machine_param_path (str): The path of a CSV file that holds the
            machine parameter table. (required)
        dem_path (str): The path of the Global Digital Elevation Model (DEM).
            (required)
        suffix (str): A python string of characters to append to each output
            filename (optional)
        valuation_container (boolean): Indicates whether the model includes
            valuation
        land_gridPts_path (str): A CSV file path containing the Landing and
            Power Grid Connection Points table. (required for Valuation)
        machine_econ_path (str): A CSV file path for the machine economic
            parameters table. (required for Valuation)
        number_of_machines (int): An integer specifying the number of
            machines for a wave farm site. (required for Valuation)

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'wave_base_data_path': 'path/to/base_data_dir',
            'analysis_area_path': 'West Coast of North America and Hawaii',
            'aoi_path': 'path/to/shapefile',
            'machine_perf_path': 'path/to/csv',
            'machine_param_path': 'path/to/csv',
            'dem_path': 'path/to/raster',
            'suffix': '_results',
            'valuation_container': True,
            'land_gridPts_path': 'path/to/csv',
            'machine_econ_path': 'path/to/csv',
            'number_of_machines': 28,
        }

    """

    # Create the Output and Intermediate directories if they do not exist.
    workspace = args['workspace_dir']
    output_dir = os.path.join(workspace, 'output')
    intermediate_dir = os.path.join(workspace, 'intermediate')
    utils.make_directories([intermediate_dir, output_dir])

    # Append a _ to the suffix if it's not empty and doesn't already have one
    file_suffix = utils.make_suffix_string(args, 'suffix')

    # Get the path for the DEM
    dem_path = args['dem_path']

    # Create a dictionary that stores the wave periods and wave heights as
    # arrays. Also store the amount of energy the machine produces
    # in a certain wave period/height state as a 2D array
    machine_perf_dict = {}
    machine_perf_data = pandas.read_csv(args['machine_perf_path'])
    # Get the column header which is the first row in the file
    # and specifies the range of wave periods
    machine_perf_dict['periods'] = machine_perf_data.columns.values[1:]
    # Build up the row header by taking the first element in each row
    # This is the range of heights
    machine_perf_dict['heights'] = machine_perf_data.iloc[:, 0].values
    # Set the key for storing the machine's performance
    for i in range(len(machine_perf_dict['heights'])):
        bin_matrix_row = machine_perf_data.iloc[i, 1:].values
        # Expand the dimension from (N,) to (N,1)
        bin_matrix_row = numpy.expand_dims(bin_matrix_row, axis=0)
        # Concatenate each row along axis 0
        if i == 0:
            machine_perf_dict['bin_matrix'] = bin_matrix_row
        else:
            machine_perf_dict['bin_matrix'] = numpy.concatenate(
                (machine_perf_dict['bin_matrix'], bin_matrix_row))

    # Check if x and y dimensions of the bin_matrix array equal the size of
    # heights and periods
    if machine_perf_dict['bin_matrix'].shape != (
            machine_perf_dict['heights'].size,
            machine_perf_dict['periods'].size):
        raise ValueError(
            'Please make sure all values are entered properly in the Machine '
            'Performance Table.')
    LOGGER.debug('Machine Performance Rows : %s', machine_perf_dict['periods'])
    LOGGER.debug('Machine Performance Cols : %s', machine_perf_dict['heights'])

    machine_param_dict = read_machine_csv_as_dict(args['machine_param_path'])

    # Check if required column headers are entered in the land grid csv file
    if 'land_gridPts_path' in args:
        # Create a grid_land_data dataframe for later use in valuation
        grid_land_data = pandas.read_csv(args['land_gridPts_path'])
        required_col_names = ['ID', 'TYPE', 'LAT', 'LONG', 'LOCATION']
        grid_land_data.columns = [
            col_name.upper() for col_name in grid_land_data.columns
        ]
        missing_col_names = []
        for col_name in required_col_names:
            if col_name not in grid_land_data.columns:
                missing_col_names.append(col_name)
        if missing_col_names:
            raise ValueError(
                'The following column headers are missing from the Grid '
                'Connection Points File: %s' % missing_col_names)

    # Build up a dictionary of possible analysis areas where the key
    # is the analysis area selected and the value is a dictionary
    # that stores the related paths to the needed inputs
    wave_base_data_path = args['wave_base_data_path']
    analysis_dict = {
        'West Coast of North America and Hawaii': {
            'point_shape':
            os.path.join(wave_base_data_path, 'NAmerica_WestCoast_4m.shp'),
            'extract_shape':
            os.path.join(wave_base_data_path, 'WCNA_extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'NAmerica_WestCoast_4m.txt.bin')
        },
        'East Coast of North America and Puerto Rico': {
            'point_shape':
            os.path.join(wave_base_data_path, 'NAmerica_EastCoast_4m.shp'),
            'extract_shape':
            os.path.join(wave_base_data_path, 'ECNA_extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'NAmerica_EastCoast_4m.txt.bin')
        },
        'North Sea 4 meter resolution': {
            'point_shape':
            os.path.join(wave_base_data_path, 'North_Sea_4m.shp'),
            'extract_shape':
            os.path.join(wave_base_data_path, 'North_Sea_4m_Extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'North_Sea_4m.bin')
        },
        'North Sea 10 meter resolution': {
            'point_shape':
            os.path.join(wave_base_data_path, 'North_Sea_10m.shp'),
            'extract_shape':
            os.path.join(wave_base_data_path, 'North_Sea_10m_Extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'North_Sea_10m.bin')
        },
        'Australia': {
            'point_shape':
            os.path.join(wave_base_data_path, 'Australia_4m.shp'),
            'extract_shape':
            os.path.join(wave_base_data_path, 'Australia_Extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'Australia_4m.bin')
        },
        'Global': {
            'point_shape':
            os.path.join(wave_base_data_path, 'Global.shp'),
            'extract_shape':
            os.path.join(wave_base_data_path, 'Global_extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'Global_WW3.txt.bin')
        }
    }

    # Get the String value for the analysis area provided from the dropdown
    # menu in the user interface
    analysis_area_path = args['analysis_area_path']
    # Use the analysis area String to get the path's to the wave seastate data,
    # the wave point shapefile, and the polygon extract shapefile
    wave_seastate_bins = load_binary_wave_data(
        analysis_dict[analysis_area_path]['ww3_path'])
    analysis_area_points_path = analysis_dict[analysis_area_path][
        'point_shape']
    analysis_area_extract_path = analysis_dict[analysis_area_path][
        'extract_shape']

    # Path for clipped wave point shapefile holding wave attribute information
    clipped_wave_vector_path = os.path.join(
        intermediate_dir, 'WEM_InputOutput_Pts%s.shp' % file_suffix)

    # Final output paths for wave energy and wave power rasters
    wave_energy_raster_path = os.path.join(
        output_dir, 'capwe_mwh%s.tif' % file_suffix)
    wave_power_raster_path = os.path.join(output_dir, 'wp_kw%s.tif' % file_suffix)

    # Paths for wave energy and wave power percentile rasters
    wp_rc_path = os.path.join(output_dir, 'wp_rc%s.tif' % file_suffix)
    capwe_rc_path = os.path.join(output_dir, 'capwe_rc%s.tif' % file_suffix)

    # Set nodata value and target_pixel_type for new rasters
    nodata = float(numpy.finfo(numpy.float32).min) + 1.0
    target_pixel_type = gdal.GDT_Float32

    # Set the source projection for a coordinate transformation
    # to the input projection from the wave watch point shapefile
    analysis_area_sr = get_vector_spatial_ref(analysis_area_points_path)

    # This try/except statement differentiates between having an AOI or doing
    # a broad run on all the wave watch points specified by
    # args['analysis_area'].
    if 'aoi_path' not in args:
        LOGGER.info('AOI not provided')

        # The path to a polygon shapefile that specifies the broader area
        # of interest
        aoi_vector_path = analysis_area_extract_path

        # Make a copy of the wave point shapefile so that the original input is
        # not corrupted
        if os.path.isfile(clipped_wave_vector_path):
            os.remove(clipped_wave_vector_path)
        analysis_area_vector = gdal.OpenEx(analysis_area_points_path,
                                           gdal.OF_VECTOR)
        drv = gdal.GetDriverByName('ESRI Shapefile')
        drv.CreateCopy(clipped_wave_vector_path, analysis_area_vector)

        # Set the pixel size to that of DEM, to be used for creating rasters
        pixel_size = pygeoprocessing.get_raster_info(dem_path)['pixel_size']
        dem_wkt = pygeoprocessing.get_raster_info(dem_path)['projection']
        LOGGER.debug('Pixel size of the DEM : %s\nProjection of the DEM : %s' %
                     (pixel_size, dem_wkt))

        # Create a coordinate transformation, because it is used below when
        # indexing the DEM
        aoi_sr = get_vector_spatial_ref(analysis_area_extract_path)
        coord_trans, coord_trans_opposite = get_coordinate_transformation(
            analysis_area_sr, aoi_sr)
    else:
        LOGGER.info('AOI was provided')
        aoi_vector_path = args['aoi_path']

        # Temporary shapefile path needed for an intermediate step when
        # changing the projection
        proj_wave_vector_path = os.path.join(
            intermediate_dir, 'projected_wave_data%s.shp' % file_suffix)

        # Set the wave data shapefile to the same projection as the
        # area of interest
        aoi_wkt = pygeoprocessing.get_vector_info(aoi_vector_path)[
            'projection']
        pygeoprocessing.reproject_vector(analysis_area_points_path, aoi_wkt,
                                         proj_wave_vector_path)

        # Clip the wave data shapefile by the bounds provided from the AOI
        clip_vector(
            proj_wave_vector_path, aoi_vector_path, clipped_wave_vector_path,
            file_suffix)

        proj_aoi_vector_path = os.path.join(
            intermediate_dir, 'aoi_proj_to_extract%s.shp' % file_suffix)

        # Get the spatial reference of the Extract shape and export to WKT to
        # use in reprojecting the AOI
        extract_wkt = pygeoprocessing.get_vector_info(
            analysis_area_extract_path)['projection']

        # Project AOI to Extract shape
        pygeoprocessing.reproject_vector(aoi_vector_path, extract_wkt,
                                         proj_aoi_vector_path)

        aoi_clipped_to_extract_path = os.path.join(
            intermediate_dir,
            'aoi_clipped_to_extract_path%s.shp' % file_suffix)

        # Clip the AOI to the Extract shape to make sure the output results do
        # not show extrapolated values outside the bounds of the points
        clip_vector_layer(proj_aoi_vector_path, analysis_area_extract_path,
                          aoi_clipped_to_extract_path)

        aoi_clip_proj_vector_path = os.path.join(
            intermediate_dir, 'aoi_clip_proj%s.shp' % file_suffix)

        # # new function to be used
        # clip_vector(
        #     aoi_vector_path, analysis_area_extract_path,
        #     aoi_clipped_to_extract_path, file_suffix)

        # Reproject the clipped AOI back
        pygeoprocessing.reproject_vector(aoi_clipped_to_extract_path, aoi_wkt,
                                         aoi_clip_proj_vector_path)

        aoi_vector_path = aoi_clip_proj_vector_path

        # Create a coordinate transformation from the given
        # WWIII point shapefile, to the area of interest's projection
        aoi_sr = get_vector_spatial_ref(aoi_vector_path)
        coord_trans, coord_trans_opposite = get_coordinate_transformation(
            analysis_area_sr, aoi_sr)

        # Get the size of the pixels in meters, to be used for creating
        # projected wave power and wave energy capacity rasters
        pixel_size = pixel_size_helper(clipped_wave_vector_path, coord_trans,
                                       coord_trans_opposite, dem_path)

        # Average the pixel sizes in case they are of different sizes
        LOGGER.debug('Pixel size of DEM in meters : %s', pixel_size)

    # We do all wave power calculations by manipulating the fields in
    # the wave data shapefile, thus we need to add proper depth values
    # from the raster DEM
    LOGGER.info('Adding a depth field to the shapefile from the DEM raster')

    def index_values_to_points(base_point_vector_path, base_raster_path,
                               field_name, coord_trans):
        """Add the values of a raster to the field of vector point features.

        Values are recorded in the attribute field of the vector. If a value is
        larger than or equal to 0, the feature will be deleted.

        Parameters:
            base_point_vector_path (str): a path to an ogr point shapefile
            base_raster_path(str): a path to a GDAL dataset
            field_name (str): the name of the new field that will be
                added to the point feature
            coord_trans (osr.CoordinateTransformation): a coordinate
                transformation used to make sure the vector and raster are
                in the same unit

        Returns:
            None

        """
        base_vector = gdal.OpenEx(base_point_vector_path, 1)
        raster_gt = pygeoprocessing.get_raster_info(base_raster_path)[
            'geotransform']

        # Create a new field for the raster attribute
        field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)

        base_layer = base_vector.GetLayer()
        base_layer.CreateField(field_defn)
        feature = base_layer.GetNextFeature()

        for _, block_matrix in pygeoprocessing.iterblocks(base_raster_path):
            # For all the features (points) add the proper raster value
            while feature is not None:
                field_index = feature.GetFieldIndex(field_name)
                geom = feature.GetGeometryRef()
                geom_x, geom_y = geom.GetX(), geom.GetY()

                # Transform two points into meters
                point_decimal_degree = coord_trans.TransformPoint(
                    geom_x, geom_y)

                # To get proper raster value we must index into the dem matrix
                # by getting where the point is located in terms of the matrix
                i = int(
                    (point_decimal_degree[0] - raster_gt[0]) / raster_gt[1])
                j = int(
                    (point_decimal_degree[1] - raster_gt[3]) / raster_gt[5])
                raster_value = block_matrix[j][i]
                # There are cases where the DEM may be too coarse and thus a
                # wave energy point falls on land. If the raster value taken is
                # greater than or equal to zero we need to delete that point as
                # it should not be used in calculations
                if raster_value >= 0.0:
                    base_layer.DeleteFeature(feature.GetFID())
                else:
                    feature.SetField(int(field_index), float(raster_value))
                    base_layer.SetFeature(feature)
                feature = base_layer.GetNextFeature()

        # It is not enough to just delete a feature from the layer. The
        # database where the information is stored must be re-packed so that
        # feature entry is properly removed
        base_vector.ExecuteSQL('REPACK ' + base_layer.GetName())

    # Add the depth value to the wave points by indexing into the DEM dataset
    index_values_to_points(clipped_wave_vector_path, dem_path, 'DEPTH_M',
                           coord_trans_opposite)
    LOGGER.info('Finished adding DEPTH_M field to wave shapefile from DEM '
                'raster.')

    # Generate an interpolate object for wave_energy_capacity
    LOGGER.info('Interpolating machine performance table.')
    energy_interp = wave_energy_interp(wave_seastate_bins, machine_perf_dict)

    # Create a dictionary with the wave energy capacity sums from each location
    LOGGER.info('Calculating Captured Wave Energy.')
    energy_cap = compute_wave_energy_capacity(
        wave_seastate_bins, energy_interp, machine_param_dict)

    # Add the sum as a field to the shapefile for the corresponding points
    LOGGER.info('Adding the wave energy sums to the WaveData shapefile')
    captured_wave_energy_to_shape(energy_cap, clipped_wave_vector_path)

    # Calculate wave power for each wave point and add it as a field
    # to the shapefile
    LOGGER.info('Calculating Wave Power.')
    compute_wave_power(clipped_wave_vector_path)

    # Create blank rasters bounded by the shape file of analysis area
    pygeoprocessing.create_raster_from_vector_extents(
        aoi_vector_path, wave_energy_raster_path, pixel_size,
        target_pixel_type, nodata)

    pygeoprocessing.create_raster_from_vector_extents(
        aoi_vector_path, wave_power_raster_path, pixel_size, target_pixel_type,
        nodata)

    # Interpolate wave energy and wave power from the shapefile over the rasters
    LOGGER.info('Interpolate wave power and wave energy capacity onto rasters')

    pygeoprocessing.interpolate_points(clipped_wave_vector_path, 'CAPWE_MWHY',
                                       (wave_energy_raster_path, 1), 'near')

    pygeoprocessing.interpolate_points(clipped_wave_vector_path, 'WE_kWM',
                                       (wave_power_raster_path, 1), 'near')

    # Create the percentile rasters for wave energy and wave power
    # These values are hard coded in because it's specified explicitly in
    # the user's guide what the percentile ranges are and what the units
    # will be.
    percentiles = [25, 50, 75, 90]
    capwe_units_short = ' MWh/yr'
    capwe_units_long = 'megawatt hours per year'
    wp_units_short = ' kW/m'
    wp_units_long = 'wave power per unit width of wave crest length'
    starting_percentile_range = '1'

    create_percentile_rasters(
        wave_energy_raster_path, capwe_rc_path, capwe_units_short, capwe_units_long,
        starting_percentile_range, percentiles, aoi_vector_path)

    create_percentile_rasters(wave_power_raster_path, wp_rc_path, wp_units_short,
                              wp_units_long, starting_percentile_range,
                              percentiles, aoi_vector_path)

    LOGGER.info('Completed Wave Energy Biophysical')

    if 'valuation_container' not in args:
        LOGGER.info('Valuation not selected')
        # The rest of the function is valuation, so we can quit now
        return
    else:
        LOGGER.info('Valuation selected')

    # Output path for landing point shapefile
    land_vector_path = os.path.join(output_dir, 'LandPts_prj%s.shp' % file_suffix)
    # Output path for grid point shapefile
    grid_vector_path = os.path.join(output_dir, 'GridPts_prj%s.shp' % file_suffix)
    # Output path for the projected net present value raster
    npv_proj_path = os.path.join(intermediate_dir,
                                 'npv_not_clipped%s.tif' % file_suffix)
    # Path for the net present value percentile raster
    npv_rc_path = os.path.join(output_dir, 'npv_rc%s.tif' % file_suffix)

    machine_econ_dict = read_machine_csv_as_dict(args['machine_econ_path'])

    # Number of machines for a given wave farm
    units = int(args['number_of_machines'])
    # Extract the machine economic parameters
    cap_max = float(machine_econ_dict['capmax'])
    capital_cost = float(machine_econ_dict['cc'])
    cml = float(machine_econ_dict['cml'])
    cul = float(machine_econ_dict['cul'])
    col = float(machine_econ_dict['col'])
    omc = float(machine_econ_dict['omc'])
    price = float(machine_econ_dict['p'])
    drate = float(machine_econ_dict['r'])
    smlpm = float(machine_econ_dict['smlpm'])

    # The NPV is for a 25 year period
    year = 25

    # A numpy array of length 25, representing the npv of a farm for each year
    time = numpy.linspace(0, year - 1, year)

    # The discount rate calculation for the npv equations
    rho = 1.0 / (1.0 + drate)

    grid_data = grid_land_data.loc[grid_land_data['TYPE'].str.lower() ==
                                   'grid']
    land_data = grid_land_data.loc[grid_land_data['TYPE'].str.lower() ==
                                   'land']

    grid_dict = grid_data.to_dict('index')
    land_dict = land_data.to_dict('index')

    # Make a point shapefile for grid points
    LOGGER.info('Creating Grid Points Shapefile.')
    dict_to_point_vector(
        grid_dict, grid_vector_path, 'grid_points', aoi_sr, coord_trans)

    # Make a point shapefile for landing points.
    LOGGER.info('Creating Landing Points Shapefile.')
    dict_to_point_vector(
        land_dict, land_vector_path, 'land_points', aoi_sr, coord_trans)

    # Get the coordinates of points of wave, land, and grid vectors
    wave_points = get_points_geometries(clipped_wave_vector_path)
    land_points = get_points_geometries(land_vector_path)
    grid_points = get_points_geometries(grid_vector_path)

    # Calculate the minimum distances between the relative point groups
    LOGGER.info('Calculating Min Distances.')
    wave_to_land_dist, wave_to_land_id = calculate_min_distances(
        wave_points, land_points)
    land_to_grid_dist, _ = calculate_min_distances(land_points, grid_points)

    def add_distance_fields_path(wave_shape_path, ocean_to_land_dist,
                                 land_to_grid_dist):
        """A wrapper function that adds two fields to the wave point shapefile:
        the distance from ocean to land and the distance from land to grid.

        Parameters:
            wave_shape_path (str): a path to the wave points shapefile
            ocean_to_land_dist (numpy.array): an array of distance values
            land_to_grid_dist (numpy.array): an array of distance values

        Returns:
            None

        """
        wave_data_shape = gdal.OpenEx(wave_shape_path, 1)
        wave_data_layer = wave_data_shape.GetLayer(0)
        # Add three new fields to the shapefile that will store the distances
        for field in ['W2L_MDIST', 'LAND_ID', 'L2G_MDIST']:
            field_defn = ogr.FieldDefn(field, ogr.OFTReal)
            field_defn.SetWidth(24)
            field_defn.SetPrecision(11)
            wave_data_layer.CreateField(field_defn)

        # For each feature in the shapefile add the corresponding distance
        # from wave_to_land_dist and land_to_grid_dist calculated above
        iterate_feat = 0
        wave_data_layer.ResetReading()
        feature = wave_data_layer.GetNextFeature()
        while feature is not None:
            wave_to_land_index = feature.GetFieldIndex('W2L_MDIST')
            land_to_grid_index = feature.GetFieldIndex('L2G_MDIST')
            id_index = feature.GetFieldIndex('LAND_ID')

            land_id = int(wave_to_land_id[iterate_feat])

            feature.SetField(wave_to_land_index,
                             ocean_to_land_dist[iterate_feat])
            feature.SetField(land_to_grid_index, land_to_grid_dist[land_id])
            feature.SetField(id_index, land_id)

            iterate_feat = iterate_feat + 1

            wave_data_layer.SetFeature(feature)
            feature = None
            feature = wave_data_layer.GetNextFeature()

    add_distance_fields_path(clipped_wave_vector_path, wave_to_land_dist,
                             land_to_grid_dist)

    def npv_wave(annual_revenue, annual_cost):
        """Calculates the NPV for a wave farm site based on the annual revenue
        and annual cost

        Parameters:
            annual_revenue (numpy.array): an array of the annual revenue for
                the first 25 years
            annual_cost (numpy.array): an array of the annual cost for the
                first 25 years

        Returns: The Total NPV which is the sum of all 25 years

        """
        npv = []
        for i in range(len(time)):
            npv.append(rho**i * (annual_revenue[i] - annual_cost[i]))
        return sum(npv)

    def compute_npv_farm_energy_path(wave_points_path):
        """A wrapper function to compute and add the Net Present Value to vector.

        Also computes the total captured wave energy for the entire farm.

        Parameters:
            wave_points_path (str): a path to the wave energy points with
                fields for calculating NPV.

        Returns:
            None

        """

        wave_points = gdal.OpenEx(wave_points_path, 1)
        wave_data_layer = wave_points.GetLayer()
        # Add Net Present Value field, Total Captured Wave Energy field, and
        # Units field to shapefile
        for field_name in ['NPV_25Y', 'CAPWE_ALL', 'UNITS']:
            field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
            field_defn.SetWidth(24)
            field_defn.SetPrecision(11)
            wave_data_layer.CreateField(field_defn)
        wave_data_layer.ResetReading()
        feat_npv = wave_data_layer.GetNextFeature()

        # For all the wave farm sites, calculate npv and write to shapefile
        LOGGER.info('Calculating the Net Present Value.')
        while feat_npv is not None:
            depth_index = feat_npv.GetFieldIndex('DEPTH_M')
            wave_to_land_index = feat_npv.GetFieldIndex('W2L_MDIST')
            land_to_grid_index = feat_npv.GetFieldIndex('L2G_MDIST')
            captured_wave_energy_index = feat_npv.GetFieldIndex('CAPWE_MWHY')
            npv_index = feat_npv.GetFieldIndex('NPV_25Y')
            capwe_all_index = feat_npv.GetFieldIndex('CAPWE_ALL')
            units_index = feat_npv.GetFieldIndex('UNITS')

            depth = feat_npv.GetFieldAsDouble(depth_index)
            wave_to_land = feat_npv.GetFieldAsDouble(wave_to_land_index)
            land_to_grid = feat_npv.GetFieldAsDouble(land_to_grid_index)
            captured_wave_energy = feat_npv.GetFieldAsDouble(
                captured_wave_energy_index)
            capwe_all_result = captured_wave_energy * units

            # Create a numpy array of length 25, filled with the captured wave
            # energy in kW/h. Represents the lifetime of this wave farm.
            captured_we = numpy.ones(
                len(time)) * (int(captured_wave_energy) * 1000.0)
            # It is expected that there is no revenue from the first year
            captured_we[0] = 0

            # Compute values to determine NPV
            lenml = 3.0 * numpy.absolute(depth)
            install_cost = units * cap_max * capital_cost
            mooring_cost = smlpm * lenml * cml * units
            trans_cost = (wave_to_land * cul / 1000.0) + (
                land_to_grid * col / 1000.0)
            initial_cost = install_cost + mooring_cost + trans_cost
            annual_revenue = price * units * captured_we
            annual_cost = omc * captured_we * units

            # The first year's costs are the initial start up costs
            annual_cost[0] = initial_cost

            npv_result = npv_wave(annual_revenue, annual_cost) / 1000.0
            feat_npv.SetField(npv_index, npv_result)
            feat_npv.SetField(capwe_all_index, capwe_all_result)
            feat_npv.SetField(units_index, units)

            wave_data_layer.SetFeature(feat_npv)
            feat_npv = None
            feat_npv = wave_data_layer.GetNextFeature()

    compute_npv_farm_energy_path(clipped_wave_vector_path)

    # Create a blank raster from the extents of the wave farm shapefile
    LOGGER.info('Creating Raster From Vector Extents')
    pygeoprocessing.create_raster_from_vector_extents(
        clipped_wave_vector_path, npv_proj_path, pixel_size, target_pixel_type,
        nodata)
    LOGGER.info('Completed Creating Raster From Vector Extents')

    # Interpolate the NPV values based on the dimensions and corresponding
    # points of the raster, then write the interpolated values to the raster
    LOGGER.info('Generating Net Present Value Raster.')
    pygeoprocessing.interpolate_points(clipped_wave_vector_path, 'NPV_25Y',
                                       (npv_proj_path, 1), 'near')

    npv_out_path = os.path.join(output_dir, 'npv_usd%s.tif' % file_suffix)

    # Clip the raster to the convex hull polygon
    clip_to_projected_coordinate_system(npv_proj_path, aoi_vector_path,
                                        npv_out_path)

    # Create the percentile raster for net present value
    percentiles = [25, 50, 75, 90]

    create_percentile_rasters(npv_out_path, npv_rc_path, ' (US$)',
                              ' thousands of US dollars (US$)', '1',
                              percentiles, aoi_vector_path)

    LOGGER.info('End of Wave Energy Valuation.')


def clip_to_projected_coordinate_system(base_raster_path, clip_vector_path,
                                        target_raster_path):
    """Clip raster with vector into projected coordinate system.

    If base raster is not already projected, choose a suitable UTM zone.

    Parameters:
        base_raster_path (string): path to base raster.
        clip_vector_path (string): path to base clip vector.
        target_raster_path (string): path to output clipped raster.

    Returns:
        None.

    """
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    clip_vector_info = pygeoprocessing.get_vector_info(clip_vector_path)

    base_raster_srs = osr.SpatialReference()
    base_raster_srs.ImportFromWkt(base_raster_info['projection'])

    if not base_raster_srs.IsProjected():
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)
        clip_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
            clip_vector_info['bounding_box'], clip_vector_info['projection'],
            wgs84_sr.ExportToWkt())

        base_raster_bounding_box = pygeoprocessing.transform_bounding_box(
            base_raster_info['bounding_box'], base_raster_info['projection'],
            wgs84_sr.ExportToWkt())

        target_bounding_box_wgs84 = pygeoprocessing._merge_bounding_boxes(
            clip_wgs84_bounding_box, base_raster_bounding_box, 'intersection')

        clip_vector_srs = osr.SpatialReference()
        clip_vector_srs.ImportFromWkt(clip_vector_info['projection'])

        centroid_x = (
            target_bounding_box_wgs84[2] + target_bounding_box_wgs84[0]) / 2
        centroid_y = (
            target_bounding_box_wgs84[3] + target_bounding_box_wgs84[1]) / 2
        utm_code = (math.floor((centroid_x + 180) / 6) % 60) + 1
        utm_code = (math.floor((centroid_x + 180.0) / 6) % 60) + 1
        lat_code = 6 if centroid_y > 0 else 7
        epsg_code = int('32%d%02d' % (lat_code, utm_code))
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(epsg_code)
        target_bounding_box = pygeoprocessing.transform_bounding_box(
            target_bounding_box_wgs84, wgs84_sr.ExportToWkt(),
            target_srs.ExportToWkt())

        target_pixel_size = convert_degree_pixel_size_to_meters(
            base_raster_info['pixel_size'], centroid_y)

        pygeoprocessing.warp_raster(
            base_raster_path,
            target_pixel_size,
            target_raster_path,
            None,
            target_bb=target_bounding_box,
            target_sr_wkt=target_srs.ExportToWkt())
    else:
        pygeoprocessing.align_and_resize_raster_stack(
            [base_raster_path], [target_raster_path], ['near'],
            base_raster_info['pixel_size'],
            'intersection',
            base_vector_path_list=[clip_vector_path],
            target_sr_wkt=base_raster_info['projection'])


def convert_degree_pixel_size_to_meters(pixel_size, center_lat):
    """Calculate meter size of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (tuple): [xsize, ysize] in degrees (float).
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        `pixel_size` in meters.

    """
    m1 = 111132.92
    m2 = -559.82
    m3 = 1.175
    m4 = -0.0023
    p1 = 111412.84
    p2 = -93.5
    p3 = 0.118

    lat = center_lat * math.pi / 180
    latlen = (m1 + m2 * math.cos(2 * lat) + m3 * math.cos(4 * lat) +
              m4 * math.cos(6 * lat))
    longlen = abs(p1 * math.cos(lat) + p2 * math.cos(3 * lat) +
                  p3 * math.cos(5 * lat))

    return (longlen * pixel_size[0], latlen * pixel_size[1])


def dict_to_point_vector(base_dict_data, target_vector_path, layer_name,
                         target_sr, coord_trans):
    """Given a dictionary of data create a point shapefile that represents it.

    Parameters:
        base_dict_data (dict): a  dictionary with the wind data, where the keys
            are tuples of the lat/long coordinates:
            {
            1 : {'TYPE':'GRID', 'LAT':49, 'LONG':-126, 'LOCATION':'Ucluelet'},
            2 : {'TYPE':'GRID', 'LAT':50, 'LONG':-127, 'LOCATION':'Tofino'},
            }
        layer_name (str): the name of the layer.
        target_vector_path (str): path to the output destination of the
            shapefile.
        target_sr (str): target spatial reference in well-known text format
        coord_trans (OGRCoordinateTransformation): a coordinate transformation
            from source to target spatial reference

    Returns:
        None

    """
    # If the target_vector_path exists delete it
    if os.path.isfile(target_vector_path):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.DeleteDataSource(target_vector_path)

    LOGGER.info('Creating new vector')
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(target_vector_path)
    target_layer = output_datasource.CreateLayer(layer_name, target_sr,
                                                 ogr.wkbPoint)

    # Construct a dictionary of field names and their corresponding types
    field_dict = {
        'ID': ogr.OFTInteger,
        'TYPE': ogr.OFTString,
        'LAT': ogr.OFTReal,
        'LONG': ogr.OFTReal,
        'LOCATION': ogr.OFTString
    }

    LOGGER.info('Creating fields for the vector')
    for field_name in ['ID', 'TYPE', 'LAT', 'LONG', 'LOCATION']:
        target_field = ogr.FieldDefn(field_name, field_dict[field_name])
        target_layer.CreateField(target_field)

    LOGGER.info('Entering iteration to create and set the features')
    # For each inner dictionary (for each point) create a point
    for point_dict in base_dict_data.itervalues():
        latitude = float(point_dict['LAT'])
        longitude = float(point_dict['LONG'])
        # When projecting to WGS84, extents -180 to 180 are used for longitude.
        # In case input longitude is from -360 to 0 convert
        if longitude < -180:
            longitude += 360
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint_2D(longitude, latitude)
        geom.Transform(coord_trans)

        output_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_layer.CreateFeature(output_feature)

        for field_name in point_dict:
            field_index = output_feature.GetFieldIndex(field_name)
            output_feature.SetField(field_index, point_dict[field_name])
        output_feature.SetGeometryDirectly(geom)
        target_layer.SetFeature(output_feature)
        output_feature = None

    output_datasource = None
    LOGGER.info('Finished dict_to_point_vector')


def get_points_geometries(base_vector_path):
    """This function retrieves the XY coordinates from a point shapefile.

    The X and Y values from each point feature in the vector are stored in pair
    as [x_location,y_location] in a numpy array.

    Parameters:
        base_vector_path (str): a path to an OGR shapefile

    Returns:
        points (numpy.array): an array of points, which represent the geometry
            of each point in the shapefile.

    """
    points = []
    base_vector = gdal.OpenEx(base_vector_path)
    base_layer = base_vector.GetLayer(0)
    feat = base_layer.GetNextFeature()
    while feat is not None:
        x_location = float(feat.GetGeometryRef().GetX())
        y_location = float(feat.GetGeometryRef().GetY())
        points.append([x_location, y_location])
        feat = None
        feat = base_layer.GetNextFeature()

    return numpy.array(points)


def calculate_min_distances(xy_1, xy_2):
    """For all points in xy_1, this function calculates the distance from point
    xy_1 to various points in xy_2, and stores the shortest distances found in
    a list min_dist. The function also stores the index from which ever point
    in xy_2 was closest, as an id in a list that corresponds to min_dist.

    Parameters:
        xy_1 (numpy.array): An array of points in the form [x,y]
        xy_2 (numpy.array): An array of points in the form [x,y]

    Returns:
        A numpy array of shortest distances and a numpy array of indexes
        corresponding to the array of shortest distances

    """
    # Create two numpy array of zeros with length set to as many points in xy_1
    min_dist = numpy.zeros(len(xy_1))
    min_id = numpy.zeros(len(xy_1))

    # For all points xy_point in xy_1 calculate the distance from xy_point to
    # xy_2 and save the shortest distance found.
    for index, xy_point in enumerate(xy_1):
        dists = numpy.sqrt(numpy.sum((xy_point - xy_2)**2, axis=1))
        min_dist[index], min_id[index] = dists.min(), dists.argmin()
    return min_dist, min_id


def load_binary_wave_data(wave_file_path):
    """The load_binary_wave_data function converts a pickled WW3 text file
        into a dictionary who's keys are the corresponding (I,J) values
        and whose value is a two-dimensional array representing a matrix
        of the number of hours a seastate occurs over a 5 year period.
        The row and column headers are extracted once and stored in the
        dictionary as well.

        wave_file_path - The path to a pickled binary WW3 file.

        returns - A dictionary of matrices representing hours of specific
              seastates, as well as the period and height ranges.
              It has the following structure:
               {'periods': [1,2,3,4,...],
                'heights': [.5,1.0,1.5,...],
                'bin_matrix': { (i0,j0): [[2,5,3,2,...], [6,3,4,1,...],...],
                                (i1,j1): [[2,5,3,2,...], [6,3,4,1,...],...],
                                 ...
                                (in, jn): [[2,5,3,2,...], [6,3,4,1,...],...]
                              }
               }
    """
    LOGGER.info('Extrapolating wave data from text to a dictionary')
    wave_file = open(wave_file_path, 'rb')
    wave_dict = {}
    # Create a key that hosts another dictionary where the matrix representation
    # of the seastate bins will be saved
    wave_dict['bin_matrix'] = {}
    wave_array = None
    wave_periods = []
    wave_heights = []
    key = None

    # get rows,cols
    row_col_bin = wave_file.read(8)
    n_cols, n_rows = struct.unpack('ii', row_col_bin)

    # get the periods and heights
    line = wave_file.read(n_cols * 4)

    wave_periods = list(struct.unpack('f' * n_cols, line))
    line = wave_file.read(n_rows * 4)
    wave_heights = list(struct.unpack('f' * n_rows, line))

    key = None
    while True:
        line = wave_file.read(8)
        if len(line) == 0:
            # end of file
            wave_dict['bin_matrix'][key] = numpy.array(wave_array)
            break

        if key != None:
            wave_dict['bin_matrix'][key] = numpy.array(wave_array)

        # Clear out array
        wave_array = []

        key = struct.unpack('ii', line)

        for _ in itertools.repeat(None, n_rows):
            line = wave_file.read(n_cols * 4)
            array = list(struct.unpack('f' * n_cols, line))
            wave_array.append(array)

    wave_file.close()
    # Add row/col header to dictionary
    LOGGER.debug('WaveData col %s', wave_periods)
    wave_dict['periods'] = numpy.array(wave_periods, dtype='f')
    LOGGER.debug('WaveData row %s', wave_heights)
    wave_dict['heights'] = numpy.array(wave_heights, dtype='f')
    LOGGER.info('Finished extrapolating wave data to dictionary.')
    return wave_dict


def read_machine_csv_as_dict(machine_csv_path):
    """Create a dictionary whose keys are the 'NAME' from the machine
    CSV table and whose values are from the corresponding 'VALUE' field.

    Parameters:
        machine_csv_path (str): path to the input machine CSV file.

    Returns:
        None.

    """
    machine_dict = {}
    machine_data = pandas.read_csv(machine_csv_path, index_col=0)
    machine_data.columns = machine_data.columns.str.lower()
    if 'value' not in machine_data.columns:
        raise ValueError('Please make sure that the "VALUE" column is in the '
                         'Machine Parameters Table.')
    # remove underscore from the keys and make them lowercased
    machine_data.index = machine_data.index.str.strip()
    machine_data.index = machine_data.index.str.lower()
    machine_dict = machine_data.to_dict('index')
    for key in machine_dict.keys():
        machine_dict[key] = machine_dict[key]['value']
    return machine_dict


def pixel_size_helper(shape_path, coord_trans, coord_trans_opposite, ds_path):
    """This function helps retrieve the pixel sizes of the global DEM
        when given an area of interest that has a certain projection.

        shape_path - A path to a point shapefile datasource indicating where
            in the world we are interested in
        coord_trans - A coordinate transformation
        coord_trans_opposite - A coordinate transformation that transforms in
                           the opposite direction of 'coord_trans'
        ds_path - A path to a gdal dataset to get the pixel size from

        returns - A tuple of the x and y pixel sizes of the global DEM
              given in the units of what 'shape' is projected in"""
    shape = gdal.OpenEx(shape_path)

    # Get a point in the clipped shape to determine output grid size
    feat = shape.GetLayer(0).GetNextFeature()
    geom = feat.GetGeometryRef()
    reference_point_x = geom.GetX()
    reference_point_y = geom.GetY()

    # Convert the point from meters to geom_x/long
    reference_point_latlng = coord_trans_opposite.TransformPoint(
        reference_point_x, reference_point_y)

    # Get the size of the pixels in meters, to be used for creating rasters
    pixel_xsize, pixel_ysize = pixel_size_based_on_coordinate_transform(
        ds_path, coord_trans, reference_point_latlng)

    # Average the pixel sizes incase they are of different sizes
    mean_pixel_size = (abs(pixel_xsize) + abs(pixel_ysize)) / 2.0
    pixel_size_tuple = (mean_pixel_size * numpy.sign(pixel_xsize),
                        mean_pixel_size * numpy.sign(pixel_ysize))

    return pixel_size_tuple


def get_vector_spatial_ref(base_vector_path):
    """Get the spatial reference of an OGR vector (datasource).

    Parameters:
        base_vector_path (str): a path to an ogr vector

    Returns:
        spat_ref: a spatial reference

    """
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer(0)
    spat_ref = layer.GetSpatialRef()
    layer = None
    vector = None
    return spat_ref


def get_coordinate_transformation(source_sr, target_sr):
    """This function takes a source and target spatial reference and creates
        a coordinate transformation from source to target, and one from target
        to source.

    Parameters:
        source_sr: A spatial reference
        target_sr: A spatial reference

    Returns:
        A tuple: coord_trans (source to target) and coord_trans_opposite
        (target to source)

    """
    coord_trans = osr.CoordinateTransformation(source_sr, target_sr)
    coord_trans_opposite = osr.CoordinateTransformation(target_sr, source_sr)
    return (coord_trans, coord_trans_opposite)


def create_percentile_rasters(base_raster_path, target_raster_path,
                              units_short, units_long, start_value,
                              percentile_list, aoi_vector_path):
    """Creates a percentile (quartile) raster based on the raster_dataset. An
        attribute table is also constructed for the raster_dataset that
        displays the ranges provided by taking the quartile of values.
        The following inputs are required:

    Parameters:
        base_raster_path (str): path to a GDAL raster with data of type
            integer
        target_raster_path (str): path to the destination of the new raster.
        units_short (str): The shorthand for the units of the raster values,
            ex: kW/m.
        units_long (str): The description of the units of the raster values,
            ex: wave power per unit width of wave crest length (kW/m).
        start_value (str): The first value that goes to the first percentile
            range (start_value: percentile_one)
        percentile_list (list): A list of the percentiles ranges,
            ex: [25, 50, 75, 90].
        aoi_vector_path (path): path to an OGR polygon shapefile to clip the
            rasters to

    Returns:
        None

    """
    LOGGER.info('Creating Percentile Rasters')

    # If the target_raster_path is already a file, delete it
    if os.path.isfile(target_raster_path):
        os.remove(target_raster_path)

    # Set nodata to a very small negative number
    target_nodata = -99999
    base_nodata = pygeoprocessing.get_raster_info(base_raster_path)['nodata'][
        0]

    # Get the percentile values for each percentile
    percentile_values = calculate_percentiles_from_raster(
        base_raster_path, percentile_list)

    # Get the percentile ranges as strings so that they can be added to an
    # output table
    value_ranges = create_value_ranges(percentile_values, start_value)

    def raster_percentile(band):
        """Operation to use in raster_calculator that takes the pixels of
            band and groups them together based on their percentile ranges.
        """
        return numpy.where(
            band != base_nodata,
            numpy.searchsorted(percentile_values, band) + 1,  # starts from 1
            target_nodata)

    # Classify the pixels of raster_dataset into groups and write to output
    pygeoprocessing.raster_calculator([(base_raster_path, 1)],
                                      raster_percentile, target_raster_path,
                                      gdal.GDT_Int32, target_nodata)

    # Create percentile groups of how percentile ranges are classified
    # using numpy.searchsorted
    percentile_groups = numpy.arange(1, len(percentile_values) + 2)

    # Get the pixel count for each group
    pixel_count = count_pixels_groups(target_raster_path, percentile_groups)

    LOGGER.debug('Pixel_count: %s; Percentile_groups: %s' %
                 (pixel_count, percentile_groups))

    # Initialize a dictionary where percentile groups map to a string
    # of corresponding percentile ranges. Used to create RAT
    percentile_dict = {}
    for index in xrange(len(percentile_groups)):
        percentile_dict[percentile_groups[index]] = value_ranges[index]
    value_range_header = 'Value Range (' + units_long + ',' + units_short + ')'
    _create_raster_attr_table(
        target_raster_path, percentile_dict, column_name=value_range_header)

    # Create a list of corresponding percentile ranges from the percentile list
    percentile_ranges = create_percentile_ranges(percentile_list)

    # Initialize a dictionary to map percentile groups to percentile range
    # string and pixel count. Used for creating CSV table
    table_dict = {}
    for index in xrange(len(percentile_groups)):
        table_dict[index] = {}
        table_dict[index]['Percentile Group'] = percentile_groups[index]
        table_dict[index]['Percentile Range'] = percentile_ranges[index]
        table_dict[index][value_range_header] = value_ranges[index]
        table_dict[index]['Pixel Count'] = pixel_count[index]

    attribute_table_path = target_raster_path[:-4] + '.csv'
    column_names = [
        'Percentile Group', 'Percentile Range', value_range_header,
        'Pixel Count'
    ]
    create_attribute_csv_table(attribute_table_path, column_names, table_dict)


def create_value_ranges(percentiles, start_value):
    """Constructs the value ranges as Strings, with the first range starting
    at 1 and the last range being greater than the last percentile mark. Each
    string range is stored in a list that gets returned.

    Parameters:
        percentiles (list): A list of the percentile marks in ascending order
        start_value (str): the first value that goes to the first percentile
            range (start_value: percentile_one)

    Returns:
        A list of Strings representing the ranges of the percentile values

    """
    length = len(percentiles)
    range_values = []
    # Add the first range with the starting value and long description of units
    # This function will fail and cause an error if the percentile list is empty
    range_first = start_value + ' - ' + str(percentiles[0])
    range_values.append(range_first)
    for index in range(length - 1):
        range_values.append(
            str(percentiles[index]) + ' - ' + str(percentiles[index + 1]))
    # Add the last range to the range of values list
    range_last = 'Greater than ' + str(percentiles[length - 1])
    range_values.append(range_last)
    LOGGER.debug('Range_values : %s', range_values)
    return range_values


def create_percentile_ranges(percentile_list):
    """Constructs the percentile ranges as Strings.

    Each string range is stored in a list that gets returned.

    Parameters:
        percentile_list (list): A list of percentiles in ascending order

    Returns:
        A list of Strings representing the ranges of the percentile values

    """
    length = len(percentile_list)
    percentile_ranges = []
    first_range = '<' + str(percentile_list[0]) + '%'
    percentile_ranges.append(first_range)
    for index in range(length - 1):
        percentile_ranges.append(
            str(percentile_list[index]) + '-' +
            str(percentile_list[index + 1]) + '%')

    # Add the last range to the percentile ranges list
    last_range = '>' + str(percentile_list[length - 1]) + '%'
    percentile_ranges.append(last_range)
    return percentile_ranges


def create_attribute_csv_table(attribute_table_path, fields, data):
    """Create a new csv table from a dictionary

        filename - a path for the new table to be written to disk

        fields - a python list of the column names. The order of the fields in
            the list will be the order in how they are written. ex:
            ['id', 'precip', 'total']

        data - a python dictionary representing the table. The dictionary
            should be constructed with unique numerical keys that point to a
            dictionary which represents a row in the table:
            data = {0 : {'id':1, 'precip':43, 'total': 65},
                    1 : {'id':2, 'precip':65, 'total': 94}}

        returns - nothing
    """
    if os.path.isfile(attribute_table_path):
        os.remove(attribute_table_path)

    csv_file = open(attribute_table_path, 'wb')

    #  Sort the keys so that the rows are written in order
    row_keys = data.keys()
    row_keys.sort()

    csv_writer = csv.DictWriter(csv_file, fields)
    #  Write the columns as the first row in the table
    csv_writer.writerow(dict((fn, fn) for fn in fields))

    # Write the rows from the dictionary
    for index in row_keys:
        csv_writer.writerow(data[index])

    csv_file.close()


def compute_wave_power(shape_path):
    """Calculates the wave power from the fields in the shapefile
        and writes the wave power value to a field for the corresponding
        feature.

    Parameters:
        shape_path (str): A path to a shapefile that has all the attributes
            represented in fields to calculate wave power at a specific
            wave farm

    Returns:
        None

    """
    shape = gdal.OpenEx(shape_path, 1)

    # Sea water density constant (kg/m^3)
    swd = 1028
    # Gravitational acceleration (m/s^2)
    grav = 9.8
    # Constant determining the shape of a wave spectrum (see users guide pg 23)
    alfa = 0.86
    # Add a waver power field to the shapefile.
    layer = shape.GetLayer()
    field_defn = ogr.FieldDefn('WE_kWM', ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)
    layer.ResetReading()
    feat = layer.GetNextFeature()
    # For every feature (point) calculate the wave power and add the value
    # to itself in a new field
    while feat is not None:
        height_index = feat.GetFieldIndex('HSAVG_M')
        period_index = feat.GetFieldIndex('TPAVG_S')
        depth_index = feat.GetFieldIndex('DEPTH_M')
        wp_index = feat.GetFieldIndex('WE_kWM')
        height = feat.GetFieldAsDouble(height_index)
        period = feat.GetFieldAsDouble(period_index)
        depth = feat.GetFieldAsInteger(depth_index)

        depth = numpy.absolute(depth)
        # wave frequency calculation (used to calculate wave number k)
        tem = (2.0 * math.pi) / (period * alfa)
        # wave number calculation (expressed as a function of
        # wave frequency and water depth)
        k = numpy.square(tem) / (grav * numpy.sqrt(
            numpy.tanh((numpy.square(tem)) * (depth / grav))))
        # Setting numpy overlow error to ignore because when numpy.sinh
        # gets a really large number it pushes a warning, but Rich
        # and Doug have agreed it's nothing we need to worry about.
        numpy.seterr(over='ignore')

        # wave group velocity calculation (expressed as a
        # function of wave energy period and water depth)
        wave_group_velocity = (((1 + (
            (2 * k * depth) / numpy.sinh(2 * k * depth))) * numpy.sqrt(
                (grav / k) * numpy.tanh(k * depth))) / 2)

        # Reset the overflow error to print future warnings
        numpy.seterr(over='print')

        # wave power calculation
        wave_pow = ((((swd * grav) / 16) *
                     (numpy.square(height)) * wave_group_velocity) / 1000)

        feat.SetField(wp_index, wave_pow)
        layer.SetFeature(feat)
        feat = layer.GetNextFeature()


def clip_vector(base_vector_path, clip_vector_path, target_vector_path, suffix):
    """Clip a vector against an AOI and output result in AOI coordinates.

    Parameters:
        base_vector_path (str): path to a base vector
        clip_vector_path (str): path to an AOI vector
        target_vector_path (str): desired output path to write the
            clipped base against AOI in AOI's coordinate system.
        suffix (str): a string to append at the end of the output files.

    Returns:
        None.

    """
    LOGGER.info('Entering clip_vector')

    if os.path.isfile(target_vector_path):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.DeleteDataSource(target_vector_path)

    # Get the spatial references as strings in Well Known Text
    source_sr_wkt = pygeoprocessing.get_vector_info(base_vector_path)[
        'projection']
    target_sr_wkt = pygeoprocessing.get_vector_info(clip_vector_path)[
        'projection']

    # Create path for the reprojected shapefile
    reprojected_vector_path = os.path.join(
        os.path.dirname(base_vector_path),
        os.path.basename(base_vector_path).replace('.shp', '_projected%s.shp')
        % suffix)

    # Reproject the shapefile to the spatial reference of AOI so that AOI
    # can be used to clip the shapefile properly
    if source_sr_wkt != target_sr_wkt:
        pygeoprocessing.reproject_vector(base_vector_path, target_sr_wkt,
                                         reprojected_vector_path)
        # Clip the shapefile to the AOI
        clip_features(reprojected_vector_path, clip_vector_path,
                      target_vector_path)
    else:
        # Clip the shapefile without reprojecting
        clip_features(base_vector_path, clip_vector_path, target_vector_path)

    LOGGER.info('Finished clip_vector')


def clip_features(base_vector_path, clip_vector_path, target_vector_path):
    """Create a new target point vector where base points are contained in the
    single polygon in clip_vector_path. Assumes all data are in the same
    projection.

    Parameters:
        base_vector_path (str): path to a point vector to clip
        clip_vector_path (str): path to a single polygon vector for clipping.
        target_vector_path (str): output path for the clipped vector.

    Returns:
        None.

    """
    LOGGER.info('Entering clip_features')

    # Get layer and geometry informations from path
    base_vector = gdal.OpenEx(base_vector_path)
    base_layer = base_vector.GetLayer()
    base_layer_defn = base_layer.GetLayerDefn()
    base_layer_geom = base_layer.GetGeomType()

    clip_vector = gdal.OpenEx(clip_vector_path)
    clip_layer = clip_vector.GetLayer()
    clip_feat = next(clip_layer)  # Assuming one feature in clip_layer
    clip_geom = clip_feat.GetGeometryRef()
    clip_shapely = shapely.wkb.loads(clip_geom.ExportToWkb())
    clip_prep = shapely.prepared.prep(clip_shapely)

    # Create a target point vector based on the properties of base point vector
    target_driver = ogr.GetDriverByName('ESRI Shapefile')
    target_vector = target_driver.CreateDataSource(target_vector_path)
    target_layer = target_vector.CreateLayer(base_layer_defn.GetName(),
                                             base_layer.GetSpatialRef(),
                                             base_layer_geom)
    target_layer = target_vector.GetLayer()
    target_defn = target_layer.GetLayerDefn()

    # Add input Layer Fields to the output Layer
    for i in range(0, base_layer_defn.GetFieldCount()):
        base_field_defn = base_layer_defn.GetFieldDefn(i)
        target_layer.CreateField(base_field_defn)

    # Write any point feature that lies within the polygon to the target vector
    for base_feat in base_layer:
        base_geom = base_feat.GetGeometryRef()
        base_shapely = shapely.wkb.loads(base_geom.ExportToWkb())

        if clip_prep.intersects(base_shapely):
            # Create output feature
            target_feat = ogr.Feature(target_defn)
            target_feat.SetGeometry(base_geom.Clone())

            # Add field values from input Layer
            for i in range(0, target_defn.GetFieldCount()):
                target_feat.SetField(
                    target_defn.GetFieldDefn(i).GetNameRef(),
                    base_feat.GetField(i))
            target_layer.CreateFeature(target_feat)
            target_feat = None

    target_layer = None
    target_vector = None
    clip_layer = None
    clip_vector = None
    base_layer = None
    base_vector = None


def clip_vector_layer(base_vector_path, clip_vector_path,
                      target_clipped_vector_path):
    """Clip Shapefile Layer by second Shapefile Layer.

    Uses ogr.Layer.Clip() to clip a Shapefile, where the output Layer
    inherits the projection and fields from the original Shapefile.

    Parameters:
        base_vector_path (str): a path to a Shapefile on disk. This is
            the Layer to clip. Must have same spatial reference as
            'clip_vector_path'.
        clip_vector_path (str): a path to a Shapefile on disk. This is
            the Layer to clip to. Must have same spatial reference as
            'base_vector_path'
        target_clipped_vector_path (str): a path on disk to write the clipped
            shapefile to. Should end with a '.shp' extension.

    Returns:
        None

    """

    if os.path.isfile(target_clipped_vector_path):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.DeleteDataSource(target_clipped_vector_path)

    base_vector = gdal.OpenEx(base_vector_path)
    clip_vector = gdal.OpenEx(clip_vector_path)

    base_layer = base_vector.GetLayer()
    clip_layer = clip_vector.GetLayer()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateDataSource(target_clipped_vector_path)
    base_layer_defn = base_layer.GetLayerDefn()
    target_layer = target_vector.CreateLayer(base_layer_defn.GetName(),
                                             base_layer.GetSpatialRef())

    base_layer.Clip(clip_layer, target_layer)

    # Add in a check to make sure the intersection didn't come back empty
    if target_layer.GetFeatureCount() == 0:
        raise IntersectionError(
            'Intersection ERROR: found no intersection between: file - %s and '
            'file - %s. This could be caused by the AOI not overlapping any '
            'Wave Energy Points.\n'
            'Suggestions: open workspace/intermediate/projected_wave_data.shp'
            'and the AOI to make sure AOI overlaps at least on point.' %
            (base_vector_path, clip_vector_path))


def wave_energy_interp(wave_data, machine_perf):
    """Generates a matrix representing the interpolation of the
        machine performance table using new ranges from wave watch data.

    Parameters:
        wave_data (dict): A dictionary holding the new x range (period) and
            y range (height) values for the interpolation.  The
            dictionary has the following structure:
              {'periods': [1,2,3,4,...],
               'heights': [.5,1.0,1.5,...],
               'bin_matrix': { (i0,j0): [[2,5,3,2,...], [6,3,4,1,...],...],
                               (i1,j1): [[2,5,3,2,...], [6,3,4,1,...],...],
                                ...
                               (in, jn): [[2,5,3,2,...], [6,3,4,1,...],...]
                             }
              }
        machine_perf (dict): a dictionary that holds the machine performance
            information with the following keys and structure:
                machine_perf['periods'] - [1,2,3,...]
                machine_perf['heights'] - [.5,1,1.5,...]
                machine_perf['bin_matrix'] - [[1,2,3,...],[5,6,7,...],...].

    Returns:
        The interpolated matrix

    """
    # Get ranges and matrix for machine performance table
    x_range = numpy.array(machine_perf['periods'], dtype='f')
    y_range = numpy.array(machine_perf['heights'], dtype='f')
    z_matrix = numpy.array(machine_perf['bin_matrix'], dtype='f')
    # Get new ranges to interpolate to, from wave_data table
    new_x = wave_data['periods']
    new_y = wave_data['heights']

    # Interpolate machine performance table and return the interpolated matrix
    interp_z_spl = scipy.interpolate.RectBivariateSpline(
        x_range, y_range, z_matrix.transpose(), kx=1, ky=1)
    return interp_z_spl(new_x, new_y).transpose()


def compute_wave_energy_capacity(wave_data, interp_z, machine_param):
    """Computes the wave energy capacity for each point.

    Also generates a dictionary whose keys are the points (i,j) and whose value
    is the wave energy capacity.

    Parameters:
        wave_data (dict): A wave watch dictionary with the following structure:
               {'periods': [1,2,3,4,...],
                'heights': [.5,1.0,1.5,...],
                'bin_matrix': { (i0,j0): [[2,5,3,2,...], [6,3,4,1,...],...],
                                (i1,j1): [[2,5,3,2,...], [6,3,4,1,...],...],
                                 ...
                                (in, jn): [[2,5,3,2,...], [6,3,4,1,...],...]
                              }
               }
        interp_z (2D-array): A 2D array of the interpolated values for the
            machine performance table
        machine_param (dict): A dictionary containing the restrictions for the
            machines (CapMax, TpMax, HsMax)

    Returns:
        A dictionary representing the wave energy capacity at each wave point

    """
    energy_cap = {}

    # Get the row,col headers (ranges) for the wave watch data
    # row is wave period label
    # col is wave height label
    wave_periods = wave_data['periods']
    wave_heights = wave_data['heights']

    # Get the machine parameter restriction values
    cap_max = float(machine_param['capmax'])
    period_max = float(machine_param['tpmax'])
    height_max = float(machine_param['hsmax'])

    # It seems that the capacity max is already set to it's limit in
    # the machine performance table. However, if it needed to be
    # restricted the following line will do it
    interp_z = numpy.array(interp_z)
    interp_z[interp_z > cap_max] = cap_max

    # Set position variables to use as a check and as an end
    # point for rows/cols if restrictions limit the ranges
    period_max_index = -1
    height_max_index = -1

    # Using the restrictions find the max position (index) for period and
    # height in the wave_periods/wave_heights ranges

    for index_pos, value in enumerate(wave_periods):
        if value > period_max:
            period_max_index = index_pos
            break

    for index_pos, value in enumerate(wave_heights):
        if value > height_max:
            height_max_index = index_pos
            break

    LOGGER.debug('Position of max period : %f', period_max_index)
    LOGGER.debug('Position of max height : %f', height_max_index)

    # For all the wave watch points, multiply the occurence matrix by the
    # interpolated machine performance matrix to get the captured wave energy
    for key, val in wave_data['bin_matrix'].iteritems():
        # Convert all values to type float
        temp_matrix = numpy.array(val, dtype='f')
        mult_matrix = numpy.multiply(temp_matrix, interp_z)
        # Set any value that is outside the restricting ranges provided by
        # machine parameters to zero
        if period_max_index != -1:
            mult_matrix[:, period_max_index:] = 0
        if height_max_index != -1:
            mult_matrix[height_max_index:, :] = 0

        # Since we are doing a cubic interpolation there is a possibility we
        # will have negative values where they should be zero. So here
        # we drive any negative values to zero.
        mult_matrix[mult_matrix < 0] = 0

        # Sum all of the values from the matrix to get the total
        # captured wave energy and convert into mega watts
        sum_we = (mult_matrix.sum() / 1000)
        energy_cap[key] = sum_we

    return energy_cap


def captured_wave_energy_to_shape(energy_cap, wave_shape_path):
    """Adds each captured wave energy value from the dictionary
        energy_cap to a field of the shapefile wave_shape. The values are
        set corresponding to the same I,J values which is the key of the
        dictionary and used as the unique identier of the shape.

        energy_cap - A dictionary with keys (I,J), representing the
            wave energy capacity values.
        wave_shape_path  - A path to a point geometry shapefile to
            write the new field/values to

        returns - Nothing"""

    cap_we_field = 'CAPWE_MWHY'
    wave_shape = gdal.OpenEx(wave_shape_path, 1)
    wave_layer = wave_shape.GetLayer()
    # Create a new field for the shapefile
    field_defn = ogr.FieldDefn(cap_we_field, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    wave_layer.CreateField(field_defn)
    # For all of the features (points) in the shapefile, get the
    # corresponding point/value from the dictionary and set the 'capWE_Sum'
    # field as the value from the dictionary
    for feat in wave_layer:
        index_i = feat.GetFieldIndex('I')
        index_j = feat.GetFieldIndex('J')
        value_i = feat.GetField(index_i)
        value_j = feat.GetField(index_j)
        we_value = energy_cap[(value_i, value_j)]

        index = feat.GetFieldIndex(cap_we_field)
        feat.SetField(index, we_value)
        # Save the feature modifications to the layer.
        wave_layer.SetFeature(feat)
        feat = None


def calculate_percentiles_from_raster(base_raster_path, percentile_list):
    """Determine the percentiles of a raster using the nearest-rank method.

    Iterate through the raster blocks and round the unique values for
    efficiency. Then add each unique value-count pair into a dictionary.
    Compute ordinal ranks given the percentile list.

    Parameters:
        base_raster_path (str): path to a gdal raster on disk
        percentile_list (list): a list of desired percentiles to lookup,
            ex: [25,50,75,90]

    Returns:
            a list of values corresponding to the percentiles from the list

    """
    nodata = pygeoprocessing.get_raster_info(base_raster_path)['nodata'][0]
    unique_value_counts = {}
    for _, block_matrix in pygeoprocessing.iterblocks(base_raster_path):
        # Sum the values with the same key in both dictionaries
        unique_values, counts = numpy.unique(block_matrix, return_counts=True)
        # Round the array so the unique values won't explode the dictionary
        numpy.round(unique_values, decimals=1, out=unique_values)

        block_unique_value_counts = dict(zip(unique_values, counts))
        for value in block_unique_value_counts.keys():
            unique_value_counts[value] = unique_value_counts.get(
                value, 0) + block_unique_value_counts.get(value, 0)

    # Remove the nodata key and its count from the dictionary
    unique_value_counts.pop(nodata, None)
    LOGGER.debug('Unique_value_counts: %s', unique_value_counts)

    # Get the total pixel count except nodata pixels
    total_count = sum(unique_value_counts.values())

    # Calculate the ordinal rank
    ordinal_rank = [
        numpy.ceil(percentile / 100.0 * total_count)
        for percentile in percentile_list
    ]

    # Get values from the ordered dictionary that correspond to the ranks
    percentile_values = []  # list for corresponding values
    ith_element = 0  # indexing the ith element in the percentile_values list
    cumulative_count = 0  # for checking if the percentile value is reached

    # Get the unique_value keys in the ascending order
    for unique_value in sorted(unique_value_counts.keys()):
        # stop iteration when all corresponding values are obtained
        if ith_element > len(ordinal_rank) - 1:
            break

        # Add count from the unique value
        count = unique_value_counts[unique_value]
        cumulative_count += count

        while ordinal_rank[ith_element] == cumulative_count or \
            ordinal_rank[ith_element] < cumulative_count:
            percentile_values.append(unique_value)
            ith_element += 1

            # stop iteration when all corresponding values are obtained
            if ith_element == len(ordinal_rank):
                break

    LOGGER.debug('Percentile_values: %s', percentile_values)
    return percentile_values


def count_pixels_groups(raster_path, group_values):
    """Does a pixel count for each value in 'group_values' over the
        raster provided by 'raster_path'. Returns a list of pixel counts
        for each value in 'group_values'

    Parameters:
        raster_path (str): path to a gdal raster on disk
        group_values (list): unique numbers for which to get a pixel count

    Returns:
        A list of integers, where each integer at an index corresponds to the
        pixel count of the value from 'group_values' found at the same index

    """
    # Initialize a list that will hold pixel counts for each group
    pixel_count = numpy.zeros(len(group_values))

    for _, block_matrix in pygeoprocessing.iterblocks(raster_path):
        # Cumulatively add the pixels count for each value in 'group_values'
        for index in xrange(len(group_values)):
            val = group_values[index]
            count_mask = numpy.zeros(block_matrix.shape)
            numpy.equal(block_matrix, val, count_mask)
            pixel_count[index] += numpy.count_nonzero(count_mask)

    return pixel_count


def pixel_size_based_on_coordinate_transform(dataset_path, coord_trans, point):
    """Get width and height of cell in meters.

    Calculates the pixel width and height in meters given a coordinate
    transform and reference point on the dataset that's close to the
    transform's projected coordinate sytem.  This is only necessary
    if dataset is not already in a meter coordinate system, for example
    dataset may be in lat/long (WGS84).

    Args:
        dataset_path (str): a String for a GDAL path on disk, projected
            in the form of lat/long decimal degrees
        coord_trans (osr.CoordinateTransformation): an OSR coordinate
            transformation from dataset coordinate system to meters
        point (tuple): a reference point close to the coordinate transform
            coordinate system.  must be in the same coordinate system as
            dataset.

    Returns:
        pixel_diff (tuple): a 2-tuple containing (pixel width in meters, pixel
            height in meters)
    """
    dataset = gdal.OpenEx(dataset_path)
    # Get the first points (x, y) from geoTransform
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
    pixel_diff_x = point_2[0] - point_1[0]
    pixel_diff_y = point_2[1] - point_1[1]

    # Close and clean up dataset
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return (pixel_diff_x, pixel_diff_y)


def _create_raster_attr_table(dataset_path, attr_dict, column_name):
    """Create a raster attribute table.

    Parameters:
        dataset_path (str): a GDAL raster dataset to create the RAT for
        attr_dict (dict): a dictionary with keys that point to a primitive type
            ex: {integer_id_1: value_1, ... integer_id_n: value_n}
        column_name (str): a string for the column name that maps the values

    Returns:
        None
    """
    dataset = gdal.OpenEx(dataset_path, gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    attr_table = gdal.RasterAttributeTable()
    attr_table.SetRowCount(len(attr_dict))

    # create columns
    attr_table.CreateColumn('Value', gdal.GFT_Integer, gdal.GFU_MinMax)
    attr_table.CreateColumn(column_name, gdal.GFT_String, gdal.GFU_Name)

    row_count = 0
    for key in sorted(attr_dict.keys()):
        attr_table.SetValueAsInt(row_count, 0, int(key))
        attr_table.SetValueAsString(row_count, 1, attr_dict[key])
        row_count += 1

    band.SetDefaultRAT(attr_table)

    # Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Wave Energy.

    Parameters:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    warnings = []
    keys_missing_value = []
    missing_keys = []
    for required_key in ('workspace_dir', 'wave_base_data_path',
                         'analysis_area_path', 'machine_perf_path',
                         'machine_param_path', 'dem_path'):
        try:
            if args[required_key] in ('', None):
                keys_missing_value.append(required_key)
        except KeyError:
            missing_keys.append(required_key)

    if len(missing_keys) > 0:
        raise KeyError('Keys are missing from args: %s' % str(missing_keys))

    if len(keys_missing_value) > 0:
        warnings.append((keys_missing_value,
                         'Parameter is required but has no value'))

    if limit_to in ('wave_base_data_path', None):
        if not os.path.isdir(args['wave_base_data_path']):
            warnings.append((['wave_base_data_path'],
                             'Parameter not found or is not a folder.'))

    if limit_to in ('analysis_area_path', None):
        if args['analysis_area_path'] not in (
                "West Coast of North America and Hawaii",
                "East Coast of North America and Puerto Rico",
                "North Sea 4 meter resolution",
                "North Sea 10 meter resolution", "Australia", "Global"):
            warnings.append((['analysis_area_path'],
                             'Parameter must be a known analysis area.'))

    if limit_to in ('aoi_path', None):
        try:
            if args['aoi_path'] not in ('', None):
                with utils.capture_gdal_logging():
                    vector = gdal.OpenEx(args['aoi_path'])
                    layer = vector.GetLayer()
                    geometry_type = layer.GetGeomType()
                    if geometry_type != ogr.wkbPolygon:
                        warnings.append((['aoi_path'],
                                         'Vector must contain only polygons.'))
                    srs = layer.GetSpatialRef()
                    units = srs.GetLinearUnitsName().lower()
                    if units not in ('meter', 'metre'):
                        warnings.append((['aoi_path'],
                                         'Vector must be projected in meters'))

                    datum = srs.GetAttrValue('DATUM')
                    if datum != 'WGS_1984':
                        warnings.append(
                            (['aoi_path'],
                             'Vector must use the WGS_1984 datum.'))
        except KeyError:
            # Parameter is not required.
            pass

    for csv_key, required_fields in (('machine_perf_path', set(
        [])), ('machine_param_path', set(
            ['name', 'value',
             'note'])), ('land_gridPts_path',
                         set(['id', 'type', 'lat', 'long', 'location'])),
                                     ('machine_econ_path',
                                      set(['name', 'value', 'note']))):
        try:
            reader = csv.reader(open(args[csv_key]))
            headers = set([field.lower() for field in reader.next()])
            missing_fields = required_fields - headers
            if len(missing_fields) > 0:
                warnings.append('CSV is missing columns :%s' % ', '.join(
                    sorted(missing_fields)))
        except KeyError:
            # Not all these are required inputs.
            pass
        except IOError:
            warnings.append(([csv_key], 'File not found.'))
        except csv.Error:
            warnings.append(([csv_key], 'CSV could not be read.'))

    if limit_to in ('dem_path', None):
        with utils.capture_gdal_logging():
            raster = gdal.OpenEx(args['dem_path'])
        if raster is None:
            warnings.append(
                (['dem_path'],
                 ('Parameter must be a filepath to a GDAL-compatible '
                  'raster file.')))

    if limit_to in ('number_of_machines', None):
        try:
            num_machines = args['number_of_machines']
            if (int(float(num_machines)) != float(num_machines)
                    or float(num_machines) < 0):
                warnings.append((['number_of_machines'],
                                 'Parameter must be a positive integer.'))
        except KeyError:
            pass
        except ValueError:
            warnings.append((['number_of_machines'],
                             'Parameter must be a number.'))

    if limit_to is None:
        if 'valuation_container' in args:
            missing_keys = []
            keys_with_no_value = []
            for required_key in ('land_gridPts_path', 'machine_econ_path',
                                 'number_of_machines'):
                try:
                    if args[required_key] in ('', None):
                        keys_with_no_value.append(required_key)
                except KeyError:
                    missing_keys.append(required_key)

            if len(missing_keys) > 0:
                raise KeyError('Keys are missing: %s' % missing_keys)

            if len(keys_with_no_value) > 0:
                warnings.append((keys_with_no_value,
                                 'Parameter must have a value'))

    return warnings
