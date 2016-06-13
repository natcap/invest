"""InVEST Wind Energy model """
import logging
import os
import csv
import struct
import shutil
import math

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import numpy as np
from scipy import integrate
#required for py2exe to build
from scipy.sparse.csgraph import _validation
import shapely.wkt
import shapely.ops
from shapely import speedups

import pygeoprocessing.geoprocessing

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
     %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.wind_energy.wind_energy')

speedups.enable()


class FieldError(Exception):
    """A custom error message for fields that are missing"""
    pass


class TimePeriodError(Exception):
    """A custom error message for when the number of years does not match
        the number of years given in the price table"""
    pass


def execute(args):
    """Wind Energy.

    This module handles the execution of the wind energy model
    given the following dictionary:

    Args:
        workspace_dir (string): a python string which is the uri path to where
            the outputs will be saved (required)
        wind_data_uri (string): path to a CSV file with the following header:
            ['LONG','LATI','LAM', 'K', 'REF']. Each following row is a location
            with at least the Longitude, Latitude, Scale ('LAM'),
            Shape ('K'), and reference height ('REF') at which the data was
            collected (required)
        aoi_uri (string): a uri to an OGR datasource that is of type polygon
            and projected in linear units of meters. The polygon specifies the
            area of interest for the wind data points. If limiting the wind
            farm bins by distance, then the aoi should also cover a portion
            of the land polygon that is of interest (optional for biophysical
            and no distance masking, required for biophysical and distance
            masking, required for valuation)
        bathymetry_uri (string): a uri to a GDAL dataset that has the depth
            values of the area of interest (required)
        land_polygon_uri (string): a uri to an OGR datasource of type polygon
            that provides a coastline for determining distances from wind farm
            bins. Enabled by AOI and required if wanting to mask by distances
            or run valuation
        global_wind_parameters_uri (string): a float for the average distance
            in kilometers from a grid connection point to a land connection
            point (required for valuation if grid connection points are not
            provided)
        suffix (string): a String to append to the end of the output files
            (optional)
        turbine_parameters_uri (string): a uri to a CSV file that holds the
            turbines biophysical parameters as well as valuation parameters
            (required)
        number_of_turbines (int): an integer value for the number of machines
            for the wind farm (required for valuation)
        min_depth (float): a float value for the minimum depth for offshore
            wind farm installation (meters) (required)
        max_depth (float): a float value for the maximum depth for offshore
            wind farm installation (meters) (required)
        min_distance (float): a float value for the minimum distance from shore
            for offshore wind farm installation (meters) The land polygon must
            be selected for this input to be active (optional, required for
            valuation)
        max_distance (float): a float value for the maximum distance from shore
            for offshore wind farm installation (meters) The land polygon must
            be selected for this input to be active (optional, required for
            valuation)
        valuation_container (boolean): Indicates whether model includes
            valuation
        foundation_cost (float): a float representing how much the foundation
            will cost for the specific type of turbine (required for valuation)
        discount_rate (float): a float value for the discount rate (required
            for valuation)
        grid_points_uri (string): a uri to a CSV file that specifies the
            landing and grid point locations (optional)
        avg_grid_distance (float): a float for the average distance in
            kilometers from a grid connection point to a land connection point
            (required for valuation if grid connection points are not provided)
        price_table (boolean): a bool indicating whether to use the wind energy
            price table or not (required)
        wind_schedule (string): a URI to a CSV file for the yearly prices of
            wind energy for the lifespan of the farm (required if 'price_table'
            is true)
        wind_price (float): a float for the wind energy price at year 0
            (required if price_table is false)
        rate_change (float): a float as a percent for the annual rate of change
            in the price of wind energy. (required if price_table is false)

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'wind_data_uri': 'path/to/file',
            'aoi_uri': 'path/to/shapefile',
            'bathymetry_uri': 'path/to/raster',
            'land_polygon_uri': 'path/to/shapefile',
            'global_wind_parameters_uri': 'path/to/csv',
            'suffix': '_results',
            'turbine_parameters_uri': 'path/to/csv',
            'number_of_turbines': 10,
            'min_depth': 3,
            'max_depth': 60,
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 3.4,
            'discount_rate': 7.0,
            'grid_points_uri': 'path/to/csv',
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': 'path/to/csv',
            'wind_price': 0.4,
            'rate_change': 0.0,

        }

    Returns:
        None

    """

    LOGGER.debug('Starting the Wind Energy Model')

    workspace = args['workspace_dir']
    inter_dir = os.path.join(workspace, 'intermediate')
    out_dir = os.path.join(workspace, 'output')
    pygeoprocessing.geoprocessing.create_directories([workspace, inter_dir, out_dir])

    bathymetry_uri = args['bathymetry_uri']
    number_of_turbines = int(args['number_of_turbines'])

    # Set the output nodata value to use throughout the model
    out_nodata = -64329.0

    # Append a _ to the suffix if it's not empty and doens't already have one
    try:
        suffix = args['suffix']
        if suffix != "" and not suffix.startswith('_'):
            suffix = '_' + suffix
    except KeyError:
        suffix = ''

    # Create a list of the biophysical parameters we are looking for from the
    # input csv files
    biophysical_params = ['cut_in_wspd', 'cut_out_wspd', 'rated_wspd',
                          'hub_height', 'turbine_rated_pwr', 'air_density',
                          'exponent_power_curve', 'air_density_coefficient',
                          'loss_parameter', 'turbines_per_circuit',
                          'rotor_diameter', 'rotor_diameter_factor']

    # Read the biophysical turbine parameters into a dictionary
    bio_turbine_dict = read_csv_wind_parameters(
            args['turbine_parameters_uri'], biophysical_params)

    # Read the biophysical global parameters into a dictionary
    bio_global_param_dict = read_csv_wind_parameters(
            args['global_wind_parameters_uri'], biophysical_params)

    # Combine the turbine and global parameters into one dictionary
    bio_parameters_dict = combine_dictionaries(
            bio_turbine_dict, bio_global_param_dict)

    LOGGER.debug('Biophysical Turbine Parameters: %s', bio_parameters_dict)

    # Check that all the necessary input fields from the CSV files have been
    # collected by comparing the number of dictionary keys to the number of
    # elements in our known list
    if len(bio_parameters_dict.keys()) != len(biophysical_params):
        raise FieldError('An Error occured from reading in a field value from '
        'either the turbine CSV file or the global parameters JSON file. '
        'Please make sure all the necessary fields are present and spelled '
        'correctly.')

    # Hub Height to use for setting weibell paramaters
    hub_height = int(bio_parameters_dict['hub_height'])

    # The scale_key is used in getting the right wind energy arguments that are
    # dependent on the hub height.
    scale_key = 'LAM'

    LOGGER.debug('hub_height : %s', hub_height)

    # Read the wind energy data into a dictionary
    LOGGER.info('Reading in Wind Data')
    wind_data = read_csv_wind_data(args['wind_data_uri'], hub_height)

    if 'aoi_uri' in args:
        LOGGER.info('AOI Provided')

        aoi_uri = args['aoi_uri']

        # Since an AOI was provided the wind energy points shapefile will need
        # to be clipped and projected. Thus save the construction of the
        # shapefile from dictionary in the intermediate directory. The final
        # projected shapefile will be written to the output directory
        wind_point_shape_uri = os.path.join(
                inter_dir, 'wind_energy_points_from_data%s.shp' % suffix)

        # Create point shapefile from wind data
        LOGGER.info('Create point shapefile from wind data')
        wind_data_to_point_shape(wind_data, 'wind_data', wind_point_shape_uri)

        # Define the uri for projecting the wind energy data points to that of
        # the AOI
        wind_points_proj_uri = os.path.join(
                out_dir, 'wind_energy_points%s.shp' % suffix)

        # Clip and project the wind energy points datasource
        LOGGER.debug('Clip and project wind points to AOI')
        clip_and_reproject_shapefile(
                wind_point_shape_uri, aoi_uri, wind_points_proj_uri)

        # Define the uri for projecting the bathymetry to AOI
        bathymetry_proj_uri = os.path.join(
                inter_dir, 'bathymetry_projected%s.tif' % suffix)

        # Clip and project the bathymetry dataset
        LOGGER.debug('Clip and project bathymetry to AOI')
        clip_and_reproject_raster(bathymetry_uri, aoi_uri, bathymetry_proj_uri)

        # Set the bathymetry and points URI to use in the rest of the model. In
        # this case these URIs refer to the projected files. This may not be the
        # case if an AOI is not provided
        final_bathymetry_uri = bathymetry_proj_uri
        final_wind_points_uri = wind_points_proj_uri

        # Try to handle the distance inputs and land datasource if they
        # are present
        try:
            min_distance = float(args['min_distance'])
            max_distance = float(args['max_distance'])
            land_polygon_uri = args['land_polygon_uri']
        except KeyError:
            LOGGER.info('Distance information not provided')
        else:
            LOGGER.info('Handling distance parameters')

            # Define the uri for reprojecting the land polygon datasource
            land_poly_proj_uri = os.path.join(
                    inter_dir, 'land_poly_projected%s.shp' % suffix)
            # Clip and project the land polygon datasource
            LOGGER.debug('Clip and project land poly to AOI')
            clip_and_reproject_shapefile(
                    land_polygon_uri, aoi_uri, land_poly_proj_uri)

            # Get the cell size to use in new raster outputs from the DEM
            cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                    final_bathymetry_uri)

            # If the distance inputs are present create a mask for the output
            # area that restricts where the wind energy farms can be based
            # on distance
            aoi_raster_uri = os.path.join(
                    inter_dir, 'aoi_raster%s.tif' % suffix)

            LOGGER.debug('Create Raster From AOI')
            # Make a raster from the AOI using the bathymetry rasters pixel size
            pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
                aoi_uri, cell_size, gdal.GDT_Float32, out_nodata,
                aoi_raster_uri)

            LOGGER.debug('Rasterize AOI onto raster')
            # Burn the area of interest onto the raster
            pygeoprocessing.geoprocessing.rasterize_layer_uri(
                aoi_raster_uri, aoi_uri, [0],
                option_list=["ALL_TOUCHED=TRUE"])

            LOGGER.debug('Rasterize Land Polygon onto raster')
            # Burn the land polygon onto the raster, covering up the AOI values
            # where they overlap
            pygeoprocessing.geoprocessing.rasterize_layer_uri(
                aoi_raster_uri, land_poly_proj_uri, [1],
                option_list=["ALL_TOUCHED=TRUE"])

            dist_mask_uri = os.path.join(
                    inter_dir, 'distance_mask%s.tif' % suffix)

            dist_trans_uri = os.path.join(
                    inter_dir, 'distance_trans%s.tif' % suffix)

            dist_meters_uri = os.path.join(
                    inter_dir, 'distance_meters%s.tif' % suffix)

            LOGGER.info('Generate Distance Mask')
            # Create a distance mask
            pygeoprocessing.geoprocessing.distance_transform_edt(aoi_raster_uri, dist_trans_uri)
            mask_by_distance(
                    dist_trans_uri, min_distance, max_distance,
                    out_nodata, dist_meters_uri, dist_mask_uri)

        # Determines whether to check projections in future vectorize_datasets
        # calls
        projected = True
    else:
        LOGGER.info("AOI argument was not selected")

        # Since no AOI was provided the wind energy points shapefile that is
        # created directly from dictionary will be the final output, so set the
        # uri to point to the output folder
        wind_point_shape_uri = os.path.join(
                out_dir, 'wind_energy_points%s.shp' % suffix)

        # Create point shapefile from wind data dictionary
        LOGGER.debug('Create point shapefile from wind data')
        wind_data_to_point_shape(wind_data, 'wind_data', wind_point_shape_uri)

        # Set the bathymetry and points URI to use in the rest of the model. In
        # this case these URIs refer to the unprojected files. This may not be
        # the case if an AOI is provided
        final_wind_points_uri = wind_point_shape_uri
        final_bathymetry_uri = bathymetry_uri

        # Determines whether to check projections in future vectorize_datasets
        # calls. Since no AOI is provided set to False since all our data is in
        # geographical format
        projected = False

    # Get the min and max depth values from the arguments and set to a negative
    # value indicating below sea level
    min_depth = abs(float(args['min_depth'])) * -1.0
    max_depth = abs(float(args['max_depth'])) * -1.0

    def depth_op(bath):
        """A vectorized function that takes one argument and uses a range to
            determine if that value falls within the range

            bath - an integer value of either positive or negative
            min_depth - a float value specifying the lower limit of the range.
                this value is set above
            max_depth - a float value specifying the upper limit of the range
                this value is set above
            out_nodata - a int or float for the nodata value described above

            returns - out_nodata if 'bath' does not fall within the range, or
                'bath' if it does"""
        return np.where(
                ((bath >= max_depth) & (bath <= min_depth)), bath, out_nodata)

    depth_mask_uri = os.path.join(inter_dir, 'depth_mask%s.tif' % suffix)

    # Get the cell size here to use from the DEM. The cell size could either
    # come in a project unprojected format
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(final_bathymetry_uri)

    # Create a mask for any values that are out of the range of the depth values
    LOGGER.info('Creating Depth Mask')
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [final_bathymetry_uri], depth_op, depth_mask_uri, gdal.GDT_Float32,
            out_nodata, cell_size, 'intersection',
            assert_datasets_projected=projected, vectorize_op = False)

    # The String name for the shape field. So far this is a default from the
    # text file given by CK. I guess we could search for the 'K' if needed.
    shape_key = 'K'

    # Weibull probability function to integrate over
    def weibull_probability(v_speed, k_shape, l_scale):
        """Calculate the weibull probability function of variable v_speed

            v_speed - a number representing wind speed
            k_shape - a float for the shape parameter
            l_scale - a float for the scale parameter of the distribution

            returns - a float"""
        return ((k_shape / l_scale) * (v_speed / l_scale)**(k_shape - 1) *
                (math.exp(-1 * (v_speed/l_scale)**k_shape)))

    # Density wind energy function to integrate over
    def density_wind_energy_fun(v_speed, k_shape, l_scale):
        """Calculate the probability density function of a weibull variable
            v_speed

            v_speed - a number representing wind speed
            k_shape - a float for the shape parameter
            l_scale - a float for the scale parameter of the distribution

            returns - a float"""
        return ((k_shape / l_scale) * (v_speed / l_scale)**(k_shape - 1) *
                (math.exp(-1 * (v_speed/l_scale)**k_shape))) * v_speed**3

    # Harvested wind energy function to integrate over
    def harvested_wind_energy_fun(v_speed, k_shape, l_scale):
        """Calculate the harvested wind energy

            v_speed - a number representing wind speed
            k_shape - a float for the shape parameter
            l_scale - a float for the scale parameter of the distribution

            returns - a float"""
        fract = ((v_speed**exp_pwr_curve - v_in**exp_pwr_curve) /
            (v_rate**exp_pwr_curve - v_in**exp_pwr_curve))

        return fract * weibull_probability(v_speed, k_shape, l_scale)

    # The harvested energy is on a per year basis
    num_days = 365

    # The rated power is expressed in units of MW but the harvested energy
    # equation calls for it in terms of Wh. Thus we multiply by a million to get
    # to Wh.
    rated_power = float(bio_parameters_dict['turbine_rated_pwr']) * 1000000

    # Get the rest of the inputs needed to compute harvested wind energy
    # from the dictionary so that it is in a more readable format
    exp_pwr_curve = int(bio_parameters_dict['exponent_power_curve'])
    air_density_standard = float(bio_parameters_dict['air_density'])
    v_rate = float(bio_parameters_dict['rated_wspd'])
    v_out = float(bio_parameters_dict['cut_out_wspd'])
    v_in = float(bio_parameters_dict['cut_in_wspd'])
    air_density_coef = float(bio_parameters_dict['air_density_coefficient'])
    losses = float(bio_parameters_dict['loss_parameter'])

    # Compute the mean air density, given by CKs formulas
    mean_air_density = air_density_standard - air_density_coef * hub_height

    # Fractional coefficient that lives outside the intregation for computing
    # the harvested wind energy
    fract_coef = rated_power * (mean_air_density / air_density_standard)

    # The coefficient that is multiplied by the integration portion of the
    # harvested wind energy equation
    scalar = num_days * 24 * fract_coef

    # The field names for the two outputs, Harvested Wind Energy and Wind
    # Density, to be added to the point shapefile
    density_field_name = 'Dens_W/m2'
    harvest_field_name = 'Harv_MWhr'

    def compute_density_harvested_uri(wind_pts_uri):
        """A URI wrapper to compute the density and harvested energy for wind
            energy. This is to help not open and pass around datasets /
            datasources.

            wind_pts_uri - a URI to a point shapefile to write the results to

            returns - nothing"""
        # Open the wind points file to edit
        wind_points = ogr.Open(wind_pts_uri, 1)
        wind_points_layer = wind_points.GetLayer()

        # Get a feature so that we can get field indices that we will use
        # multiple times
        feature = wind_points_layer.GetFeature(0)

        # Get the indexes for the scale and shape parameters
        scale_index = feature.GetFieldIndex(scale_key)
        shape_index = feature.GetFieldIndex(shape_key)
        LOGGER.debug('scale/shape index : %s:%s', scale_index, shape_index)

        wind_points_layer.ResetReading()

        LOGGER.debug('Creating Harvest and Density Fields')
        # Create new fields for the density and harvested values
        for new_field_name in [density_field_name, harvest_field_name]:
            new_field = ogr.FieldDefn(new_field_name, ogr.OFTReal)
            wind_points_layer.CreateField(new_field)

        LOGGER.debug('Entering Density and Harvest Calculations for each point')
        # For all the locations compute the weibull density and
        # harvested wind energy. Save in a field of the feature
        for feat in wind_points_layer:
            # Get the scale and shape values
            scale_value = feat.GetField(scale_index)
            shape_value = feat.GetField(shape_index)

            # Integrate over the probability density function. 0 and 50 are hard
            # coded values set in CKs documentation
            density_results = integrate.quad(
                    density_wind_energy_fun, 0, 50, (shape_value, scale_value))

            # Compute the final wind power density value
            density_results = 0.5 * mean_air_density * density_results[0]

            # Integrate over the harvested wind energy function
            harv_results = integrate.quad(
                    harvested_wind_energy_fun, v_in, v_rate,
                    (shape_value, scale_value))

            # Integrate over the weibull probability function
            weibull_results = integrate.quad(weibull_probability, v_rate, v_out,
                    (shape_value, scale_value))

            # Compute the final harvested wind energy value
            harvested_wind_energy = (
                    scalar * (harv_results[0] + weibull_results[0]))

            # Convert harvested energy from Whr/yr to MWhr/yr by dividing by
            # 1,000,000
            harvested_wind_energy = harvested_wind_energy / 1000000.00

            # Now factor in the percent losses due to turbine
            # downtime (mechanical failure, storm damage, etc.)
            # and due to electrical resistance in the cables
            harvested_wind_energy = (1 - losses) * harvested_wind_energy

            # Finally, multiply the harvested wind energy by the number of
            # turbines to get the amount of energy generated for the entire farm
            harvested_wind_energy = harvested_wind_energy * number_of_turbines

            # Save the results to their respective fields
            for field_name, result_value in [
                    (density_field_name, density_results),
                    (harvest_field_name, harvested_wind_energy)]:
                out_index = feat.GetFieldIndex(field_name)
                feat.SetField(out_index, result_value)

            # Save the feature and set to None to clean up
            wind_points_layer.SetFeature(feat)

        wind_points = None

    # Compute Wind Density and Harvested Wind Energy, adding the values to the
    # points in the wind point shapefile
    compute_density_harvested_uri(final_wind_points_uri)

    # Temp URIs for creating density and harvested rasters
    density_temp_uri = pygeoprocessing.geoprocessing.temporary_filename()
    harvested_temp_uri = pygeoprocessing.geoprocessing.temporary_filename()

    # Create rasters for density and harvested values
    LOGGER.info('Create Density Raster')
    pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
            final_wind_points_uri, cell_size, gdal.GDT_Float32, out_nodata,
            density_temp_uri)

    LOGGER.info('Create Harvested Raster')
    pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
            final_wind_points_uri, cell_size, gdal.GDT_Float32, out_nodata,
            harvested_temp_uri)

    # Interpolate points onto raster for density values and harvested values:
    LOGGER.info('Vectorize Density Points')
    pygeoprocessing.geoprocessing.vectorize_points_uri(
            final_wind_points_uri, density_field_name, density_temp_uri,
            interpolation = 'linear')

    LOGGER.info('Vectorize Harvested Points')
    pygeoprocessing.geoprocessing.vectorize_points_uri(
            final_wind_points_uri, harvest_field_name, harvested_temp_uri,
            interpolation = 'linear')

    def mask_out_depth_dist(*rasters):
        """Returns the value of the first item in the list if and only if all
            other values are not a nodata value.

            *rasters - a list of values as follows:
                rasters[0] - the desired output value (required)
                rasters[1] - the depth mask value (required)
                rasters[2] - the distance mask value (optional)

            returns - a float of either out_nodata or rasters[0]"""

        nodata_mask = np.empty(rasters[0].shape, dtype=np.int8)
        nodata_mask[:] = 0
        for array in rasters:
            nodata_mask = nodata_mask | (array == out_nodata)

        return np.where(nodata_mask, out_nodata, rasters[0])

    # Output URIs for final Density and Harvested rasters after they've been
    # masked by depth and distance
    density_masked_uri = os.path.join(
            out_dir, 'density_W_per_m2%s.tif' % suffix)
    harvested_masked_uri = os.path.join(
            out_dir, 'harvested_energy_MWhr_per_yr%s.tif' % suffix)

    # List of URIs to pass to vectorize_datasets for operations
    density_mask_list = [density_temp_uri, depth_mask_uri]
    harvest_mask_list = [harvested_temp_uri, depth_mask_uri]

    # If a distance mask was created then add it to the raster list to pass in
    # for masking out the output datasets
    try:
        density_mask_list.append(dist_mask_uri)
        harvest_mask_list.append(dist_mask_uri)
    except NameError:
        LOGGER.debug('NO Distance Mask to add to list')

    # Mask out any areas where distance or depth has determined that wind farms
    # cannot be located
    LOGGER.info('Mask out depth and [distance] areas from Density raster')
    pygeoprocessing.geoprocessing.vectorize_datasets(
            density_mask_list, mask_out_depth_dist, density_masked_uri,
            gdal.GDT_Float32, out_nodata, cell_size, 'intersection',
            assert_datasets_projected = projected, vectorize_op = False)

    LOGGER.info('Mask out depth and [distance] areas from Harvested raster')
    pygeoprocessing.geoprocessing.vectorize_datasets(
            harvest_mask_list, mask_out_depth_dist, harvested_masked_uri,
            gdal.GDT_Float32, out_nodata, cell_size, 'intersection',
            assert_datasets_projected = projected, vectorize_op = False)

    # Create the farm polygon shapefile, which is an example of how big the farm
    # will be with a rough representation of its dimensions.
    # The number of turbines allowed per circuit for infield cabling
    turbines_per_circuit = int(bio_parameters_dict['turbines_per_circuit'])
    # The rotor diameter of the turbines
    rotor_diameter = int(bio_parameters_dict['rotor_diameter'])
    # The rotor diameter factor is a rule by which to use in deciding how far
    # apart the turbines should be spaced
    rotor_diameter_factor = int(bio_parameters_dict['rotor_diameter_factor'])

    # Calculate the number of circuits there will be based on the number of
    # turbines and the number of turbines per circuit. If a fractional value is
    # returned we want to round up and error on the side of having the farm be
    # slightly larger
    num_circuits = math.ceil(float(number_of_turbines) / turbines_per_circuit)
    # The distance needed between turbines
    spacing_dist = rotor_diameter * rotor_diameter_factor

    # Calculate the width
    width = (num_circuits - 1) * spacing_dist
    # Calculate the length
    length = (turbines_per_circuit - 1) * spacing_dist

    # Retrieve the geometry for a point that has the highest harvested energy
    # value, to use as the starting location for building the polygon
    pt_geometry = get_highest_harvested_geom(final_wind_points_uri)
    # Get the X and Y location for the selected wind farm point. These
    # coordinates will be the starting point of which to create the farm lines
    center_x = pt_geometry.GetX()
    center_y = pt_geometry.GetY()
    start_point = (center_x, center_y)
    spat_ref = pygeoprocessing.geoprocessing.get_spatial_ref_uri(final_wind_points_uri)

    farm_poly_uri = os.path.join(out_dir,
        'example_size_and_orientation_of_a_possible_wind_farm%s.shp' % suffix)
    # If the file path already exist, remove it.
    if os.path.isfile(farm_poly_uri):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.DeleteDataSource(farm_poly_uri)

    # Create the actual polygon
    LOGGER.info('Creating Example Farm Polygon')
    create_wind_farm_box(spat_ref, start_point, width, length, farm_poly_uri)

    LOGGER.info('Wind Energy Biophysical Model Complete')

    if 'valuation_container' in args:
        valuation_checked = args['valuation_container']
    else:
        valuation_checked = False

    if not valuation_checked:
        LOGGER.debug('Valuation Not Selected')
        return

    LOGGER.info('Starting Wind Energy Valuation Model')

    # Create a list of the valuation parameters we are looking for from the
    # input files
    valuation_turbine_params = ['turbine_cost', 'turbine_rated_pwr']

    valuation_global_params = [
            'carbon_coefficient', 'time_period', 'infield_cable_cost',
            'infield_cable_length', 'installation_cost',
            'miscellaneous_capex_cost', 'operation_maintenance_cost',
            'decommission_cost', 'ac_dc_distance_break', 'mw_coef_ac',
            'mw_coef_dc', 'cable_coef_ac', 'cable_coef_dc']

    # Read the valuation turbine parameters into a dictionary
    val_turbine_dict = read_csv_wind_parameters(
            args['turbine_parameters_uri'], valuation_turbine_params)

    # Read the valuation global parameters into a dictionary
    val_global_param_dict = read_csv_wind_parameters(
            args['global_wind_parameters_uri'], valuation_global_params)

    # Combine the turbine and global parameters into one dictionary
    val_parameters_dict = combine_dictionaries(
            val_turbine_dict, val_global_param_dict)

    LOGGER.debug('Valuation Turbine Parameters: %s', val_parameters_dict)

    val_param_len = len(valuation_turbine_params) + len(valuation_global_params)
    if len(val_parameters_dict.keys()) != val_param_len:
        raise FieldError('An Error occured from reading in a field value from '
                'either the turbine CSV file or the global parameters JSON '
                'file. Please make sure all the necessary fields are present '
                'and spelled correctly.')

    LOGGER.debug('Turbine Dictionary: %s', val_parameters_dict)

    # Pixel size to be used in later calculations and raster creations
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(harvested_masked_uri)
    # URI for final distance transform used in valuation calculations
    tmp_dist_final_uri = os.path.join(
                inter_dir, 'val_distance_trans%s.tif' % suffix)

    # Handle Grid Points
    if 'grid_points_uri' in args:
        LOGGER.info('Grid Points Provided')
        LOGGER.info('Reading in the grid points')

        grid_file = open(args['grid_points_uri'], 'rU')
        reader = csv.DictReader(grid_file)

        grid_dict = {}
        land_dict = {}
        # Making a shallow copy of the attribute 'fieldnames' explicitly to
        # edit to all the fields to lowercase because it is more readable
        # and easier than editing the attribute itself
        field_names = reader.fieldnames

        for index in range(len(field_names)):
            field_names[index] = field_names[index].lower()

        # Iterate through the CSV file and construct two different dictionaries
        # for grid and land points.
        for row in reader:
            if row['type'].lower() == 'grid':
                grid_dict[row['id']] = row
            else:
                land_dict[row['id']] = row

        grid_file.close()

        # It's possible that no land points were provided, and we need to
        # handle both cases
        if land_dict:
            land_exists = True
        else:
            land_exists = False

        grid_ds_uri = os.path.join(inter_dir, 'val_grid_points%s.shp' % suffix)

        # Create a point shapefile from the grid point dictionary.
        # This makes it easier for future distance calculations and provides a
        # nice intermediate output for users
        pygeoprocessing.geoprocessing.dictionary_to_point_shapefile(
                grid_dict, 'grid_points', grid_ds_uri)

        # In case any of the above points lie outside the AOI, clip the
        # shapefiles and then project them to the AOI as well.
        # NOTE: There could be an error here where NO points lie within the AOI,
        # what then????????
        grid_projected_uri = os.path.join(
                inter_dir, 'grid_point_projected%s.shp' % suffix)
        clip_and_reproject_shapefile(grid_ds_uri, aoi_uri, grid_projected_uri)

        if land_exists:
            land_ds_uri = os.path.join(
                inter_dir, 'val_land_points%s.shp' % suffix)
            # Create a point shapefile from the land point dictionary.
            # This makes it easier for future distance calculations and
            # provides a nice intermediate output for users
            pygeoprocessing.geoprocessing.dictionary_to_point_shapefile(
                    land_dict, 'land_points', land_ds_uri)

            # In case any of the above points lie outside the AOI, clip the
            # shapefiles and then project them to the AOI as well.
            # NOTE: There could be an error here where NO points lie within
            # the AOI, what then????????
            land_projected_uri = os.path.join(
                    inter_dir, 'land_point_projected%s.shp' % suffix)
            clip_and_reproject_shapefile(
                land_ds_uri, aoi_uri, land_projected_uri)

            # Get the shortest distances from each grid point to the land
            # points
            grid_to_land_dist_local = point_to_polygon_distance(
                    grid_projected_uri, land_projected_uri)

            # Add the distances for land to grid points as a new field onto the
            # land points datasource
            LOGGER.debug(
                'Adding land to grid distances to land point datasource')
            land_to_grid_field = 'L2G'
            add_field_to_shape_given_list(
                    land_projected_uri, grid_to_land_dist_local,
                    land_to_grid_field)

            # Calculate distance raster
            calculate_distances_land_grid(
                land_projected_uri, harvested_masked_uri, tmp_dist_final_uri)
        else:
            # Calculate distance raster
            calculate_distances_grid(
                grid_projected_uri, harvested_masked_uri, tmp_dist_final_uri)
    else:
        LOGGER.info('Grid points not provided')
        LOGGER.debug('No grid points, calculating distances using land polygon')
        # Since the grid points were not provided use the land polygon to get
        # near shore distances
        # The average land cable distance in km converted to meters
        avg_grid_distance = float(args['avg_grid_distance']) * 1000.0

        land_poly_rasterized_uri = pygeoprocessing.geoprocessing.temporary_filename('.tif')
        # Create new raster and fill with 0s to set up for distance transform
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            harvested_masked_uri, land_poly_rasterized_uri, 'GTiff',
            out_nodata, gdal.GDT_Float32, fill_value=0.0)
        # Burn polygon features into raster with values of 1s to set up for
        # distance transform
        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            land_poly_rasterized_uri, land_poly_proj_uri, burn_values=[1.0],
            option_list=["ALL_TOUCHED=TRUE"])

        tmp_dist_uri = pygeoprocessing.geoprocessing.temporary_filename('.tif')
        pygeoprocessing.geoprocessing.distance_transform_edt(
            land_poly_rasterized_uri, tmp_dist_uri, process_pool=None)

        def add_avg_dist_op(tmp_dist):
            """vectorize_datasets operation to convert distances
                to meters and add in average grid to land distances

                tmp_dist - a numpy array of distances

                returns - distance values in meters with average grid to
                    land distance factored in
            """
            return np.where(
                tmp_dist != out_nodata,
                tmp_dist * pixel_size + avg_grid_distance, out_nodata)

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [tmp_dist_uri], add_avg_dist_op, tmp_dist_final_uri,
            gdal.GDT_Float32, out_nodata, pixel_size, 'intersection',
            vectorize_op=False)

    # Get constants from val_parameters_dict to make it more readable
    # The length of infield cable in km
    infield_length = float(val_parameters_dict['infield_cable_length'])
    # The cost of infield cable in millions of dollars per km
    infield_cost = float(val_parameters_dict['infield_cable_cost'])
    # The cost of the foundation in millions of dollars
    foundation_cost = float(args['foundation_cost'])
    # The cost of each turbine unit in millions of dollars
    unit_cost = float(val_parameters_dict['turbine_cost'])
    # The installation cost as a decimal
    install_cost = float(val_parameters_dict['installation_cost'])
    # The miscellaneous costs as a decimal factore of CAPEX
    misc_capex_cost = float(val_parameters_dict['miscellaneous_capex_cost'])
    # The operations and maintenance costs as a decimal factor of CAPEX
    op_maint_cost = float(val_parameters_dict['operation_maintenance_cost'])
    # The distcount rate as a decimal
    discount_rate = float(args['discount_rate'])
    # The cost to decommission the farm as a decmial factor of CAPEX
    decom = float(val_parameters_dict['decommission_cost'])
    # The mega watt value for the turbines in MW
    mega_watt = float(val_parameters_dict['turbine_rated_pwr'])
    # The distance at which AC switches over to DC power
    circuit_break = float(val_parameters_dict['ac_dc_distance_break'])
    # The coefficients for the AC/DC megawatt and cable cost from the CAP
    # function
    mw_coef_ac = float(val_parameters_dict['mw_coef_ac'])
    mw_coef_dc = float(val_parameters_dict['mw_coef_dc'])
    cable_coef_ac = float(val_parameters_dict['cable_coef_ac'])
    cable_coef_dc = float(val_parameters_dict['cable_coef_dc'])

    time = int(val_parameters_dict['time_period'])

    # If Price Table provided use that for price of energy
    if args["price_table"]:
        csv_file = open(args["wind_schedule"], 'rU')
        csv_reader = csv.DictReader(csv_file)
        price_dict = {}

        # Making a shallow copy of the attribute 'fieldnames' explicitly to
        # edit to all the fields to lowercase because it is more readable
        # and easier than editing the attribute itself
        field_names = csv_reader.fieldnames
        for index in range(len(field_names)):
            field_names[index] = field_names[index].lower()
        # Build up temporary dictionary for year and price
        for row in csv_reader:
            price_dict[int(row['year'])] = float(row['price'])
        csv_file.close()

        # Get the years or time steps and sort
        year_keys = price_dict.keys()
        year_keys.sort()

        if len(year_keys) != time + 1:
            raise TimePeriodError("The 'time' argument in the global parameter"
                "file must equal the number years provided in the table.")

        # Save the price values into a list where the indices of the list
        # indicate the time steps for the lifespand of the wind farm
        price_list = []
        for index in xrange(len(year_keys)):
            price_list.append(price_dict[year_keys[index]])
    else:
        change_rate = float(args["rate_change"])
        wind_price = float(args["wind_price"])
        # Build up a list of price values where the indices of the list
        # are the time steps for the lifespan of the farm and values
        # are adjusted based on the rate of change
        price_list = []
        for time_step in xrange(time + 1):
            price_list.append(wind_price * (1 + change_rate) ** (time_step))

    # The total mega watt compacity of the wind farm where mega watt is the
    # turbines rated power
    total_mega_watt = mega_watt * number_of_turbines

    # Total infield cable cost
    infield_cable_cost = infield_length * infield_cost * number_of_turbines
    LOGGER.debug('infield_cable_cost : %s', infield_cable_cost)

    # Total foundation cost
    total_foundation_cost = (foundation_cost + unit_cost) * number_of_turbines
    LOGGER.debug('total_foundation_cost : %s', total_foundation_cost)

    # Nominal Capital Cost (CAP) minus the cost of cable which needs distances
    cap_less_dist = infield_cable_cost + total_foundation_cost
    LOGGER.debug('cap_less_dist : %s', cap_less_dist)

    # Discount rate plus one to get that constant
    disc_const = discount_rate + 1.0
    LOGGER.debug('discount_rate : %s', disc_const)

    # Discount constant raised to the total time, a constant found in the NPV
    # calculation (1+i)^T
    disc_time = disc_const**time
    LOGGER.debug('disc_time : %s', disc_time)

    def calculate_npv_op(harvested_row, distance_row):
        """vectorize_datasets operation that computes the net present value

            harvested_row - a nd numpy array for wind harvested

            distance_row - a nd numpy array for distances

            returns - net present value
        """
        # Total cable distance converted to kilometers
        total_cable_dist = distance_row / 1000.0

        # The energy value converted from MWhr/yr (Mega Watt hours as output
        # from CK's biophysical model equations) to kWhr for the
        # valuation model
        energy_val = harvested_row * 1000.0

        # Initialize cable cost variable
        cable_cost = 0.0

        # The break at 'circuit_break' indicates the difference in using AC
        # and DC current systems
        cable_cost = np.where(total_cable_dist <= circuit_break,
            (mw_coef_ac * total_mega_watt) + (cable_coef_ac * total_cable_dist),
            (mw_coef_dc * total_mega_watt) + (cable_coef_dc * total_cable_dist))
        # Mask out nodata values
        cable_cost = np.where(
            harvested_row == out_nodata, out_nodata, cable_cost)

        # Compute the total CAP
        cap = cap_less_dist + cable_cost

        # Nominal total capital costs including installation and
        # miscellaneous costs (capex)
        capex = cap / (1.0 - install_cost - misc_capex_cost)

        # The ongoing cost of the farm
        ongoing_capex = op_maint_cost * capex

        # The cost to decommission the farm
        decommish_capex = decom * capex / disc_time

        # Variable to store the summation of the revenue less the
        # ongoing costs, adjusted for discount rate
        comp_one_sum = 0.0

        # Calculate the total NPV summation over the lifespan of the wind
        # farm. Starting at year 1, because year 0 yields no revenue
        for year in xrange(1, len(price_list)):
            # Dollar per kiloWatt hour
            dollar_per_kwh = float(price_list[year])

            # The price per kWh for energy converted to units of millions of
            #  dollars to correspond to the units for valuation costs
            mill_dollar_per_kwh = dollar_per_kwh / 1000000.0

            # The revenue in millions of dollars for the wind farm. The
            # energy_val is in kWh the farm.
            rev = energy_val * mill_dollar_per_kwh

            # Calculate the first component summation of the NPV equation
            comp_one_sum = (
                comp_one_sum + (rev - ongoing_capex) / disc_const**year)

        return np.where(
            (harvested_row != out_nodata) & (distance_row != out_nodata),
            comp_one_sum - decommish_capex - capex, out_nodata)

    def calculate_levelized_op(harvested_row, distance_row):
        """vectorize_datasets operation that computes the levelized cost

            harvested_row - a nd numpy array for wind harvested

            distance_row - a nd numpy array for distances

            returns - the levelized cost
        """
        # Total cable distance converted to kilometers
        total_cable_dist = distance_row / 1000.0

        # The energy value converted from MWhr/yr (Mega Watt hours as output
        # from CK's biophysical model equations) to kWhr for the
        # valuation model
        energy_val = harvested_row * 1000.0

        # Initialize cable cost variable
        cable_cost = 0.0

        # The break at 'circuit_break' indicates the difference in using AC
        # and DC current systems
        cable_cost = np.where(total_cable_dist <= circuit_break,
            (mw_coef_ac * total_mega_watt) + (cable_coef_ac * total_cable_dist),
            (mw_coef_dc * total_mega_watt) + (cable_coef_dc * total_cable_dist))
        # Mask out nodata values
        cable_cost = np.where(
            harvested_row == out_nodata, out_nodata, cable_cost)

        # Compute the total CAP
        cap = cap_less_dist + cable_cost

        # Nominal total capital costs including installation and
        # miscellaneous costs (capex)
        capex = cap / (1.0 - install_cost - misc_capex_cost)

        # The ongoing cost of the farm
        ongoing_capex = op_maint_cost * capex

        # The cost to decommission the farm
        decommish_capex = decom * capex / disc_time

        # Variable to store the numerator summation part of the
        # levelized cost
        levelized_cost_sum = 0.0
        # Variable to store the denominator summation part of the
        # levelized cost
        levelized_cost_denom = 0.0

        # Calculate the denominator summation value for levelized
        # cost of energy at year 0
        levelized_cost_denom = levelized_cost_denom + (
                energy_val / disc_const**0)

        # Calculate the levelized cost over the lifespan of the farm
        for year in xrange(1, len(price_list)):
            # Calculate the numerator summation value for levelized
            # cost of energy
            levelized_cost_sum = levelized_cost_sum + (
                    (ongoing_capex / disc_const**year))

            # Calculate the denominator summation value for levelized
            # cost of energy
            levelized_cost_denom = levelized_cost_denom + (
                    energy_val / disc_const**year)

        # Calculate the levelized cost of energy
        levelized_cost = ((levelized_cost_sum + decommish_capex + capex) /
                levelized_cost_denom)

        # Levelized cost of energy converted from millions of dollars to
        # dollars
        return np.where(harvested_row == out_nodata,
            out_nodata, levelized_cost * 1000000.0)

    # The amount of CO2 not released into the atmosphere, with the
    # constant conversion factor provided in the users guide by
    # Rob Griffin
    carbon_coef = float(val_parameters_dict['carbon_coefficient'])

    def calculate_carbon_op(harvested_row):
        """vectorize_dataset operation to calculate the carbon offset

            harvested_row - a nd numpy array

            returns - a nd numpy array of carbon offset values
        """
        # The energy value converted from MWhr/yr (Mega Watt hours as output
        # from CK's biophysical model equations) to kWhr for the
        # valuation model
        energy_val = harvested_row * 1000.0

        return np.where(
            harvested_row == out_nodata, out_nodata, carbon_coef * energy_val)

    # URIs for output rasters
    npv_uri = os.path.join(out_dir, 'npv_US_millions%s.tif' % suffix)
    levelized_uri = os.path.join(
            out_dir, 'levelized_cost_price_per_kWh%s.tif' % suffix)
    carbon_uri = os.path.join(out_dir, 'carbon_emissions_tons%s.tif' % suffix)

    pygeoprocessing.geoprocessing.vectorize_datasets(
                [harvested_masked_uri, tmp_dist_final_uri], calculate_npv_op,
                npv_uri, gdal.GDT_Float32, out_nodata, pixel_size,
                'intersection', vectorize_op=False)

    pygeoprocessing.geoprocessing.vectorize_datasets(
                [harvested_masked_uri, tmp_dist_final_uri],
                calculate_levelized_op, levelized_uri, gdal.GDT_Float32,
                out_nodata, pixel_size, 'intersection', vectorize_op=False)

    pygeoprocessing.geoprocessing.vectorize_datasets(
                [harvested_masked_uri], calculate_carbon_op, carbon_uri,
                gdal.GDT_Float32, out_nodata, pixel_size, 'intersection',
                vectorize_op=False)
    LOGGER.info('Wind Energy Valuation Model Complete')

def add_field_to_shape_given_list(shape_ds_uri, value_list, field_name):
    """Adds a field and a value to a given shapefile from a list of values. The
        list of values must be the same size as the number of features in the
        shape

        shape_ds_uri - a URI to an OGR datasource

        value_list - a list of values that is the same length as there are
            features in 'shape_ds'

        field_name - a String for the name of the new field

        returns - nothing"""
    LOGGER.debug('Entering add_field_to_shape_given_list')
    shape_ds = ogr.Open(shape_ds_uri, 1)
    layer = shape_ds.GetLayer()

    # Create new field
    LOGGER.debug('Creating new field')
    new_field = ogr.FieldDefn(field_name, ogr.OFTReal)
    layer.CreateField(new_field)

    # Iterator for indexing into array
    value_iterator = 0
    LOGGER.debug('Length of value_list : %s', len(value_list))
    LOGGER.debug('Feature Count : %s', layer.GetFeatureCount())
    LOGGER.debug('Adding values to new field for each point')
    for feat in layer:
        field_index = feat.GetFieldIndex(field_name)
        feat.SetField(field_index, value_list[value_iterator])
        layer.SetFeature(feat)
        value_iterator = value_iterator + 1

    layer.SyncToDisk()
    shape_ds = None

def point_to_polygon_distance(poly_ds_uri, point_ds_uri):
    """Calculates the distances from points in a point geometry shapefile to the
        nearest polygon from a polygon shapefile. Both datasources must be
        projected in meters

        poly_ds_uri - a URI to an OGR polygon geometry datasource projected in
            meters
        point_ds_uri - a URI to an OGR point geometry datasource projected in
            meters

        returns - a list of the distances from each point"""
    poly_ds = ogr.Open(poly_ds_uri)
    point_ds = ogr.Open(point_ds_uri)

    poly_layer = poly_ds.GetLayer()
    # List to store the polygons geometries as shapely objects
    poly_list = []

    LOGGER.debug('Loading the polygons into Shapely')
    for poly_feat in poly_layer:
        # Get the geometry of the polygon in WKT format
        poly_wkt = poly_feat.GetGeometryRef().ExportToWkt()
        # Load the geometry into shapely making it a shapely object
        shapely_polygon = shapely.wkt.loads(poly_wkt)
        # Add the shapely polygon geometry to a list, but first simplify the
        # geometry which smooths the edges making operations a lot faster
        poly_list.append(
                shapely_polygon.simplify(0.01, preserve_topology=False))

    # Take the union over the list of polygons to get one defined polygon object
    LOGGER.debug('Get the collection of polygon geometries by taking the union')
    polygon_collection = shapely.ops.unary_union(poly_list)

    point_layer = point_ds.GetLayer()
    # List to store the shapely point objects
    point_list = []

    LOGGER.debug('Loading the points into shapely')
    for point_feat in point_layer:
        # Get the geometry of the point in WKT format
        point_wkt = point_feat.GetGeometryRef().ExportToWkt()
        # Load the geometry into shapely making it a shapely object
        shapely_point = shapely.wkt.loads(point_wkt)
        # Add the point to a list to iterate through
        point_list.append(shapely_point)

    LOGGER.debug('find distances')
    distances = []
    for point in point_list:
        # Get the distance in meters and convert to km
        point_dist = point.distance(polygon_collection) / 1000.0
        # Add the distances to a list
        distances.append(point_dist)

    LOGGER.debug('Distance List Length : %s', len(distances))

    point_ds = None
    poly_ds = None

    return distances

def read_csv_wind_parameters(csv_uri, parameter_list):
    """Construct a dictionary from a csv file given a list of keys in
        'parameter_list'. The list of keys corresponds to the parameters names
        in 'csv_uri' which are represented in the first column of the file.

        csv_uri - a URI to a CSV file where every row is a parameter with the
            parameter name in the first column followed by the value in the
            second column

        parameter_list - a List of Strings that represent the parameter names to
            be found in 'csv_uri'. These Strings will be the keys in the
            returned dictionary

        returns - a Dictionary where the the 'parameter_list' Strings are the
            keys that have values pulled from 'csv_uri'
    """
    csv_file = open(csv_uri, 'rU')
    csv_reader = csv.reader(csv_file)
    output_dict = {}

    for csv_row in csv_reader:
        # Only get the biophysical parameters and leave out the valuation ones
        if csv_row[0].lower() in parameter_list:
            output_dict[csv_row[0].lower()] = csv_row[1]

    csv_file.close()
    return output_dict

def combine_dictionaries(dict_1, dict_2):
    """Add dict_2 to dict_1 and return in a new dictionary. Both dictionaries
        should be single level with a key that points to a value. If there is a
        key in 'dict_2' that already exists in 'dict_1' it will be ignored.

        dict_1 - a python dictionary
            ex: {'ws_id':1, 'vol':65}

        dict_2 - a python dictionary
            ex: {'size':11, 'area':5}

        returns - a python dictionary that is the combination of 'dict_1' and
        'dict_2' ex:
            ex: {'ws_id':1, 'vol':65, 'area':5, 'size':11}
    """
    # Make a copy of dict_1 the dictionary we want to add on to
    dict_3 = dict_1.copy()
    # Iterate through dict_2, the dictionary we want to get new fields/values
    # from
    for key, value in dict_2.iteritems():
        # Ignore fields that already exist in dictionary we are adding to
        if not key in dict_3.keys():
            dict_3[key] = value

    return dict_3

def create_wind_farm_box(spat_ref, start_point, x_len, y_len, out_uri):
    """Create an OGR shapefile where the geometry is a set of lines

        spat_ref - a SpatialReference to use in creating the output shapefile
            (required)
        start_point - a tuple of floats indicating the first vertice of the
            line (required)
        x_len - an integer value for the length of the line segment in
            the X direction (required)
        y_len - an integer value for the length of the line segment in
            the Y direction (required)
        out_uri - a string representing the file path to disk for the new
            shapefile (required)

        return - nothing"""
    LOGGER.debug('Entering create_wind_farm_box')

    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(out_uri)

    # Create the layer name from the uri paths basename without the extension
    uri_basename = os.path.basename(out_uri)
    layer_name = os.path.splitext(uri_basename)[0].encode("utf-8")

    layer = datasource.CreateLayer(layer_name, spat_ref, ogr.wkbLineString)

    # Add a single ID field
    field = ogr.FieldDefn('id', ogr.OFTReal)
    layer.CreateField(field)

    # Create the 3 other points that will make up the vertices for the lines
    top_left = (start_point[0], start_point[1] + y_len)
    top_right = (start_point[0] + x_len, start_point[1] + y_len)
    bottom_right = (start_point[0] + x_len, start_point[1])

    # Create a new feature, setting the field and geometry
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(start_point[0], start_point[1])
    line.AddPoint(top_left[0], top_left[1])
    line.AddPoint(top_right[0], top_right[1])
    line.AddPoint(bottom_right[0], bottom_right[1])
    line.AddPoint(start_point[0], start_point[1])

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(line)
    feature.SetField(0, 1)
    layer.CreateFeature(feature)

    feature = None
    layer = None

    datasource.SyncToDisk()
    datasource = None

def get_highest_harvested_geom(wind_points_uri):
    """Find the point with the highest harvested value for wind energy and
        return its geometry

        wind_points_uri - a URI to an OGR Datasource of a point geometry
            shapefile for wind energy

        returns - the geometry of the point with the highest harvested value
    """

    wind_points = ogr.Open(wind_points_uri)
    layer = wind_points.GetLayer()

    # Initiate some variables to use
    geom = None
    harv_value = None
    high_harv_value = 0.0

    feature = layer.GetNextFeature()
    harv_index = feature.GetFieldIndex('Harv_MWhr')
    high_harv_value = feature.GetField(harv_index)
    feat_geom = feature.GetGeometryRef()
    geom = feat_geom.Clone()

    for feat in layer:
        harv_value = feat.GetField(harv_index)
        if harv_value > high_harv_value:
            high_harv_value = harv_value
            feat_geom = feat.GetGeometryRef()
            geom = feat_geom.Clone()

    wind_points = None

    return geom

def mask_by_distance(
        dataset_uri, min_dist, max_dist, out_nodata, dist_uri, mask_uri):
    """Given a raster whose pixels are distances, bound them by a minimum and
        maximum distance

        dataset_uri - a URI to a GDAL raster with distance values

        min_dist - an integer of the minimum distance allowed in meters

        max_dist - an integer of the maximum distance allowed in meters

        mask_uri - the URI output of the raster masked by distance values

        dist_uri - the URI output of the raster converted from distance
            transform ranks to distance values in meters

        out_nodata - the nodata value of the raster

        returns - nothing"""

    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(dataset_uri)
    dataset_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(dataset_uri)

    def dist_op(dist_pix):
        """Vectorize_dataset operation that multiplies distance
            transform values by a cell size, to get distances
            in meters"""
        return np.where(
            dist_pix == dataset_nodata, out_nodata, dist_pix * cell_size)

    def mask_op(dist_pix):
        """Vectorize_dataset operation to bound dist_pix values between
            two integer values"""
        return np.where(
            ((dist_pix >= max_dist) | (dist_pix <= min_dist)), out_nodata,
            dist_pix)

    pygeoprocessing.geoprocessing.vectorize_datasets(
            [dataset_uri], dist_op, dist_uri, gdal.GDT_Float32,
            out_nodata, cell_size, 'intersection',
            assert_datasets_projected = True, vectorize_op = False)

    pygeoprocessing.geoprocessing.vectorize_datasets(
            [dist_uri], mask_op, mask_uri, gdal.GDT_Float32,
            out_nodata, cell_size, 'intersection',
            assert_datasets_projected = True, vectorize_op = False)

def read_csv_wind_data(wind_data_uri, hub_height):
    """Unpack the csv wind data into a dictionary.

    Parameters:
        wind_data_uri (string): a path for the csv wind data file with header
            of: "LONG","LATI","LAM","K","REF"

        hub_height (int): the hub height to use for calculating weibell
            parameters and wind energy values

    Returns:
        A dictionary where the keys are lat/long tuples which point
            to dictionaries that hold wind data at that location.
    """

    LOGGER.debug('Entering read_wind_data')

    # Constant used in getting Scale value at hub height from reference height
    # values. See equation 3 in the users guide.
    alpha = 0.11

    wind_dict = {}

    # LONG, LATI, RAM, K, REF
    wind_file = open(wind_data_uri, 'rU')
    reader = csv.DictReader(wind_file)

    for row in reader:
        ref_height = float(row['REF'])
        ref_scale = float(row['LAM'])
        ref_shape = float(row['K'])
        # Calculate scale value at new hub height given reference values.
        # See equation 3 in users guide
        scale_value = (ref_scale * (hub_height / ref_height)**alpha)

        wind_dict[float(row['LATI']), float(row['LONG'])] = {
            'LONG': float(row['LONG']), 'LATI': float(row['LATI']),
            'LAM': scale_value, 'K': ref_shape, 'REF_LAM': ref_scale}

    wind_file.close()
    return wind_dict

def wind_data_to_point_shape(dict_data, layer_name, output_uri):
    """Given a dictionary of the wind data create a point shapefile that
        represents this data

        dict_data - a python dictionary with the wind data, where the keys are
            tuples of the lat/long coordinates:
            {
            (97, 43) : {'LATI':97, 'LONG':43, 'LAM':6.3, 'K':2.7, 'REF':10},
            (55, 51) : {'LATI':55, 'LONG':51, 'LAM':6.2, 'K':2.4, 'REF':10},
            (73, 47) : {'LATI':73, 'LONG':47, 'LAM':6.5, 'K':2.3, 'REF':10}
            }

        layer_name - a python string for the name of the layer

        output_uri - a uri for the output destination of the shapefile

        return - nothing"""

    LOGGER.debug('Entering wind_data_to_point_shape')

    # If the output_uri exists delete it
    if os.path.isfile(output_uri):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        driver.DeleteDataSource(output_uri)

    LOGGER.debug('Creating new datasource')
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_uri)

    # Set the spatial reference to WGS84 (lat/long)
    source_sr = osr.SpatialReference()
    source_sr.SetWellKnownGeogCS("WGS84")

    output_layer = output_datasource.CreateLayer(
            layer_name, source_sr, ogr.wkbPoint)

    # Construct a list of fields to add from the keys of the inner dictionary
    field_list = dict_data[dict_data.keys()[0]].keys()
    LOGGER.debug('field_list : %s', field_list)

    LOGGER.debug('Creating fields for the datasource')
    for field in field_list:
        output_field = ogr.FieldDefn(field, ogr.OFTReal)
        output_layer.CreateField(output_field)

    LOGGER.debug('Entering iteration to create and set the features')
    # For each inner dictionary (for each point) create a point
    for point_dict in dict_data.itervalues():
        latitude = float(point_dict['LATI'])
        longitude = float(point_dict['LONG'])
        # When projecting to WGS84, extents -180 to 180 are used for
        # longitude. In case input longitude is from -360 to 0 convert
        if longitude < -180:
            longitude += 360
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint_2D(longitude, latitude)

        output_feature = ogr.Feature(output_layer.GetLayerDefn())
        output_layer.CreateFeature(output_feature)

        for field_name in point_dict:
            field_index = output_feature.GetFieldIndex(field_name)
            output_feature.SetField(field_index, point_dict[field_name])

        output_feature.SetGeometryDirectly(geom)
        output_layer.SetFeature(output_feature)
        output_feature = None

    LOGGER.debug('Leaving wind_data_to_point_shape')
    output_datasource = None

def clip_and_reproject_raster(raster_uri, aoi_uri, projected_uri):
    """Clip and project a Dataset to an area of interest

        raster_uri - a URI to a gdal Dataset

        aoi_uri - a URI to a ogr DataSource of geometry type polygon

        projected_uri - a URI string for the output dataset to be written to
            disk

        returns - nothing"""

    LOGGER.debug('Entering clip_and_reproject_raster')
    # Get the AOIs spatial reference as strings in Well Known Text
    aoi_sr = pygeoprocessing.geoprocessing.get_spatial_ref_uri(aoi_uri)
    aoi_wkt = aoi_sr.ExportToWkt()

    # Get the Well Known Text of the raster
    raster_wkt = pygeoprocessing.geoprocessing.get_dataset_projection_wkt_uri(raster_uri)

    # Temporary filename for an intermediate step
    aoi_reprojected_uri = pygeoprocessing.geoprocessing.temporary_folder()

    # Reproject the AOI to the spatial reference of the raster so that the
    # AOI can be used to clip the raster properly
    pygeoprocessing.geoprocessing.reproject_datasource_uri(
            aoi_uri, raster_wkt, aoi_reprojected_uri)

    # Temporary URI for an intermediate step
    clipped_uri = pygeoprocessing.geoprocessing.temporary_filename()

    LOGGER.debug('Clipping dataset')
    pygeoprocessing.geoprocessing.clip_dataset_uri(
            raster_uri, aoi_reprojected_uri, clipped_uri, False)

    # Get a point from the clipped data object to use later in helping
    # determine proper pixel size
    raster_gt = pygeoprocessing.geoprocessing.get_geotransform_uri(clipped_uri)
    point_one = (raster_gt[0], raster_gt[3])

    # Create a Spatial Reference from the rasters WKT
    raster_sr = osr.SpatialReference()
    raster_sr.ImportFromWkt(raster_wkt)

    # A coordinate transformation to help get the proper pixel size of
    # the reprojected raster
    coord_trans = osr.CoordinateTransformation(raster_sr, aoi_sr)

    pixel_size = pixel_size_based_on_coordinate_transform_uri(
            clipped_uri, coord_trans, point_one)

    LOGGER.debug('Reprojecting dataset')
    # Reproject the raster to the projection of the AOI
    pygeoprocessing.geoprocessing.reproject_dataset_uri(
            clipped_uri, pixel_size[0], aoi_wkt, 'bilinear', projected_uri)

    LOGGER.debug('Leaving clip_and_reproject_dataset')

def clip_and_reproject_shapefile(shapefile_uri, aoi_uri, projected_uri):
    """Clip and project a DataSource to an area of interest

        shapefile_uri - a URI to a ogr Datasource

        aoi_uri - a URI to a ogr DataSource of geometry type polygon

        projected_uri - a URI string for the output shapefile to be written to
            disk

        returns - nothing"""

    LOGGER.debug('Entering clip_and_reproject_shapefile')
    # Get the AOIs spatial reference as strings in Well Known Text
    aoi_sr = pygeoprocessing.geoprocessing.get_spatial_ref_uri(aoi_uri)
    aoi_wkt = aoi_sr.ExportToWkt()

    # Get the Well Known Text of the shapefile
    shapefile_sr = pygeoprocessing.geoprocessing.get_spatial_ref_uri(shapefile_uri)
    shapefile_wkt = shapefile_sr.ExportToWkt()

    # Temporary URI for an intermediate step
    aoi_reprojected_uri = pygeoprocessing.geoprocessing.temporary_folder()

    # Reproject the AOI to the spatial reference of the shapefile so that the
    # AOI can be used to clip the shapefile properly
    pygeoprocessing.geoprocessing.reproject_datasource_uri(
            aoi_uri, shapefile_wkt, aoi_reprojected_uri)

    # Temporary URI for an intermediate step
    clipped_uri = pygeoprocessing.geoprocessing.temporary_folder()

    # Clip the shapefile to the AOI
    LOGGER.debug('Clipping datasource')
    clip_datasource(aoi_reprojected_uri, shapefile_uri, clipped_uri)

    # Reproject the clipped shapefile to that of the AOI
    LOGGER.debug('Reprojecting datasource')
    pygeoprocessing.geoprocessing.reproject_datasource_uri(clipped_uri, aoi_wkt, projected_uri)

    LOGGER.debug('Leaving clip_and_reproject_maps')

def clip_datasource(aoi_uri, orig_ds_uri, output_uri):
    """Clip an OGR Datasource of geometry type polygon by another OGR Datasource
        geometry type polygon. The aoi should be a shapefile with a layer
        that has only one polygon feature

        aoi_uri - a URI to an OGR Datasource that is the clipping bounding box

        orig_ds_uri - a URI to an OGR Datasource to clip

        out_uri - output uri path for the clipped datasource

        returns - Nothing"""

    LOGGER.debug('Entering clip_datasource')

    aoi_ds = ogr.Open(aoi_uri)
    orig_ds = ogr.Open(orig_ds_uri)

    orig_layer = orig_ds.GetLayer()
    aoi_layer = aoi_ds.GetLayer()

    LOGGER.debug('Creating new datasource')
    # Create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_uri)

    # Get the original_layer definition which holds needed attribute values
    original_layer_dfn = orig_layer.GetLayerDefn()

    # Create the new layer for output_datasource using same name and geometry
    # type from original_datasource as well as spatial reference
    output_layer = output_datasource.CreateLayer(
            original_layer_dfn.GetName(), orig_layer.GetSpatialRef(),
            original_layer_dfn.GetGeomType())

    # Get the number of fields in original_layer
    original_field_count = original_layer_dfn.GetFieldCount()

    LOGGER.debug('Creating new fields')
    # For every field, create a duplicate field and add it to the new
    # shapefiles layer
    for fld_index in range(original_field_count):
        original_field = original_layer_dfn.GetFieldDefn(fld_index)
        output_field = ogr.FieldDefn(
                original_field.GetName(), original_field.GetType())
        # NOT setting the WIDTH or PRECISION because that seems to be unneeded
        # and causes interesting OGR conflicts
        output_layer.CreateField(output_field)

    # Get the feature and geometry of the aoi
    aoi_feat = aoi_layer.GetFeature(0)
    aoi_geom = aoi_feat.GetGeometryRef()

    LOGGER.debug('Starting iteration over geometries')
    # Iterate over each feature in original layer
    for orig_feat in orig_layer:
        # Get the geometry for the feature
        orig_geom = orig_feat.GetGeometryRef()

        # Check to see if the feature and the aoi intersect. This will return a
        # new geometry if there is an intersection. If there is not an
        # intersection it will return an empty geometry or it will return None
        # and print an error to standard out
        intersect_geom = aoi_geom.Intersection(orig_geom)

        if not intersect_geom == None and not intersect_geom.IsEmpty():
            # Copy original_datasource's feature and set as new shapes feature
            output_feature = ogr.Feature(
                    feature_def=output_layer.GetLayerDefn())

            # Since the original feature is of interest add it's fields and
            # Values to the new feature from the intersecting geometries
            # The False in SetFrom() signifies that the fields must match
            # exactly
            output_feature.SetFrom(orig_feat, False)
            output_feature.SetGeometry(intersect_geom)
            output_layer.CreateFeature(output_feature)
            output_feature = None

    LOGGER.debug('Leaving clip_datasource')
    output_datasource = None

def calculate_distances_land_grid(land_shape_uri, harvested_masked_uri, tmp_dist_final_uri):
    """Creates a distance transform raster based on the shortest distances
        of each point feature in 'land_shape_uri' and each features
        'L2G' field.

        land_shape_uri - a URI to an OGR shapefile that has the desired
            features to get the distance from (required)

        harvested_masked_uri - a URI to a GDAL raster that is used to get
            the proper extents and configuration for new rasters

        tmp_dist_final_uri - a URI to a GDAL raster for the final
            distance transform raster output

        returns - Nothing
    """
    # Open the point shapefile and get the layer
    land_points = ogr.Open(land_shape_uri)
    land_pts_layer = land_points.GetLayer()
    # A list to hold the land to grid distances in order for each point
    # features 'L2G' field
    l2g_dist = []
    # A list to hold the individual distance transform URI's in order
    uri_list = []

    # Get nodata value from biophsyical output raster
    out_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(harvested_masked_uri)
    # Get pixel size
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(harvested_masked_uri)

    for feat in land_pts_layer:
        # Get the point features land to grid value and add it to the list
        field_index = feat.GetFieldIndex("L2G")
        l2g_dist.append(float(feat.GetField(field_index)))

        # Create a new shapefile with only one feature to burn onto a raster
        # in order to get the distance transform based on that one feature
        output_driver = ogr.GetDriverByName('ESRI Shapefile')
        tmp_uri = pygeoprocessing.geoprocessing.temporary_folder()
        output_datasource = output_driver.CreateDataSource(tmp_uri)

        # Get the original_layer definition which holds needed attribute values
        original_layer_dfn = land_pts_layer.GetLayerDefn()

        # Create the new layer for output_datasource using same name and
        # geometry type from original_datasource as well as spatial reference
        output_layer = output_datasource.CreateLayer(
                original_layer_dfn.GetName(), land_pts_layer.GetSpatialRef(),
                original_layer_dfn.GetGeomType())

        # Get the number of fields in original_layer
        original_field_count = original_layer_dfn.GetFieldCount()

        # For every field, create a duplicate field and add it to the new
        # shapefiles layer
        for fld_index in range(original_field_count):
            original_field = original_layer_dfn.GetFieldDefn(fld_index)
            output_field = ogr.FieldDefn(
                    original_field.GetName(), original_field.GetType())
            # NOT setting the WIDTH or PRECISION because that seems to be
            # unneeded and causes interesting OGR conflicts
            output_layer.CreateField(output_field)

        # Copy original_datasource's feature and set as new shapes feature
        output_feature = ogr.Feature(
                feature_def=output_layer.GetLayerDefn())

        # Since the original feature is of interest add it's fields and
        # Values to the new feature from the intersecting geometries
        # The False in SetFrom() signifies that the fields must match
        # exactly
        output_feature.SetFrom(feat, False)
        output_layer.CreateFeature(output_feature)

        output_feature = None
        output_layer = None
        output_datasource = None

        land_pts_rasterized_uri = pygeoprocessing.geoprocessing.temporary_filename('.tif')
        # Create a new raster based on a biophysical output and fill with 0's
        # to set up for distance transform
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            harvested_masked_uri, land_pts_rasterized_uri, 'GTiff',
            out_nodata, gdal.GDT_Float32, fill_value=0.0)
        # Burn single feature onto the raster with value of 1 to set up for
        # distance transform
        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            land_pts_rasterized_uri, tmp_uri, burn_values=[1.0],
            option_list=["ALL_TOUCHED=TRUE"])

        dist_uri = pygeoprocessing.geoprocessing.temporary_filename('.tif')
        pygeoprocessing.geoprocessing.distance_transform_edt(
            land_pts_rasterized_uri, dist_uri, process_pool=None)
        # Add each features distance transform result to list
        uri_list.append(dist_uri)

    def land_ocean_dist(*rasters):
        """vectorize_dataset operation to aggregate each features distance
            transform output and create one distance output that has the
            shortest distances combined with each features land to grid
            distance

            *rasters - a numpy array of numpy nd arrays

            returns - a nd numpy array of the shortest distances
        """
        # Get the shape of the incoming numpy arrays
        shape = rasters[0].shape
        # Create a numpy array of 1's with proper shape
        land_grid = np.ones(shape)
        # Initialize numpy array with land to grid distances from the first
        # array
        land_grid = land_grid * l2g_dist[0]
        # Initialize final minimum distances array to first rasters
        distances = rasters[0]
        # Get the length of rasters lists to use in iteration and
        # indexing
        size = len(rasters)

        for index in range(1, size):
            raster = rasters[index]
            # Get the land to grid distances corresponding to current
            # indexed raster
            new_dist = np.ones(shape)
            new_dist = new_dist * l2g_dist[index]
            # Create a mask to indicate minimum distances
            mask = raster < distances
            # Replace distance values with minimum
            distances = np.where(mask, raster, distances)
            # Replace land to grid distances based on mask
            land_grid = np.where(mask, new_dist, land_grid)

        # Convert to meters from number of pixels
        distances = distances * pixel_size
        # Return and add land to grid distances to final distances
        return distances + land_grid

    pygeoprocessing.geoprocessing.vectorize_datasets(
                uri_list, land_ocean_dist, tmp_dist_final_uri, gdal.GDT_Float32,
                out_nodata, pixel_size, 'intersection', vectorize_op=False)

def calculate_distances_grid(land_shape_uri, harvested_masked_uri, tmp_dist_final_uri):
    """Creates a distance transform raster from an OGR shapefile. The function
        first burns the features from 'land_shape_uri' onto a raster using
        'harvested_masked_uri' as the base for that raster. It then does a
        distance transform from those locations and converts from pixel
        distances to distance in meters.

        land_shape_uri - a URI to an OGR shapefile that has the desired
            features to get the distance from (required)

        harvested_masked_uri - a URI to a GDAL raster that is used to get
            the proper extents and configuration for new rasters

        tmp_dist_final_uri - a URI to a GDAL raster for the final
            distance transform raster output

        returns - Nothing
    """
    land_pts_rasterized_uri = pygeoprocessing.geoprocessing.temporary_filename('.tif')
    # Get nodata value to use in raster creation and masking
    out_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(harvested_masked_uri)
    # Get pixel size from biophysical output
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(harvested_masked_uri)
    # Create a new raster based on harvested_masked_uri and fill with 0's
    # to set up for distance transform
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(
        harvested_masked_uri, land_pts_rasterized_uri, 'GTiff',
        out_nodata, gdal.GDT_Float32, fill_value=0.0)
    # Burn features from land_shape_uri onto raster with values of 1 to
    # set up for distance transform
    pygeoprocessing.geoprocessing.rasterize_layer_uri(
        land_pts_rasterized_uri, land_shape_uri, burn_values=[1.0],
        option_list=["ALL_TOUCHED=TRUE"])

    tmp_dist_uri = pygeoprocessing.geoprocessing.temporary_filename('.tif')
    # Run distance transform
    pygeoprocessing.geoprocessing.distance_transform_edt(
        land_pts_rasterized_uri, tmp_dist_uri, process_pool=None)

    def dist_meters_op(tmp_dist):
        """vectorize_dataset operation that multiplies by the pixel size

            tmp_dist - an nd numpy array

            returns - an nd numpy array multiplied by a pixel size
        """
        return np.where(
            tmp_dist != out_nodata, tmp_dist * pixel_size, out_nodata)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [tmp_dist_uri], dist_meters_op, tmp_dist_final_uri, gdal.GDT_Float32,
        out_nodata, pixel_size, 'intersection', vectorize_op=False)


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
