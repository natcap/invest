"""InVEST Wave Energy Model Core Code"""
import math
import os
import logging
import struct
import itertools
import shutil
import tempfile

import numpy
import pandas
from rtree import index
import scipy
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

import taskgraph
import pygeoprocessing
from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": "Wave Energy",
    "module": __name__,
    "userguide_html": "wave_energy.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "wave_base_data_path": {
            "validation_options": {
                "exists": True,
            },
            "type": "directory",
            "required": True,
            "about": "Select the folder that has the packaged Wave Energy Data.",
            "name": "Wave Base Data Folder"
        },
        "analysis_area_path": {
            "validation_options": {
                "options": [
                    "West Coast of North America and Hawaii",
                    "East Coast of North America and Puerto Rico",
                    "North Sea 4 meter resolution",
                    "North Sea 10 meter resolution",
                    "Australia",
                    "Global"
                ]
            },
            "type": "option_string",
            "required": True,
            "about": (
                "A list of analysis areas for which the model can currently "
                "be run.  All the wave energy data needed for these areas "
                "are pre-packaged in the WaveData folder."),
            "name": "Analysis Area"
        },
        "aoi_path": {
            "validation_options": {
                "projected": True,
                "projection_units": "meters"
            },
            "type": "vector",
            "required": False,
            "about": (
                "An OGR-supported vector file containing a single polygon "
                "representing the area of interest.  This input is required "
                "for computing valuation and is recommended for biophysical "
                "runs as well.  The AOI should be projected in linear units "
                "of meters."),
            "name": "Area of Interest"
        },
        "machine_perf_path": {
            "type": "csv",
            "required": True,
            "about": (
                "A CSV Table that has the performance of a particular wave "
                "energy machine at certain sea state conditions."),
            "name": "Machine Performance Table"
        },
        "machine_param_path": {
            "validation_options": {
                "required_fields": ["name", "value", "note"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "A CSV Table that has parameter values for a wave energy "
                "machine.  This includes information on the maximum "
                "capacity of the device and the upper limits for wave height "
                "and period."),
            "name": "Machine Parameter Table"
        },
        "dem_path": {
            "type": "raster",
            "required": True,
            "about": (
                "A GDAL-supported raster file containing a digital elevation "
                "model dataset that has elevation values in meters.  Used to "
                "get the cable distance for wave energy transmission."),
            "name": "Global Digital Elevation Model"
        },
        "valuation_container": {
            "type": "boolean",
            "required": False,
            "about": "Indicates whether the model includes valuation",
            "name": "Valuation"
        },
        "land_gridPts_path": {
            "validation_options": {
                "required_fields": ['id', 'type', 'lat', 'long', 'location'],
            },
            "type": "csv",
            "required": "valuation_container",
            "about": (
                "A CSV Table that has the landing points and grid points "
                "locations for computing cable distances."),
            "name": "Grid Connection Points Table"
        },
        "machine_econ_path": {
            "validation_options": {
                'required_fields': ['name', 'value', 'note'],
            },
            "type": "csv",
            "required": "valuation_container",
            "about": (
                "A CSV Table that has the economic parameters for the wave "
                "energy machine."),
            "name": "Machine Economic Table"
        },
        "number_of_machines": {
            "validation_options": {
                "expression": "int(value) > 0"
            },
            "type": "number",
            "required": "valuation_container",
            "about": (
                "An integer for how many wave energy machines will be in the "
                "wave farm."),
            "name": "Number of Machines"
        }
    }
}


# Set nodata value and target_pixel_type for new rasters
_NODATA = float(numpy.finfo(numpy.float32).min) + 1.0
_TARGET_PIXEL_TYPE = gdal.GDT_Float32

# The life span (25 years) of a wave energy conversion facility.
_LIFE_SPAN = 25
# Sea water density constant (kg/m^3)
_SWD = 1028
# Gravitational acceleration (m/s^2)
_GRAV = 9.8
# Constant determining the shape of a wave spectrum (see users guide pg 23)
_ALFA = 0.86

# Depth field name to be added to the wave vector from DEM data
_DEPTH_FIELD = 'DEPTH_M'
# Captured wave energy field name to be added to the wave vector
_CAP_WE_FIELD = 'CAPWE_MWHY'
# Wave power field name to be added to the wave vector
_WAVE_POWER_FIELD = 'WE_kWM'

# Preexisting field names in the wave energy vector
_HEIGHT_FIELD = 'HSAVG_M'
_PERIOD_FIELD = 'TPAVG_S'

# Field names for storing distance in wave vector
_W2L_DIST_FIELD = 'W2L_MDIST'
_L2G_DIST_FIELD = 'L2G_MDIST'
_LAND_ID_FIELD = 'LAND_ID'

# Net Present Value (NPV) field for wave vector
_NPV_25Y_FIELD = 'NPV_25Y'
# Total captured wave energy field for wave vector
_CAPWE_ALL_FIELD = 'CAPWE_ALL'
# Units field for wave vector
_UNIT_FIELD = 'UNITS'

# Resampling method for target rasters
_TARGET_RESAMPLE_METHOD = 'near'

# Percentile values and units specified explicitly in the user's guide
_PERCENTILES = [25, 50, 75, 90]
_CAPWE_UNITS_SHORT = ' MWh/yr'
_CAPWE_UNITS_LONG = 'megawatt hours per year'
_WP_UNITS_SHORT = ' kW/m'
_WP_UNITS_LONG = 'wave power per unit width of wave crest length'
_NPV_UNITS_SHORT = ' US$'
_NPV_UNITS_LONG = 'thousands of US dollars'
_STARTING_PERC_RANGE = '1'

# Driver name for creating vector and raster files
_VECTOR_DRIVER_NAME = "ESRI Shapefile"
_RASTER_DRIVER_NAME = "GTiff"


class IntersectionError(Exception):
    """A custom error message for when the AOI does not intersect any wave
        data points.

    """
    pass


def execute(args):
    """Wave Energy.

    Executes both the biophysical and valuation parts of the wave energy model
    (WEM). Files will be written on disk to the intermediate and output
    directories. The outputs computed for biophysical and valuation include:
    wave energy capacity raster, wave power raster, net present value raster,
    percentile rasters for the previous three, and a point shapefile of the
    wave points with attributes.

    Args:
        workspace_dir (str): Where the intermediate and output folder/files
            will be saved. (required)
        wave_base_data_path (str): Directory location of wave base data
            including WAVEWATCH III (WW3) data and analysis area shapefile.
            (required)
        analysis_area_path (str): A string identifying the analysis area of
            interest. Used to determine wave data shapefile, wave data text
            file, and analysis area boundary shape. (required)
        aoi_path (str): A polygon OGR vector outlining a more detailed area
            within the analysis area. This vector should be projected with
            linear units being in meters. (required to run Valuation model)
        machine_perf_path (str): The path of a CSV file that holds the
            machine performance table. (required)
        machine_param_path (str): The path of a CSV file that holds the
            machine parameter table. (required)
        dem_path (str): The path of the Global Digital Elevation Model (DEM).
            (required)
        results_suffix (str): A python string of characters to append to each output
            filename (optional)
        valuation_container (boolean): Indicates whether the model includes
            valuation
        land_gridPts_path (str): A CSV file path containing the Landing and
            Power Grid Connection Points table. (required for Valuation)
        machine_econ_path (str): A CSV file path for the machine economic
            parameters table. (required for Valuation)
        number_of_machines (int): An integer specifying the number of machines
            for a wave farm site. (required for Valuation)
        n_workers (int): The number of worker processes to use for processing
            this model.  If omitted, computation will take place in the current
            process. (optional)

    """
    LOGGER.info('Starting the Wave Energy Model.')
    invalid_parameters = validate(args)
    if invalid_parameters:
        raise ValueError("Invalid parameters passed: %s" % invalid_parameters)

    # Create the Output and Intermediate directories if they do not exist.
    workspace = args['workspace_dir']
    output_dir = os.path.join(workspace, 'output')
    intermediate_dir = os.path.join(workspace, 'intermediate')
    utils.make_directories([intermediate_dir, output_dir])

    # Initialize a TaskGraph
    taskgraph_working_dir = os.path.join(
        intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_working_dir, n_workers)

    # Append a _ to the suffix if it's not empty and doesn't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Get the path for the DEM
    dem_path = args['dem_path']

    # Create a dictionary that stores the wave periods and wave heights as
    # arrays. Also store the amount of energy the machine produces
    # in a certain wave period/height state as a 2D array
    machine_perf_dict = {}
    machine_perf_data = pandas.read_csv(args['machine_perf_path'])
    # Get the wave period fields, starting from the second column of the table
    machine_perf_dict['periods'] = machine_perf_data.columns.values[1:]
    # Build up the height field by taking the first column of the table
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

    machine_param_dict = _machine_csv_to_dict(args['machine_param_path'])

    # Check if required column fields are entered in the land grid csv file
    if 'land_gridPts_path' in args:
        # Create a grid_land_data dataframe for later use in valuation
        grid_land_data = pandas.read_csv(args['land_gridPts_path'])
        required_col_names = ['ID', 'TYPE', 'LAT', 'LONG', 'LOCATION']
        grid_land_data, missing_grid_land_fields = _get_validated_dataframe(
            args['land_gridPts_path'], required_col_names)
        if missing_grid_land_fields:
            raise ValueError(
                'The following column fields are missing from the Grid '
                'Connection Points File: %s' % missing_grid_land_fields)

    if 'valuation_container' in args and args['valuation_container']:
        machine_econ_dict = _machine_csv_to_dict(args['machine_econ_path'])

    # Build up a dictionary of possible analysis areas where the key
    # is the analysis area selected and the value is a dictionary
    # that stores the related paths to the needed inputs
    wave_base_data_path = args['wave_base_data_path']
    analysis_dict = {
        'West Coast of North America and Hawaii': {
            'point_vector':
            os.path.join(wave_base_data_path, 'NAmerica_WestCoast_4m.shp'),
            'extract_vector':
            os.path.join(wave_base_data_path, 'WCNA_extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'NAmerica_WestCoast_4m.txt.bin')
        },
        'East Coast of North America and Puerto Rico': {
            'point_vector':
            os.path.join(wave_base_data_path, 'NAmerica_EastCoast_4m.shp'),
            'extract_vector':
            os.path.join(wave_base_data_path, 'ECNA_extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'NAmerica_EastCoast_4m.txt.bin')
        },
        'North Sea 4 meter resolution': {
            'point_vector':
            os.path.join(wave_base_data_path, 'North_Sea_4m.shp'),
            'extract_vector':
            os.path.join(wave_base_data_path, 'North_Sea_4m_Extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'North_Sea_4m.bin')
        },
        'North Sea 10 meter resolution': {
            'point_vector':
            os.path.join(wave_base_data_path, 'North_Sea_10m.shp'),
            'extract_vector':
            os.path.join(wave_base_data_path, 'North_Sea_10m_Extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'North_Sea_10m.bin')
        },
        'Australia': {
            'point_vector':
            os.path.join(wave_base_data_path, 'Australia_4m.shp'),
            'extract_vector':
            os.path.join(wave_base_data_path, 'Australia_Extract.shp'),
            'ww3_path':
            os.path.join(wave_base_data_path, 'Australia_4m.bin')
        },
        'Global': {
            'point_vector':
            os.path.join(wave_base_data_path, 'Global.shp'),
            'extract_vector':
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
    wave_seastate_bins = _binary_wave_data_to_dict(
        analysis_dict[analysis_area_path]['ww3_path'])
    analysis_area_points_path = analysis_dict[analysis_area_path][
        'point_vector']
    analysis_area_extract_path = analysis_dict[analysis_area_path][
        'extract_vector']

    # Remove the wave point shapefile if it exists
    wave_vector_path = os.path.join(intermediate_dir,
                                    'WEM_InputOutput_Pts%s.shp' % file_suffix)
    if os.path.isfile(wave_vector_path):
        os.remove(wave_vector_path)

    # Set the source projection for a coordinate transformation
    # to the input projection from the wave watch point shapefile
    analysis_area_sr = _get_vector_spatial_ref(analysis_area_points_path)

    # This if/else statement differentiates between having an AOI or doing
    # a broad run on all the wave points specified by args['analysis_area'].
    if 'aoi_path' not in args or not args['aoi_path']:
        LOGGER.info('AOI not provided.')

        # Make a copy of the wave point shapefile so that the original input is
        # not corrupted when we clip the vector
        _copy_vector_or_raster(analysis_area_points_path, wave_vector_path)

        # The path to a polygon shapefile that specifies the broader AOI
        aoi_vector_path = analysis_area_extract_path

        # Set the pixel size to that of DEM, to be used for creating rasters
        target_pixel_size = pygeoprocessing.get_raster_info(dem_path)[
            'pixel_size']

        # Create a coordinate transformation, because it is used below when
        # indexing the DEM
        aoi_sr = _get_vector_spatial_ref(aoi_vector_path)
        aoi_sr_wkt = aoi_sr.ExportToWkt()
        analysis_area_sr_wkt = analysis_area_sr.ExportToWkt()

    else:
        LOGGER.info('AOI provided.')
        aoi_vector_path = args['aoi_path']
        # Create a coordinate transformation from the projection of the given
        # wave energy point shapefile, to the AOI's projection
        aoi_sr = _get_vector_spatial_ref(aoi_vector_path)
        aoi_sr_wkt = aoi_sr.ExportToWkt()
        analysis_area_sr_wkt = analysis_area_sr.ExportToWkt()

        # Clip the wave data shapefile by the bounds provided from the AOI
        task_graph.add_task(
            func=_clip_vector_by_vector,
            args=(analysis_area_points_path, aoi_vector_path, wave_vector_path,
                  aoi_sr_wkt, intermediate_dir),
            target_path_list=[wave_vector_path],
            task_name='clip_wave_points_to_aoi')

        # Clip the AOI to the Extract shape to make sure the output results do
        # not show extrapolated values outside the bounds of the points
        aoi_clipped_to_extract_path = os.path.join(
            intermediate_dir,
            'aoi_clipped_to_extract_path%s.shp' % file_suffix)
        task_graph.add_task(
            func=_clip_vector_by_vector,
            args=(aoi_vector_path, analysis_area_extract_path,
                  aoi_clipped_to_extract_path, aoi_sr_wkt, intermediate_dir),
            target_path_list=[aoi_clipped_to_extract_path],
            task_name='clip_aoi_to_extract_data')

        # Replace the AOI path with the clipped AOI path
        aoi_vector_path = aoi_clipped_to_extract_path

        # Join here since we need pixel size for creating output rasters
        task_graph.join()

        # Get the size of the pixels in meters, to be used for creating
        # projected wave power and wave energy capacity rasters
        coord_trans, coord_trans_opposite = _get_coordinate_transformation(
            analysis_area_sr, aoi_sr)
        target_pixel_size = _pixel_size_helper(wave_vector_path, coord_trans,
                                               coord_trans_opposite, dem_path)

    LOGGER.debug('target_pixel_size: %s, target_projection: %s',
                 target_pixel_size, aoi_sr_wkt)

    # We do all wave power calculations by manipulating the fields in
    # the wave data shapefile, thus we need to add proper depth values
    # from the raster DEM
    LOGGER.info('Adding DEPTH_M field to the wave shapefile from the DEM')
    # Add the depth value to the wave points by indexing into the DEM dataset
    indexed_wave_vector_path = os.path.join(
        intermediate_dir, 'Indexed_WEM_InputOutput_Pts%s.shp' % file_suffix)
    index_depth_to_wave_vector_task = task_graph.add_task(
        func=_index_raster_value_to_point_vector,
        args=(wave_vector_path, dem_path, indexed_wave_vector_path,
              _DEPTH_FIELD),
        target_path_list=[indexed_wave_vector_path],
        task_name='index_depth_to_wave_vector')

    # Generate an interpolate object for wave_energy_capacity
    LOGGER.info('Interpolating machine performance table.')
    energy_interp = _wave_energy_interp(wave_seastate_bins, machine_perf_dict)

    # Create a dictionary with the wave energy capacity sums from each location
    LOGGER.info('Calculating Captured Wave Energy.')
    energy_cap = _wave_energy_capacity_to_dict(
        wave_seastate_bins, energy_interp, machine_param_dict)

    # Add wave energy and wave power fields to the shapefile for the
    # corresponding points
    LOGGER.info('Adding wave energy and power fields to the wave vector.')
    wave_energy_power_vector_path = os.path.join(
        intermediate_dir, 'Captured_WEM_InputOutput_Pts%s.shp' % file_suffix)
    create_wave_energy_and_power_raster_task = task_graph.add_task(
        func=_energy_and_power_to_wave_vector,
        args=(energy_cap, indexed_wave_vector_path,
              wave_energy_power_vector_path),
        target_path_list=[wave_energy_power_vector_path],
        task_name='get_wave_energy_and_power',
        dependent_task_list=[index_depth_to_wave_vector_task])

    # Intermediate/final output paths for wave energy and wave power rasters
    unclipped_energy_raster_path = os.path.join(
        intermediate_dir, 'unclipped_capwe_mwh%s.tif' % file_suffix)
    unclipped_power_raster_path = os.path.join(
        intermediate_dir, 'unclipped_wp_kw%s.tif' % file_suffix)
    interpolated_energy_raster_path = os.path.join(
        intermediate_dir, 'interpolated_capwe_mwh%s.tif' % file_suffix)
    interpolated_power_raster_path = os.path.join(
        intermediate_dir, 'interpolated_wp_kw%s.tif' % file_suffix)
    energy_raster_path = os.path.join(output_dir,
                                      'capwe_mwh%s.tif' % file_suffix)
    wave_power_raster_path = os.path.join(output_dir,
                                          'wp_kw%s.tif' % file_suffix)

    # Create blank rasters bounded by the vector of analysis area (AOI)
    LOGGER.info('Create wave power and energy rasters from AOI extent')
    create_unclipped_energy_raster_task = task_graph.add_task(
        func=pygeoprocessing.create_raster_from_vector_extents,
        args=(aoi_vector_path, unclipped_energy_raster_path, target_pixel_size,
              _TARGET_PIXEL_TYPE, _NODATA),
        target_path_list=[unclipped_energy_raster_path],
        task_name='create_unclipped_energy_raster')

    create_unclipped_power_raster_task = task_graph.add_task(
        func=pygeoprocessing.create_raster_from_vector_extents,
        args=(aoi_vector_path, unclipped_power_raster_path, target_pixel_size,
              _TARGET_PIXEL_TYPE, _NODATA),
        target_path_list=[unclipped_power_raster_path],
        task_name='create_unclipped_power_raster')

    # Interpolate wave energy and power from the wave vector over the rasters
    LOGGER.info('Interpolate wave power and wave energy capacity onto rasters')
    interpolate_energy_points_task = task_graph.add_task(
        func=_interpolate_vector_field_onto_raster,
        args=(wave_energy_power_vector_path, unclipped_energy_raster_path,
              interpolated_energy_raster_path, _CAP_WE_FIELD),
        target_path_list=[interpolated_energy_raster_path],
        task_name='interpolate_energy_points',
        dependent_task_list=[create_wave_energy_and_power_raster_task,
                             create_unclipped_energy_raster_task])

    interpolate_power_points_task = task_graph.add_task(
        func=_interpolate_vector_field_onto_raster,
        args=(wave_energy_power_vector_path, unclipped_power_raster_path,
              interpolated_power_raster_path, _WAVE_POWER_FIELD),
        target_path_list=[interpolated_power_raster_path],
        task_name='interpolate_power_points',
        dependent_task_list=[create_wave_energy_and_power_raster_task,
                             create_unclipped_power_raster_task])

    clip_energy_raster_task = task_graph.add_task(
        func=pygeoprocessing.mask_raster,
        args=((interpolated_energy_raster_path, 1), aoi_vector_path,
              energy_raster_path,),
        kwargs={'all_touched': True},
        target_path_list=[energy_raster_path],
        task_name='clip_energy_raster',
        dependent_task_list=[interpolate_energy_points_task])

    clip_power_raster_task = task_graph.add_task(
        func=pygeoprocessing.mask_raster,
        args=((interpolated_power_raster_path, 1), aoi_vector_path,
              wave_power_raster_path,),
        kwargs={'all_touched': True},
        target_path_list=[wave_power_raster_path],
        task_name='clip_power_raster',
        dependent_task_list=[interpolate_power_points_task])

    # Paths for wave energy and wave power percentile rasters
    wp_rc_path = os.path.join(output_dir, 'wp_rc%s.tif' % file_suffix)
    capwe_rc_path = os.path.join(output_dir, 'capwe_rc%s.tif' % file_suffix)

    # Create the percentile rasters for wave energy and wave power
    task_graph.add_task(
        func=_create_percentile_rasters,
        args=(energy_raster_path, capwe_rc_path, _CAPWE_UNITS_SHORT,
              _CAPWE_UNITS_LONG, _PERCENTILES, intermediate_dir),
        kwargs={'start_value': _STARTING_PERC_RANGE},
        target_path_list=[capwe_rc_path],
        task_name='create_energy_percentile_raster',
        dependent_task_list=[clip_energy_raster_task])

    task_graph.add_task(
        func=_create_percentile_rasters,
        args=(wave_power_raster_path, wp_rc_path, _WP_UNITS_SHORT,
              _WP_UNITS_LONG, _PERCENTILES, intermediate_dir),
        kwargs={'start_value': _STARTING_PERC_RANGE},
        target_path_list=[wp_rc_path],
        task_name='create_power_percentile_raster',
        dependent_task_list=[clip_power_raster_task])

    LOGGER.info('Completed Wave Energy Biophysical')

    if 'valuation_container' not in args or not args['valuation_container']:
        # The rest of the function is valuation, so we can quit now
        LOGGER.info('Valuation not selected')
        task_graph.close()
        task_graph.join()
        return
    else:
        LOGGER.info('Valuation selected')

    # Output path for landing point shapefile
    land_vector_path = os.path.join(
        output_dir, 'LandPts_prj%s.shp' % file_suffix)
    # Output path for grid point shapefile
    grid_vector_path = os.path.join(
        output_dir, 'GridPts_prj%s.shp' % file_suffix)

    grid_data = grid_land_data.loc[
        grid_land_data['TYPE'].str.lower() == 'grid']
    land_data = grid_land_data.loc[
        grid_land_data['TYPE'].str.lower() == 'land']

    grid_dict = grid_data.to_dict('index')
    land_dict = land_data.to_dict('index')

    # Make a point shapefile for grid points
    LOGGER.info('Creating Grid Points Vector.')
    create_grid_points_vector_task = task_graph.add_task(
        func=_dict_to_point_vector,
        args=(grid_dict, grid_vector_path, 'grid_points', analysis_area_sr_wkt,
              aoi_sr_wkt),
        target_path_list=[grid_vector_path],
        task_name='create_grid_points_vector')

    # Make a point shapefile for landing points.
    LOGGER.info('Creating Landing Points Vector.')
    create_land_points_vector_task = task_graph.add_task(
        func=_dict_to_point_vector,
        args=(land_dict, land_vector_path, 'land_points', analysis_area_sr_wkt,
              aoi_sr_wkt),
        target_path_list=[land_vector_path],
        task_name='create_land_points_vector')

    # Add new fields to the wave vector.
    final_wave_energy_power_vector_path = os.path.join(
        intermediate_dir, 'Final_WEM_InputOutput_Pts%s.shp' % file_suffix)
    add_target_fields_to_wave_vector_task = task_graph.add_task(
        func=_add_target_fields_to_wave_vector,
        args=(wave_energy_power_vector_path, land_vector_path,
              grid_vector_path, final_wave_energy_power_vector_path,
              machine_econ_dict, int(args['number_of_machines'])),
        target_path_list=[final_wave_energy_power_vector_path],
        task_name='add_fields_to_wave_vector',
        dependent_task_list=[create_wave_energy_and_power_raster_task,
                             create_land_points_vector_task,
                             create_grid_points_vector_task])

    # Intermediate path for the projected net present value raster
    inter_npv_raster_path = os.path.join(
        intermediate_dir, 'npv_not_clipped%s.tif' % file_suffix)
    # Path for the net present value percentile raster
    target_npv_rc_path = os.path.join(output_dir, 'npv_rc%s.tif' % file_suffix)
    # Output path for the projected net present value raster
    target_npv_raster_path = os.path.join(
        output_dir, 'npv_usd%s.tif' % file_suffix)

    LOGGER.info('Create NPV raster from wave vector and AOI extents.')
    create_npv_raster_task = task_graph.add_task(
        func=_create_npv_raster,
        args=(final_wave_energy_power_vector_path, aoi_vector_path,
              inter_npv_raster_path, target_npv_raster_path,
              target_pixel_size),
        target_path_list=[inter_npv_raster_path, target_npv_raster_path],
        task_name='create_npv_raster',
        dependent_task_list=[add_target_fields_to_wave_vector_task])

    LOGGER.info('Create percentile NPV raster.')
    task_graph.add_task(
        func=_create_percentile_rasters,
        args=(target_npv_raster_path, target_npv_rc_path, _NPV_UNITS_SHORT,
              _NPV_UNITS_LONG, _PERCENTILES, intermediate_dir),
        target_path_list=[target_npv_rc_path],
        task_name='create_npv_percentile_raster',
        dependent_task_list=[create_npv_raster_task])

    # Close Taskgraph
    task_graph.close()
    task_graph.join()
    LOGGER.info('End of Wave Energy Valuation.')


def _copy_vector_or_raster(base_file_path, target_file_path):
    """Make a copy of a vector or raster.

    Parameters:
        base_file_path (str): a path to the base vector or raster to be copied
            from.
        target_file_path (str): a path to the target copied vector or raster.

    Returns:
        None

    Raises:
        ValueError if the base file can't be opened by GDAL.

    """
    # Open the file as raster first
    source_dataset = gdal.OpenEx(base_file_path, gdal.OF_RASTER)
    target_driver_name = _RASTER_DRIVER_NAME
    if source_dataset is None:
        # File didn't open as a raster; assume it's a vector
        source_dataset = gdal.OpenEx(base_file_path, gdal.OF_VECTOR)
        target_driver_name = _VECTOR_DRIVER_NAME

        # Raise an exception if the file can't be opened by GDAL
        if source_dataset is None:
            raise ValueError(
                'File %s is neither a GDAL-compatible raster nor vector.'
                % base_file_path)

    driver = gdal.GetDriverByName(target_driver_name)
    driver.CreateCopy(target_file_path, source_dataset)
    source_dataset = None


def _interpolate_vector_field_onto_raster(
        base_vector_path, base_raster_path, target_interpolated_raster_path,
        field_name):
    """Interpolate a vector field onto a target raster.

    Copy the base raster to the target interpolated raster, so taskgraph could
    trace the file state correctly.

    Parameters:
        base_vector_path (str): a path to a base vector that has field_name to
            be interpolated.
        base_raster_path (str): a path to a base raster to make a copy from.
        target_interpolated_raster_path (str): a path to a target raster copied
            from the base raster and will have the interpolated values.
        field_name (str): a field name on the base vector whose values will be
            interpolated onto the target raster.

    Returns:
        None

    """
    _copy_vector_or_raster(base_raster_path, target_interpolated_raster_path)
    pygeoprocessing.interpolate_points(
        base_vector_path, field_name, (target_interpolated_raster_path, 1),
        _TARGET_RESAMPLE_METHOD)


def _create_npv_raster(
        base_wave_vector_path, base_aoi_vector_path, inter_npv_raster_path,
        target_npv_raster_path, target_pixel_size):
    """Generate final NPV raster from the wave vector and AOI extent.

    Parameters:
        base_wave_vector_path (str): a path to the wave vector that contains
            the NPV field.
        base_aoi_vector_path (str): a path to the AOI vector used for
            generating and clipping the NPV raster.
        inter_npv_raster_path (str): a path to the intermediate NPV raster
            generated based on AOI vector extents.
        target_npv_raster_path (str): a path to the target NPV raster that
            has the NPV value and is clipped from the AOI.
        target_pixel_size (tuple): a tuple of floats representing the target
            x and y pixel sizes.

    Returns:
        None

    """
    # Create a blank raster from the extents of the wave farm shapefile
    LOGGER.info('Creating NPV Raster From AOI Vector Extents')
    pygeoprocessing.create_raster_from_vector_extents(
        base_aoi_vector_path, inter_npv_raster_path, target_pixel_size,
        _TARGET_PIXEL_TYPE, _NODATA)

    # Interpolate the NPV values based on the dimensions and corresponding
    # points of the raster, then write the interpolated values to the raster
    LOGGER.info('Interpolating Net Present Value onto Raster.')
    pygeoprocessing.interpolate_points(
        base_wave_vector_path, _NPV_25Y_FIELD, (inter_npv_raster_path, 1),
        _TARGET_RESAMPLE_METHOD)

    # Clip the raster to the AOI vector
    LOGGER.info('Masking NPV raster with AOI vector.')
    pygeoprocessing.mask_raster(
        (inter_npv_raster_path, 1),
        base_aoi_vector_path,
        target_npv_raster_path,
        all_touched=True)


def _get_npv_results(captured_wave_energy, depth, number_of_machines,
                     wave_to_land_dist, land_to_grid_dist, machine_econ_dict):
    """Compute NPV, total captured wave energy, and units for wave point.

    Parameters:
        captured_wave_energy (double): the amount of captured wave energy for
            a wave machine.
        depth (double): the depth of that wave point.
        number_of_machines (int): the number of machines for a given wave point
        wave_to_land_dist (float): the shortest distance from the wave point to
            land points.
        land_to_grid_dist (float): the shortest distance from the wave point to
            grid points.
        machine_econ_dict (dict): a dictionary of keys from the first
            column of the machine economy CSV file and corresponding
            values from the `VALUE` column.

    Returns:
        npv_result (float): the sum of total NPV of all 25 years.
        capwe_all_result (float): the total captured wave energy per year.

    """
    # Extract the machine economic parameters
    cap_max = float(machine_econ_dict['capmax'])  # maximum capacity, kW
    capital_cost = float(machine_econ_dict['cc'])  # capital cost, $/kW
    cml = float(machine_econ_dict['cml'])  # cost of mooring lines, $ per m
    cul = float(machine_econ_dict['cul'])  # cost of underwater cable, $ per km
    col = float(machine_econ_dict['col'])  # cost of overland cable, $ per km
    omc = float(machine_econ_dict['omc'])  # operating & maintenance cost, $ per kWh
    price = float(machine_econ_dict['p'])  # price of electricity, $ per kWh
    smlpm = float(machine_econ_dict['smlpm'])  # slack-moored
    d_rate = float(machine_econ_dict['r'])  # discount rate

    capwe_all_result = captured_wave_energy * number_of_machines

    # Create a numpy array of length 25, filled with the captured wave
    # energy in kW/h. Represents the lifetime of this wave farm.
    # Note: Divide the result by 1000 to convert W/h to kW/h
    captured_we = numpy.ones(_LIFE_SPAN) * (int(captured_wave_energy) * 1000.0)
    # It is expected that there is no revenue from the first year
    captured_we[0] = 0

    # Compute values to determine NPV
    lenml = 3.0 * numpy.absolute(depth)
    install_cost = number_of_machines * cap_max * capital_cost
    mooring_cost = smlpm * lenml * cml * number_of_machines
    # Divide by 1000.0 to convert cul and col from [$ per km] to [$ per m]
    trans_cost = (wave_to_land_dist * cul / 1000.0) + (
        land_to_grid_dist * col / 1000.0)
    initial_cost = install_cost + mooring_cost + trans_cost
    annual_revenue = price * number_of_machines * captured_we
    annual_cost = omc * captured_we * number_of_machines

    # The first year's cost is the initial start up cost
    annual_cost[0] = initial_cost

    # Calculate the total NPV of total life span
    rho = 1.0 / (1.0 + d_rate)
    npv = numpy.power(rho, numpy.arange(_LIFE_SPAN)) * (
        annual_revenue - annual_cost)
    npv_result = numpy.sum(npv) / 1000.0  # Convert [$US] to [thousands of $US]

    return npv_result, capwe_all_result


def _add_target_fields_to_wave_vector(
        base_wave_vector_path, base_land_vector_path, base_grid_vector_path,
        target_wave_vector_path, machine_econ_dict, number_of_machines):
    """Add six target fields to the target wave point vector.

    The target fields are the distance from ocean to land, the distance
    from land to grid, the land point ID, NPV, total captured wave energy, and
    units.

    Parameters:
        base_wave_vector_path (str): a path to the wave point vector with
            fields to calculate distances and NPV.
        base_land_vector_path (str): a path to the land point vector to get
            the wave to land distances from.
        base_grid_vector_path (str): a path to the grid point vector to get
            the land to grid distances from.
        target_wave_vector_path (str): a path to the target wave point vector
            to create new fields in.
        machine_econ_dict (dict): a dictionary of keys from the first column of
            the machine economy CSV file and corresponding values from the
            `VALUE` column.
        number_of_machines (int): the number of machines for each wave point.

    Returns:
        None

    """
    _copy_vector_or_raster(base_wave_vector_path, target_wave_vector_path)
    target_wave_vector = gdal.OpenEx(
        target_wave_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_wave_layer = target_wave_vector.GetLayer(0)

    # Get the coordinates of points of wave, land, and grid vectors
    wave_point_list = _get_points_geometries(base_wave_vector_path)
    land_point_list = _get_points_geometries(base_land_vector_path)
    grid_point_list = _get_points_geometries(base_grid_vector_path)

    # Calculate the minimum distances between the relative point groups
    LOGGER.info('Calculating Min Distances from wave to land and from land '
                'to grid.')
    wave_to_land_dist_list, wave_to_land_id_list = _calculate_min_distances(
        wave_point_list, land_point_list)
    land_to_grid_dist_list, _ = _calculate_min_distances(
        land_point_list, grid_point_list)

    # Add target fields to the wave vector to store results
    for field in [_W2L_DIST_FIELD, _L2G_DIST_FIELD, _LAND_ID_FIELD,
                  _NPV_25Y_FIELD, _CAPWE_ALL_FIELD, _UNIT_FIELD]:
        field_defn = ogr.FieldDefn(field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        target_wave_layer.CreateField(field_defn)

    # For each feature in the shapefile add the corresponding distance
    # from wave_to_land_dist and land_to_grid_dist calculated above
    target_wave_layer.ResetReading()

    LOGGER.info('Calculating and adding new fields to wave layer.')
    for i, feat in enumerate(target_wave_layer):
        # Get corresponding distances and land ID for the wave point
        land_id = int(wave_to_land_id_list[i])
        wave_to_land_dist = wave_to_land_dist_list[i]
        land_to_grid_dist = land_to_grid_dist_list[land_id]

        # Set distance and land ID fields to the feature
        feat.SetField(_W2L_DIST_FIELD, wave_to_land_dist)
        feat.SetField(_L2G_DIST_FIELD, land_to_grid_dist)
        feat.SetField(_LAND_ID_FIELD, land_id)

        # Get depth and captured wave energy for calculating NPV, total
        # captured energy, and units
        captured_wave_energy = feat.GetFieldAsDouble(_CAP_WE_FIELD)
        depth = feat.GetFieldAsDouble(_DEPTH_FIELD)
        npv_result, capwe_all_result = _get_npv_results(
            captured_wave_energy, depth, number_of_machines,
            wave_to_land_dist, land_to_grid_dist, machine_econ_dict)

        feat.SetField(_NPV_25Y_FIELD, npv_result)
        feat.SetField(_CAPWE_ALL_FIELD, capwe_all_result)
        feat.SetField(_UNIT_FIELD, number_of_machines)

        target_wave_layer.SetFeature(feat)
        feat = None

    target_wave_layer = None
    target_wave_vector = None


def _get_validated_dataframe(csv_path, field_list):
    """Return a dataframe with upper cased fields, and a list of missing fields.

    Parameters:
        csv_path (str): path to the csv to be converted to a dataframe.
        field_list (list): a list of fields in string format.

    Returns:
        dataframe (pandas.DataFrame): from csv with upper-cased fields.
        missing_fields (list): missing fields as string format in dataframe.

    """
    dataframe = pandas.read_csv(csv_path)
    field_list = [field.upper() for field in field_list]
    dataframe.columns = [col_name.upper() for col_name in dataframe.columns]
    missing_fields = []
    for field in field_list:
        if field not in dataframe.columns:
            missing_fields.append(field)
    return dataframe, missing_fields


def _dict_to_point_vector(base_dict_data, target_vector_path, layer_name,
                          base_sr_wkt, target_sr_wkt):
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
        base_sr_wkt (str): the spatial reference of the data from
            base_dict_data in well-known text format.
        target_sr_wkt (str): target spatial reference in well-known text format

    Returns:
        None

    """
    # If the target_vector_path exists delete it
    if os.path.isfile(target_vector_path):
        driver = ogr.GetDriverByName(_VECTOR_DRIVER_NAME)
        driver.DeleteDataSource(target_vector_path)

    base_sr = osr.SpatialReference()
    base_sr.ImportFromWkt(base_sr_wkt)
    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(target_sr_wkt)
    # Get coordinate transformation from base spatial reference to target,
    # in order to transform wave points to target_sr
    coord_trans, _ = _get_coordinate_transformation(base_sr, target_sr)

    LOGGER.info('Creating new vector')
    output_driver = ogr.GetDriverByName(_VECTOR_DRIVER_NAME)
    output_vector = output_driver.CreateDataSource(target_vector_path)
    target_layer = output_vector.CreateLayer(str(layer_name), target_sr,
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
    for point_dict in base_dict_data.values():
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
            output_feature.SetField(field_name, point_dict[field_name])
        output_feature.SetGeometryDirectly(geom)
        target_layer.SetFeature(output_feature)
        output_feature = None

    output_vector = None
    LOGGER.info('Finished _dict_to_point_vector')


def _get_points_geometries(base_vector_path):
    """Retrieve the XY coordinates from a point shapefile.

    The X and Y values from each point feature in the vector are stored in pair
    as [x_location,y_location] in a numpy array.

    Parameters:
        base_vector_path (str): a path to an OGR vector file.

    Returns:
        an array of points, representing the geometry of each point in the
            shapefile.

    """
    points = []
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer(0)
    for feat in base_layer:
        x_location = float(feat.GetGeometryRef().GetX())
        y_location = float(feat.GetGeometryRef().GetY())
        points.append([x_location, y_location])
        feat = None
    base_layer = None
    base_vector = None

    return numpy.array(points)


def _calculate_min_distances(xy_1, xy_2):
    """Calculate the shortest distances and indexes of points in xy_1 to xy_2.

    For all points in xy_1, this function calculates the distance from point
    xy_1 to various points in xy_2, and stores the shortest distances found in
    a list min_dist. The function also stores the index from which ever point
    in xy_2 was closest, as an id in a list that corresponds to min_dist.

    Parameters:
        xy_1 (numpy.array): An array of points in the form [x,y]
        xy_2 (numpy.array): An array of points in the form [x,y]

    Returns:
        min_dist (numpy.array): An array of shortest distances for each point
            in xy_1 to xy_2.
        min_id (numpy.array): An array of indexes corresponding to the array
            of shortest distances (min_dist).

    """
    # Create two numpy array of zeros with length set to as many points in xy_1
    min_dist = numpy.zeros(len(xy_1))
    min_id = numpy.zeros(len(xy_1))

    # For all points xy_point in xy_1 calculate the distance from xy_point to
    # xy_2 and save the shortest distance found.
    for idx, xy_point in enumerate(xy_1):
        dists = numpy.sqrt(numpy.sum((xy_point - xy_2)**2, axis=1))
        min_dist[idx], min_id[idx] = dists.min(), dists.argmin()

    return min_dist, min_id


def _binary_wave_data_to_dict(wave_file_path):
    """Convert a pickled binary WW3 text file into a dictionary.

    The dictionary's keys are the corresponding (I,J) values and the value is
    a two-dimensional array representing a matrix of the number of hours a
    seastate occurs over a 5 year period. The row and column fields are
    extracted once and stored in the dictionary as well.

    Parameters:
        wave_file_path (str): path to a pickled binary WW3 file.

    Returns:
        wave_dict (dict): a dictionary of matrices representing hours of
            specific seastates, as well as the period and height ranges.
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
    # Create a key that hosts another dictionary where the matrix
    # representation of the seastate bins will be saved
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
        if not line:
            # end of file
            wave_dict['bin_matrix'][key] = numpy.array(wave_array)
            break

        if key is not None:
            wave_dict['bin_matrix'][key] = numpy.array(wave_array)

        # Clear out array
        wave_array = []

        key = struct.unpack('ii', line)

        for _ in itertools.repeat(None, n_rows):
            line = wave_file.read(n_cols * 4)
            array = list(struct.unpack('f' * n_cols, line))
            wave_array.append(array)

    wave_file.close()
    # Add row/col field to dictionary
    LOGGER.debug('WaveData col %s', wave_periods)
    wave_dict['periods'] = numpy.array(wave_periods, dtype='f')
    LOGGER.debug('WaveData row %s', wave_heights)
    wave_dict['heights'] = numpy.array(wave_heights, dtype='f')
    LOGGER.info('Finished extrapolating wave data to dictionary.')
    return wave_dict


def _machine_csv_to_dict(machine_csv_path):
    """Create a dictionary from the table in machine csv file.

    The dictionary's keys are the 'NAME' from the machine table and its values
    are from the corresponding 'VALUE' field. No need to check for missing
    columns since the file is validated by validate() function.

    Parameters:
        machine_csv_path (str): path to the input machine CSV file.

    Returns:
        machine_dict (dict): a dictionary of keys from the first column of the
            CSV file and corresponding values from the `VALUE` column.

    """
    machine_dict = {}
    machine_data = pandas.read_csv(machine_csv_path, index_col=0)
    # make columns and indexes lowercased
    machine_data.columns = machine_data.columns.str.lower()
    # remove underscore from the keys
    machine_data.index = machine_data.index.str.strip()
    machine_data.index = machine_data.index.str.lower()

    # drop NaN indexed rows in dataframe
    machine_data = machine_data[machine_data.index.notnull()]
    LOGGER.debug('machine_data dataframe from %s: %s' %
                 (machine_csv_path, machine_data))
    machine_dict = machine_data.to_dict('index')
    for key in machine_dict.keys():
        machine_dict[key] = machine_dict[key]['value']
    return machine_dict


def _get_vector_spatial_ref(base_vector_path):
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


def _get_coordinate_transformation(source_sr, target_sr):
    """Create coordinate transformations between two spatial references.

    One transformation is from source to target, and the other from target to
    source.

    Parameters:
        source_sr (osr.SpatialReference): A spatial reference
        target_sr (osr.SpatialReference): A spatial reference

    Returns:
        A tuple: coord_trans (source to target) and coord_trans_opposite
            (target to source)

    """
    coord_trans = osr.CoordinateTransformation(source_sr, target_sr)
    coord_trans_opposite = osr.CoordinateTransformation(target_sr, source_sr)
    return (coord_trans, coord_trans_opposite)


def _create_percentile_rasters(base_raster_path, target_raster_path,
                               units_short, units_long, percentile_list,
                               working_dir, start_value=None):
    """Create a percentile (quartile) raster based on the raster_dataset.

    An attribute table is also constructed for the raster_dataset that displays
    the ranges provided by taking the quartile of values.

    Parameters:
        base_raster_path (str): path to a GDAL raster with data of type
            integer
        target_raster_path (str): path to the destination of the new raster.
        units_short (str): The shorthand for the units of the raster values,
            ex: kW/m.
        units_long (str): The description of the units of the raster values,
            ex: wave power per unit width of wave crest length (kW/m).
        percentile_list (list): A list of the _PERCENTILES ranges,
            ex: [25, 50, 75, 90].
        start_value (str): The first value that goes to the first percentile
            range (start_value: percentile_one) (optional)

    Returns:
        None

    """
    LOGGER.info('Creating Percentile Rasters')
    temp_dir = tempfile.mkdtemp(dir=working_dir)

    # If the target_raster_path is already a file, delete it
    if os.path.isfile(target_raster_path):
        os.remove(target_raster_path)

    target_nodata = 255
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    base_nodata = base_raster_info['nodata'][0]
    base_dtype = base_raster_info['datatype']

    def _mask_below_start_value(array):
        valid_mask = (array != base_nodata) & (array >= float(start_value))
        result = numpy.empty_like(array)
        result[:] = base_nodata
        result[valid_mask] = array[valid_mask]
        return result

    if start_value is not None:
        masked_raster_path = os.path.join(
            temp_dir, os.path.basename(base_raster_path))
        pygeoprocessing.raster_calculator(
            [(base_raster_path, 1)], _mask_below_start_value,
            masked_raster_path, base_dtype, base_nodata)
        input_raster_path = masked_raster_path
    else:
        input_raster_path = base_raster_path

    # Get the percentile values for each percentile
    percentile_values = pygeoprocessing.raster_band_percentile(
        (input_raster_path, 1),
        os.path.join(temp_dir, 'percentile'),
        percentile_list)

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Get the percentile ranges as strings so that they can be added to the
    # output table. Also round them for readability.
    value_ranges = []
    rounded_percentiles = numpy.round(percentile_values, decimals=2)
    # Add the first range with the starting value if it exists
    if start_value:
        value_ranges.append('%s to %s' % (start_value, rounded_percentiles[0]))
    else:
        value_ranges.append('Less than or equal to %s' % rounded_percentiles[0])
    value_ranges += ['%s to %s' % (p, q) for (p, q) in
                     zip(rounded_percentiles[:-1], rounded_percentiles[1:])]
    # Add the last range to the range of values list
    value_ranges.append('Greater than %s' % rounded_percentiles[-1])
    LOGGER.debug('Range_values : %s', value_ranges)

    def raster_percentile(band):
        """Group the band pixels together based on _PERCENTILES, starting from 1.
        """
        valid_data_mask = (band != base_nodata)
        band[valid_data_mask] = numpy.searchsorted(
            percentile_values, band[valid_data_mask]) + 1
        band[~valid_data_mask] = target_nodata
        return band

    # Classify the pixels of raster_dataset into groups and write to output
    pygeoprocessing.raster_calculator([(base_raster_path, 1)],
                                      raster_percentile, target_raster_path,
                                      gdal.GDT_Byte, target_nodata)

    # Create percentile groups of how percentile ranges are classified
    percentile_groups = numpy.arange(1, len(percentile_values) + 2)

    # Get the pixel count for each group
    pixel_count = _count_pixels_groups(target_raster_path, percentile_groups)

    LOGGER.debug('Pixel_count: %s; Percentile_groups: %s' %
                 (pixel_count, percentile_groups))

    # Initialize a dictionary where percentile groups map to a string
    # of corresponding percentile ranges. Used to create RAT
    percentile_dict = {}
    for idx in range(len(percentile_groups)):
        percentile_dict[percentile_groups[idx]] = value_ranges[idx]
    value_range_field = 'Value Range (' + units_long + ',' + units_short + ')'
    _create_raster_attr_table(
        target_raster_path, percentile_dict, column_name=value_range_field)

    # Create a list of corresponding percentile ranges from the percentile list
    length = len(percentile_list)
    percentile_ranges = []
    first_range = '<' + str(percentile_list[0]) + '%'
    percentile_ranges.append(first_range)
    for idx in range(length - 1):
        percentile_ranges.append(
            str(percentile_list[idx]) + '-' +
            str(percentile_list[idx + 1]) + '%')
    # Add the last range to the percentile ranges list
    last_range = '>' + str(percentile_list[length - 1]) + '%'
    percentile_ranges.append(last_range)

    # Initialize a dictionary to map percentile groups to percentile range
    # string and pixel count. Used for creating CSV table
    column_names = [
        'Percentile Group', 'Percentile Range', value_range_field,
        'Pixel Count'
    ]
    table_dict = dict((col_name, []) for col_name in column_names)
    for idx in range(len(percentile_groups)):
        table_dict['Percentile Group'].append(percentile_groups[idx])
        table_dict['Percentile Range'].append(percentile_ranges[idx])
        table_dict[value_range_field].append(value_ranges[idx])
        table_dict['Pixel Count'].append(pixel_count[idx])

    table_df = pandas.DataFrame(table_dict)

    # Write dataframe to csv, with columns in designated sequence
    base_attribute_table_path = os.path.splitext(target_raster_path)[0]
    attribute_table_path = base_attribute_table_path + '.csv'
    table_df.to_csv(attribute_table_path, index=False, columns=column_names)


def _clip_vector_by_vector(base_vector_path, clip_vector_path,
                           target_clipped_vector_path, target_sr_wkt,
                           work_dir):
    """Clip Shapefile Layer by second Shapefile Layer.

    Clip a shapefile layer where the output Layer inherits the projection and
    fields from the original Shapefile.

    Parameters:
        base_vector_path (str): a path to a Shapefile on disk. This is
            the Layer to clip. Must have same spatial reference as
            'clip_vector_path'.
        clip_vector_path (str): a path to a Shapefile on disk. This is
            the Layer to clip to. Must have same spatial reference as
            'base_vector_path'
        target_clipped_vector_path (str): a path on disk to write the clipped
            shapefile to. Should end with a '.shp' extension.
        target_sr_wkt (str): projection for the target_clipped_vector.
        work_dir (str): path to directory for saving temporary output files.

    Returns:
        None

    """
    if os.path.isfile(target_clipped_vector_path):
        driver = ogr.GetDriverByName(_VECTOR_DRIVER_NAME)
        driver.DeleteDataSource(target_clipped_vector_path)

    # Create a temporary folder within work_dir for saving reprojected files
    temp_work_dir = tempfile.mkdtemp(dir=work_dir, prefix='reproject-')

    def reproject_vector(base_vector_path, target_sr_wkt, temp_work_dir):
        """Reproject the vector to target projection."""
        base_sr_wkt = pygeoprocessing.get_vector_info(base_vector_path)[
            'projection']

        if base_sr_wkt != target_sr_wkt:
            LOGGER.info(
                'Base and target projections are different. '
                'Reprojecting %s to %s.', base_vector_path, target_sr_wkt)

            # Create path for the reprojected shapefile
            reproject_base_vector_path = os.path.join(
                temp_work_dir,
                os.path.basename(base_vector_path).replace(
                    '.shp', '_projected.shp'))
            pygeoprocessing.reproject_vector(base_vector_path, target_sr_wkt,
                                             reproject_base_vector_path)
            # Replace the base shapefile path with the reprojected path
            base_vector_path = reproject_base_vector_path

        return base_vector_path

    reproject_base_vector_path = reproject_vector(base_vector_path,
                                                  target_sr_wkt, temp_work_dir)
    reproject_clip_vector_path = reproject_vector(clip_vector_path,
                                                  target_sr_wkt, temp_work_dir)

    base_vector = gdal.OpenEx(reproject_base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()
    base_layer_defn = base_layer.GetLayerDefn()

    clip_vector = gdal.OpenEx(reproject_clip_vector_path, gdal.OF_VECTOR)
    clip_layer = clip_vector.GetLayer()

    driver = ogr.GetDriverByName(_VECTOR_DRIVER_NAME)
    target_vector = driver.CreateDataSource(target_clipped_vector_path)
    target_layer = target_vector.CreateLayer(base_layer_defn.GetName(),
                                             base_layer.GetSpatialRef())
    base_layer.Clip(clip_layer, target_layer)

    # Add in a check to make sure the intersection didn't come back empty
    if target_layer.GetFeatureCount() == 0:
        raise IntersectionError(
            "Intersection ERROR: found no intersection between base vector %s "
            "and clip vector %s." % (base_vector_path, clip_vector_path))

    base_layer = None
    clip_layer = None
    target_layer = None
    base_vector = None
    clip_vector = None
    target_vector = None

    shutil.rmtree(temp_work_dir, ignore_errors=True)


def _wave_energy_interp(wave_data, machine_perf):
    """Generate an interpolation matrix representing the machine perf table.

    The matrix is generated using new ranges from wave watch data.

    Parameters:
        wave_data (dict): A dictionary holding the new x range (period) and
            y range (height) values for the interpolation. The dictionary has
            the following structure:
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


def _wave_energy_capacity_to_dict(wave_data, interp_z, machine_param):
    """Compute and save the wave energy capacity for each point to a dict.

    The dictionary keys are the points (i,j) and their corresponding value
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
        interp_z (np.array): A 2D array of the interpolated values for the
            machine performance table
        machine_param (dict): A dictionary containing the restrictions for the
            machines (CapMax, TpMax, HsMax)

    Returns:
        energy_cap (dict): key - wave point, value - the wave energy capacity

    """
    energy_cap = {}

    # Get the row,col fields (ranges) for the wave watch data
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

    # For all the wave watch points, multiply the occurrence matrix by the
    # interpolated machine performance matrix to get the captured wave energy
    for key, val in wave_data['bin_matrix'].items():
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
        # captured wave energy and convert kWh into MWh
        sum_we = (mult_matrix.sum() / 1000)
        energy_cap[key] = sum_we

    return energy_cap


def _index_raster_value_to_point_vector(
        base_point_vector_path, base_raster_path, target_point_vector_path,
        field_name):
    """Add the values of a raster to the field of vector point features.

    Values are recorded in the attribute field of the vector. Note: If a value
    is larger than or equal to 0, the feature will be deleted, since a wave
    energy point on land should not be used in calculations.

    Parameters:
        base_point_vector_path (str): a path to an OGR point vector file.
        base_raster_path (str): a path to a GDAL dataset.
        target_point_vector_path (str): a path to a shapefile that has the
            target field name in addition to the existing fields in the base
            point vector.
        field_name (str): the name of the new field that will be added to the
            point feature. An exception will be raised if this field has
            existed in the base point vector.

    Returns:
        None

    """
    _copy_vector_or_raster(base_point_vector_path, target_point_vector_path)

    target_vector = gdal.OpenEx(
        target_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_layer = target_vector.GetLayer()
    raster_gt = pygeoprocessing.get_raster_info(base_raster_path)[
        'geotransform']
    pixel_size_x, pixel_size_y, raster_min_x, raster_max_y = \
        abs(raster_gt[1]), abs(raster_gt[5]), raster_gt[0], raster_gt[3]

    # Create a new field for the VECTOR attribute
    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)

    # Raise an exception if the field name already exists in the vector
    exact_match = True
    if target_layer.FindFieldIndex(field_name, exact_match) == -1:
        target_layer.CreateField(field_defn)
    else:
        raise ValueError(
            "'%s' field should not have existed in the wave data shapefiles. "
            "Please make sure it's renamed or removed from the attribute table."
            % field_name)

    # Create coordinate transformation from vector to raster, to make sure the
    # vector points are in the same projection as raster
    raster_sr = osr.SpatialReference()
    raster_sr.ImportFromWkt(
        pygeoprocessing.get_raster_info(base_raster_path)['projection'])
    vector_sr = osr.SpatialReference()
    vector_sr.ImportFromWkt(
        pygeoprocessing.get_vector_info(target_point_vector_path)[
            'projection'])
    vector_coord_trans = osr.CoordinateTransformation(vector_sr, raster_sr)

    # Initialize an R-Tree indexing object with point geom from base_vector
    def generator_function():
        for feat in target_layer:
            fid = feat.GetFID()
            geom = feat.GetGeometryRef()
            geom_x, geom_y = geom.GetX(), geom.GetY()
            geom_trans_x, geom_trans_y, _ = vector_coord_trans.TransformPoint(
                geom_x, geom_y)
            yield (fid, (
                geom_trans_x, geom_trans_x, geom_trans_y, geom_trans_y), None)

    vector_idx = index.Index(generator_function(), interleaved=False)

    # For all the features (points) add the proper raster value
    encountered_fids = set()
    for block_info, block_matrix in pygeoprocessing.iterblocks(
            (base_raster_path, 1)):
        # Calculate block bounding box in decimal degrees
        block_min_x = raster_min_x + block_info['xoff'] * pixel_size_x
        block_max_x = raster_min_x + (
            block_info['win_xsize'] + block_info['xoff']) * pixel_size_x

        block_max_y = raster_max_y - block_info['yoff'] * pixel_size_y
        block_min_y = raster_max_y - (
            block_info['win_ysize'] + block_info['yoff']) * pixel_size_y

        # Obtain a list of vector points that fall within the block
        intersect_vectors = list(
            vector_idx.intersection(
                (block_min_x, block_max_x, block_min_y, block_max_y),
                objects=True))

        for vector in intersect_vectors:
            vector_fid = vector.id

            # Occasionally there will be points that intersect multiple block
            # bounding boxes (like points that lie on the boundary of two
            # blocks) and we don't want to double-count.
            if vector_fid in encountered_fids:
                continue

            vector_trans_x, vector_trans_y = vector.bbox[0], vector.bbox[1]

            # To get proper raster value we must index into the dem matrix
            # by getting where the point is located in terms of the matrix
            i = int((vector_trans_x - block_min_x) / pixel_size_x)
            j = int((block_max_y - vector_trans_y) / pixel_size_y)

            try:
                block_value = block_matrix[j][i]
            except IndexError:
                # It is possible for an index to be *just* on the edge of a
                # block and exceed the block dimensions.  If this happens,
                # pass on this point, as another block's bounding box should
                # catch this point instead.
                continue

            # There are cases where the DEM may be too coarse and thus a
            # wave energy point falls on land. If the raster value taken is
            # greater than or equal to zero we need to delete that point as
            # it should not be used in calculations
            encountered_fids.add(vector_fid)
            if block_value >= 0.0:
                target_layer.DeleteFeature(vector_fid)
            else:
                feat = target_layer.GetFeature(vector_fid)
                feat.SetField(field_name, float(block_value))
                target_layer.SetFeature(feat)
                feat = None

    # It is not enough to just delete a feature from the layer. The
    # database where the information is stored must be re-packed so that
    # feature entry is properly removed
    target_vector.ExecuteSQL('REPACK ' + target_layer.GetName())
    target_layer = None
    target_vector = None


def _energy_and_power_to_wave_vector(
        energy_cap, base_wave_vector_path, target_wave_vector_path):
    """Add captured wave energy value from energy_cap to a field in wave_vector.

    The values are set corresponding to the same I,J values which is the key of
    the dictionary and used as the unique identifier of the shape.

    Parameters:
        energy_cap (dict): a dictionary with keys (I,J), representing the
            wave energy capacity values.
        base_wave_vector_path (str): a path to a wave point shapefile with
            existing fields to copy from.
        target_wave_vector_path (str): a path to the wave point shapefile
            to write the new field/values to.

    Returns:
        None.

    """
    _copy_vector_or_raster(base_wave_vector_path, target_wave_vector_path)

    target_wave_vector = gdal.OpenEx(
        target_wave_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_wave_layer = target_wave_vector.GetLayer()
    # Create the Captured Energy and Wave Power fields for the shapefile
    for field_name in [_CAP_WE_FIELD, _WAVE_POWER_FIELD]:
        field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        target_wave_layer.CreateField(field_defn)

    # For all of the features (points) in the shapefile, get the corresponding
    # point/value from the dictionary and set the _CAP_WE_FIELD field as
    # the value from the dictionary
    for feat in target_wave_layer:
        # Calculate and set the Captured Wave Energy field
        value_i = feat.GetField('I')
        value_j = feat.GetField('J')
        we_value = energy_cap[(value_i, value_j)]

        feat.SetField(_CAP_WE_FIELD, we_value)

        # Calculate and set the Wave Power field
        height = feat.GetFieldAsDouble(_HEIGHT_FIELD)  # in meters
        period = feat.GetFieldAsDouble(_PERIOD_FIELD)
        depth = feat.GetFieldAsInteger(_DEPTH_FIELD)

        depth = numpy.absolute(depth)
        # wave frequency calculation (used to calculate wave number k)
        tem = (2.0 * math.pi) / (period * _ALFA)
        # wave number calculation (expressed as a function of
        # wave frequency and water depth)
        k = numpy.square(tem) / (_GRAV * numpy.sqrt(
            numpy.tanh((numpy.square(tem)) * (depth / _GRAV))))
        # Setting numpy overflow error to ignore because when numpy.sinh
        # gets a really large number it pushes a warning, but Rich
        # and Doug have agreed it's nothing we need to worry about.
        numpy.seterr(over='ignore')

        # wave group velocity calculation (expressed as a
        # function of wave energy period and water depth)
        wave_group_velocity = (((1 + (
            (2 * k * depth) / numpy.sinh(2 * k * depth))) * numpy.sqrt(
                (_GRAV / k) * numpy.tanh(k * depth))) / 2)

        # Reset the overflow error to print future warnings
        numpy.seterr(over='print')

        # Wave power calculation. Divide by 1000 to convert W/m to kW/m
        # Note: _SWD: Sea water density constant (kg/m^3),
        # _GRAV: Gravitational acceleration (m/s^2),
        # height: in m, wave_group_velocity: in m/s
        wave_pow = ((((_SWD * _GRAV) / 16) *
                     (numpy.square(height)) * wave_group_velocity) / 1000)

        feat.SetField(_WAVE_POWER_FIELD, wave_pow)

        # Save the feature modifications to the layer.
        target_wave_layer.SetFeature(feat)
        feat = None

    target_wave_layer = None
    target_wave_vector = None


def _count_pixels_groups(raster_path, group_values):
    """Count pixels for each value in 'group_values' over a raster.

    Parameters:
        raster_path (str): path to a GDAL raster on disk
        group_values (list): unique numbers for which to get a pixel count

    Returns:
        pixel_count (list): a list of integers, where each integer at an index
            corresponds to the pixel count of the value from 'group_values'

    """
    # Initialize a list that will hold pixel counts for each group
    pixel_count = numpy.zeros(len(group_values))

    for _, block_matrix in pygeoprocessing.iterblocks((raster_path, 1)):
        # Cumulatively add the pixels count for each value in 'group_values'
        for idx in range(len(group_values)):
            val = group_values[idx]
            count_mask = numpy.zeros(block_matrix.shape)
            numpy.equal(block_matrix, val, count_mask)
            pixel_count[idx] += numpy.count_nonzero(count_mask)

    return pixel_count


def _pixel_size_helper(base_vector_path, coord_trans, coord_trans_opposite,
                       base_raster_path):
    """Retrieve pixel size of a raster given a vector w/ certain projection.

    Parameters:
        base_vector_path (str): path to a shapefile indicating where in the
            world we are interested in
        coord_trans (osr.CoordinateTransformation): a coordinate transformation
        coord_trans_opposite (osr.CoordinateTransformation): a coordinate
            transformation in the opposite direction of 'coord_trans'
        base_raster_path (str): path to a raster to get the pixel size from

    Returns:
        pixel_size_tuple (tuple): x and y pixel sizes of the raster given in
            the units of what base vector is projected in

    """
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer(0)

    # Get a point in the clipped vector to determine output grid size
    feat = layer.GetNextFeature()
    geom = feat.GetGeometryRef()
    reference_point_x = geom.GetX()
    reference_point_y = geom.GetY()

    # Convert the point from meters to geom_x/long
    reference_point_latlng = coord_trans_opposite.TransformPoint(
        reference_point_x, reference_point_y)

    # Get the size of the pixels in meters, to be used for creating rasters
    pixel_size_x, pixel_size_y = _pixel_size_based_on_coordinate_transform(
        base_raster_path, coord_trans, reference_point_latlng)

    feat = None
    layer = None
    vector = None

    return (pixel_size_x, pixel_size_y)


def _pixel_size_based_on_coordinate_transform(base_raster_path, coord_trans,
                                              reference_point):
    """Get width and height of cell in meters.

    Calculates the pixel width and height in meters given a coordinate
    transform and reference point on the dataset that's close to the
    transformed projected coordinate system. This is only necessary if raster
    is not already in a meter coordinate system, for example raster may be in
    lat/long (WGS84).

    Args:
        base_raster_path (str): path to a GDAL raster path, projected in the
            form of lat/long decimal degrees
        coord_trans (osr.CoordinateTransformation): an OSR coordinate
            transformation from dataset coordinate system to meters
        reference_point (tuple): a reference point close to the transformed
            coordinate system. Must be in the same coordinate system as
            the base raster.

    Returns:
        pixel_diff (tuple): a 2-tuple containing (pixel width in meters, pixel
            height in meters)

    """
    # Get the first points (x, y) from geoTransform
    geotransform = pygeoprocessing.get_raster_info(base_raster_path)[
        'geotransform']
    pixel_size_x = geotransform[1]
    pixel_size_y = geotransform[5]
    top_left_x = reference_point[0]
    top_left_y = reference_point[1]
    # Create the second point by adding the pixel width/height
    new_x = top_left_x + pixel_size_x
    new_y = top_left_y + pixel_size_y
    # Transform two points into meters
    point_1 = coord_trans.TransformPoint(top_left_x, top_left_y)
    point_2 = coord_trans.TransformPoint(new_x, new_y)

    # Calculate the x/y difference between two points
    # taking the absolute value because the direction doesn't matter for pixel
    # size in the case of most coordinate systems where y increases up and x
    # increases to the right (right handed coordinate system).
    pixel_diff_x = point_2[0] - point_1[0]
    pixel_diff_y = point_2[1] - point_1[1]

    return (pixel_diff_x, pixel_diff_y)


def _create_raster_attr_table(base_raster_path, attr_dict, column_name):
    """Create a raster attribute table (RAT).

    Parameters:
        base_raster_path (str): a GDAL raster dataset to create the RAT for
        attr_dict (dict): a dictionary with keys that point to a primitive type
            ex: {integer_id_1: value_1, ... integer_id_n: value_n}
        column_name (str): a string for the column name that maps the values

    Returns:
        None

    """
    raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    band = raster.GetRasterBand(1)
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
    raster = None


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Wave Energy.

    Parameters:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        warnings (list): A list of tuples where tuple[0] is an iterable of keys
            that the error message applies to and tuple[1] is the string
            validation warning.

    """
    return validation.validate(args, ARGS_SPEC['args'])
