"""InVEST Wind Energy model."""
import logging
import math
import os
import pickle
import shutil
import tempfile

import numpy
import pandas
from scipy import integrate

import shapely.wkb
import shapely.wkt
import shapely.ops
import shapely.prepared
from shapely import speedups

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import pygeoprocessing
import taskgraph
from . import utils
from . import spec_utils
from .spec_utils import u
from . import validation
from .model_metadata import MODEL_METADATA
from . import gettext


LOGGER = logging.getLogger(__name__)
speedups.enable()

ARGS_SPEC = {
    "model_name": MODEL_METADATA["wind_energy"].model_title,
    "pyname": MODEL_METADATA["wind_energy"].pyname,
    "userguide": MODEL_METADATA["wind_energy"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ['aoi_vector_path', 'bathymetry_path',
                         'land_polygon_vector_path'],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "wind_data_path": {
            "type": "csv",
            "columns": {
                "long": {
                    "type": "number",
                    "units": u.degree,
                    "about": gettext("Longitude of the data point.")
                },
                "lati": {
                    "type": "number",
                    "units": u.degree,
                    "about": gettext("Latitude of the data point.")
                },
                "lam": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "Weibull scale factor at the reference hub height at "
                        "this point.")
                },
                "k": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("Weibull shape factor at this point.")
                },
                "ref": {
                    "type": "number",
                    "units": u.meter,
                    "about": gettext(
                        "The reference hub height at this point, at which "
                        "wind speed data was collected and LAM was estimated.")
                }
            },
            "about": gettext("Table of Weibull parameters for each wind data point."),
            "name": gettext("wind data points")
        },
        "aoi_vector_path": {
            **spec_utils.AOI,
            "projected": True,
            "projection_units": u.meter,
            "required": "valuation_container & grid_points_path",
            "about": gettext(
                "Map of the area(s) of interest over which to run the model "
                "and aggregate valuation results. Required if Run Valuation "
                "is selected and the Grid Connection Points table is provided."
            )
        },
        "bathymetry_path": {
            "type": "raster",
            "bands": {1: {"type": "number", "units": u.meter}},
            "about": gettext("Map of ocean depth. Values should be negative."),
            "name": gettext("bathymetry")
        },
        "land_polygon_vector_path": {
            "type": "vector",
            "fields": {},
            "geometries": {"POLYGON", "MULTIPOLYGON"},
            "required": "min_distance | max_distance | valuation_container",
            "about": gettext(
                "Map of the coastlines of landmasses in the area of interest. "
                "Required if the Minimum Distance and Maximum Distance inputs "
                "are provided."),
            "name": gettext("land polygon")
        },
        "global_wind_parameters_path": {
            "type": "csv",
            "rows": {
                "air_density": {
                    "type": "number",
                    "units": u.kilogram/(u.meter**3),
                    "about": gettext("Standard atmosphere air density.")},
                "exponent_power_curve": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("Exponent to use in the power curve function.")},
                "decommission_cost": {
                    "type": "ratio",
                    "about": gettext(
                        "Cost to decommission a turbine as a proportion of "
                        "the total upfront costs (cables, foundations, "
                        "installation?)")
                },
                "operation_maintenance_cost": {
                    "type": "ratio",
                    "about": gettext(
                        "The operations and maintenance costs as a proportion "
                        "of capex_arr")},
                "miscellaneous_capex_cost": {
                    "type": "ratio",
                    "about": gettext(
                        "The miscellaneous costs as a proportion of capex_arr")
                },
                "installation_cost": {
                    "type": "ratio",
                    "about": gettext(
                        "The installation costs as a proportion of capex_arr")
                },
                "infield_cable_length": {
                    "type": "number",
                    "units": u.kilometer,
                    "about": gettext("The length of infield cable.")},
                "infield_cable_cost": {
                    "type": "number",
                    "units": u.currency/u.kilometer,
                    "about": gettext("The cost of infield cable.")},
                "mw_coef_ac": {
                    "type": "number",
                    "units": u.currency/u.megawatt,
                    "about": gettext("Cost of AC cable that scales with capacity.")},
                "mw_coef_dc": {
                    "type": "number",
                    "units": u.currency/u.megawatt,
                    "about": gettext("Cost of DC cable that scales with capacity.")},
                "cable_coef_ac": {
                    "type": "number",
                    "units": u.currency/u.kilometer,
                    "about": gettext("Cost of AC cable that scales with length.")},
                "cable_coef_dc": {
                    "type": "number",
                    "units": u.currency/u.kilometer,
                    "about": gettext("Cost of DC cable that scales with length.")},
                "ac_dc_distance_break": {
                    "type": "number",
                    "units": u.kilometer,
                    "about": gettext(
                        "The threshold above which a wind farm’s distance "
                        "from the grid requires a switch from AC to DC power "
                        "to overcome line losses which reduce the amount of "
                        "energy delivered")},
                "time_period": {
                    "type": "number",
                    "units": u.year,
                    "about": gettext("The expected lifetime of the facility")},
                "carbon_coefficient": {
                    "type": "number",
                    "units": u.metric_ton/u.kilowatt_hour,
                    "about": gettext(
                        "Factor that translates carbon-free wind power to a "
                        "corresponding amount of avoided CO2 emissions")},
                "air_density_coefficient": {
                    "type": "number",
                    "units": u.kilogram/(u.meter**3 * u.meter),
                    "about": gettext(
                        "The reduction in air density per meter above sea "
                        "level")},
                "loss_parameter": {
                    "type": "ratio",
                    "about": gettext(
                        "The fraction of energy lost due to downtime, power "
                        "conversion inefficiency, and electrical grid losses")}
            },
            "about": gettext(
                "A table of wind energy infrastructure parameters."),
            "name": gettext("global wind energy parameters")
        },
        "turbine_parameters_path": {
            "type": "csv",
            "rows": {
                "hub_height": {
                    "type": "number",
                    "units": u.meter,
                    "about": gettext("Height of the turbine hub above sea level.")},
                "cut_in_wspd": {
                    "type": "number",
                    "units": u.meter/u.second,
                    "about": gettext(
                        "Wind speed at which the turbine begins producing "
                        "power.")},
                "rated_wspd": {
                    "type": "number",
                    "units": u.meter/u.second,
                    "about": gettext(
                        "Minimum wind speed at which the turbine reaches its "
                        "rated power output.")},
                "cut_out_wspd": {
                    "type": "number",
                    "units": u.meter/u.second,
                    "about": gettext(
                        "Wind speed above which the turbine stops generating "
                        "power for safety reasons.")},
                "turbine_rated_pwr": {
                    "type": "number",
                    "units": u.kilowatt,
                    "about": gettext("The turbine's rated power output.")},
                "turbine_cost": {
                    "type": "number",
                    "units": u.currency,
                    "about": gettext("The cost of one turbine.")}
            },
            "about": gettext("A table of parameters specific to the type of turbine."),
            "name": gettext("turbine parameters")
        },
        "number_of_turbines": {
            "expression": "value > 0",
            "type": "number",
            "units": u.none,
            "about": gettext("The number of wind turbines per wind farm."),
            "name": gettext("number of turbines")
        },
        "min_depth": {
            "type": "number",
            "units": u.meter,
            "about": gettext("Minimum depth for offshore wind farm installation."),
            "name": gettext("minimum depth")
        },
        "max_depth": {
            "type": "number",
            "units": u.meter,
            "about": gettext("Maximum depth for offshore wind farm installation."),
            "name": gettext("maximum depth")
        },
        "min_distance": {
            "type": "number",
            "units": u.meter,
            "required": "valuation_container",
            "about": gettext(
                "Minimum distance from shore for offshore wind farm "
                "installation. Required if Run Valuation is selected."),
            "name": gettext("minimum distance")
        },
        "max_distance": {
            "type": "number",
            "units": u.meter,
            "required": "valuation_container",
            "about": gettext(
                "Maximum distance from shore for offshore wind farm "
                "installation. Required if Run Valuation is selected."),
            "name": gettext("maximum distance")
        },
        "valuation_container": {
            "type": "boolean",
            "required": False,
            "about": gettext("Run the valuation component of the model."),
            "name": gettext("run valuation")
        },
        "foundation_cost": {
            "type": "number",
            "units": u.currency,
            "required": "valuation_container",
            "about": gettext("The cost of the foundation for one turbine."),
            "name": gettext("foundation cost")
        },
        "discount_rate": {
            "type": "ratio",
            "required": "valuation_container",
            "about": gettext("Annual discount rate to apply to valuation."),
            "name": gettext("discount rate")
        },
        "grid_points_path": {
            "type": "csv",
            "columns": {
                "id": {
                    "type": "integer",
                    "about": gettext("Unique identifier for each point.")},
                "type": {
                    "type": "option_string",
                    "options": {
                        "LAND": {"description": gettext(
                            "This is a land connection point")},
                        "GRID": {"description": gettext(
                            "This is a grid connection point")},
                    },
                    "about": gettext("The type of connection at this point.")
                },
                "lati": {
                    "type": "number",
                    "units": u.degree,
                    "about": gettext("Latitude of the connection point.")
                },
                "long": {
                    "type": "number",
                    "units": u.degree,
                    "about": gettext("Longitude of the connection point.")
                }
            },
            "required": "valuation_container & (not avg_grid_distance)",
            "about": gettext(
                "Table of grid and land connection points to which cables "
                "will connect. Required if Run Valuation is selected and "
                "Average Shore-to-Grid Distance is not provided."),
            "name": gettext("grid connection points")
        },
        "avg_grid_distance": {
            "expression": "value > 0",
            "type": "number",
            "units": u.kilometer,
            "required": "valuation_container & (not grid_points_path)",
            "about": gettext(
                "Average distance to the onshore grid from coastal cable "
                "landing points. Required if Run Valuation is selected and "
                "the Grid Connection Points table is not provided."),
            "name": gettext("average shore-to-grid distance")
        },
        "price_table": {
            "type": "boolean",
            "required": "valuation_container",
            "about": gettext(
                "Use a Wind Energy Price Table instead of calculating annual "
                "prices from the initial Energy Price and Rate of Price Change "
                "inputs."),
            "name": gettext("use price table")
        },
        "wind_schedule": {
            "type": "csv",
            "columns": {
                "year": {
                    "type": "number",
                    "units": u.year_AD,
                    "about": gettext(
                        "Consecutive years for each year in the lifespan of "
                        "the wind farm. These may be the actual years: 2010, "
                        "2011, 2012..., or the number of the years after the "
                        "starting date: 1, 2, 3,...")
                },
                "price": {
                    "type": "number",
                    "units": u.currency/u.kilowatt_hour,
                    "about": gettext("Price of energy for each year.")
                }
            },
            "required": "valuation_container & price_table",
            "about": gettext(
                "Table of yearly prices for wind energy. There must be a row "
                "for each year in the lifespan given in the 'time_period' "
                "column in the Global Wind Energy Parameters table. Required "
                "if Run Valuation and Use Price Table are selected."),
            "name": gettext("wind energy price table")
        },
        "wind_price": {
            "type": "number",
            "units": u.currency/u.kilowatt_hour,
            "required": "valuation_container & (not price_table)",
            "about": gettext(
                "The initial price of wind energy, at the first year in the "
                "wind energy farm lifespan. Required if Run Valuation is "
                "selected and Use Price Table is not selected."),
            "name": gettext("price of energy")
        },
        "rate_change": {
            "type": "ratio",
            "required": "valuation_container & (not price_table)",
            "about": gettext(
                "The annual rate of change in the price of wind energy. "
                "Required if Run Valuation is selected and Use Price Table "
                "is not selected."),
            "name": gettext("rate of price change")
        }
    }
}


# The _SCALE_KEY is used in getting the right wind energy arguments that are
# dependent on the hub height.
_SCALE_KEY = 'LAM'

# The str name for the shape field. So far this is a default from the
# text file given by CK. I guess we could search for the 'K' if needed.
_SHAPE_KEY = 'K'

# Set the raster nodata value and data type to use throughout the model
_TARGET_NODATA = -64329
_TARGET_DATA_TYPE = gdal.GDT_Float32

# The harvested energy is on a per year basis
_NUM_DAYS = 365

# Constant used in getting Scale value at hub height from reference height
# values. See equation 3 in the users guide.
_ALPHA = 0.11

# Field name to be added to the land point shapefile
_LAND_TO_GRID_FIELD = 'L2G'

# The field names for the two output fields, Harvested Wind Energy and Wind
# Density, to be added to the point shapefile
_DENSITY_FIELD_NAME = 'Dens_W/m2'
_HARVESTED_FIELD_NAME = 'Harv_MWhr'

# Resample method for target rasters
_TARGET_RESAMPLE_METHOD = 'near'


def execute(args):
    """Wind Energy.

    This module handles the execution of the wind energy model
    given the following dictionary:

    Args:
        workspace_dir (str): a path to the output workspace folder (required)
        wind_data_path (str): path to a CSV file with the following header
            ['LONG','LATI','LAM', 'K', 'REF']. Each following row is a location
            with at least the Longitude, Latitude, Scale ('LAM'),
            Shape ('K'), and reference height ('REF') at which the data was
            collected (required)
        aoi_vector_path (str): a path to an OGR polygon vector that is
            projected in linear units of meters. The polygon specifies the
            area of interest for the wind data points. If limiting the wind
            farm bins by distance, then the aoi should also cover a portion
            of the land polygon that is of interest (optional for biophysical
            and no distance masking, required for biophysical and distance
            masking, required for valuation)
        bathymetry_path (str): a path to a GDAL raster that has the depth
            values of the area of interest (required)
        land_polygon_vector_path (str): a path to an OGR polygon vector that
            provides a coastline for determining distances from wind farm bins.
            Enabled by AOI and required if wanting to mask by distances or run
            valuation
        global_wind_parameters_path (str): a float for the average distance
            in kilometers from a grid connection point to a land connection
            point (required for valuation if grid connection points are not
            provided)
        results_suffix (str): a str to append to the end of the output files
            (optional)
        turbine_parameters_path (str): a path to a CSV file that holds the
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
        grid_points_path (str): a path to a CSV file that specifies the
            landing and grid point locations (optional)
        avg_grid_distance (float): a float for the average distance in
            kilometers from a grid connection point to a land connection point
            (required for valuation if grid connection points are not provided)
        price_table (boolean): a bool indicating whether to use the wind energy
            price table or not (required)
        wind_schedule (str): a path to a CSV file for the yearly prices of
            wind energy for the lifespan of the farm (required if 'price_table'
            is true)
        wind_price (float): a float for the wind energy price at year 0
            (required if price_table is false)
        rate_change (float): a float as a percent for the annual rate of change
            in the price of wind energy. (required if price_table is false)
        n_workers (int): The number of worker processes to use for processing
            this model.  If omitted, computation will take place in the current
            process. (optional)

    Returns:
        None

    """
    LOGGER.info('Starting the Wind Energy Model')
    invalid_parameters = validate(args)
    if invalid_parameters:
        raise ValueError("Invalid parameters passed: %s" % invalid_parameters)

    workspace = args['workspace_dir']
    inter_dir = os.path.join(workspace, 'intermediate')
    out_dir = os.path.join(workspace, 'output')
    utils.make_directories([inter_dir, out_dir])

    # Append a _ to the suffix if it's not empty and doesn't already have one
    suffix = utils.make_suffix_string(args, 'results_suffix')

    # Initialize a TaskGraph
    taskgraph_working_dir = os.path.join(inter_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_working_dir, n_workers)

    # Resample the bathymetry raster if it does not have square pixel size
    try:
        bathy_pixel_size = pygeoprocessing.get_raster_info(
            args['bathymetry_path'])['pixel_size']
        mean_pixel_size, _ = utils.mean_pixel_size_and_area(bathy_pixel_size)
        target_pixel_size = (mean_pixel_size, -mean_pixel_size)
        LOGGER.debug('Target pixel size: %s' % (target_pixel_size,))
        bathymetry_path = args['bathymetry_path']
        # The task list would be empty for clipping and reprojecting bathymetry
        bathy_dependent_task_list = None

    except ValueError:
        LOGGER.debug(
            '%s has pixels that are not square. Resampling the raster to have '
            'square pixels.' % args['bathymetry_path'])
        bathymetry_path = os.path.join(
            inter_dir, 'bathymetry_resampled%s.tif' % suffix)

        # Get the minimum absolute value from the bathymetry pixel size tuple
        mean_pixel_size = numpy.min(numpy.absolute(bathy_pixel_size))
        # Use it as the target pixel size for resampling and warping rasters
        target_pixel_size = (mean_pixel_size, -mean_pixel_size)
        LOGGER.debug('Target pixel size: %s' % (target_pixel_size,))

        resapmle_bathymetry_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(args['bathymetry_path'], target_pixel_size, bathymetry_path,
                  _TARGET_RESAMPLE_METHOD),
            target_path_list=[bathymetry_path],
            task_name='resample_bathymetry')

        # Build the task list when clipping and reprojecting bathymetry later.
        bathy_dependent_task_list = [resapmle_bathymetry_task]

    number_of_turbines = int(args['number_of_turbines'])

    # Create a list of the biophysical parameters we are looking for from the
    # input csv files
    biophysical_params = [
        'cut_in_wspd', 'cut_out_wspd', 'rated_wspd', 'hub_height',
        'turbine_rated_pwr', 'air_density', 'exponent_power_curve',
        'air_density_coefficient', 'loss_parameter'
    ]

    # Read the biophysical turbine parameters into a dictionary
    bio_turbine_dict = _read_csv_wind_parameters(
        args['turbine_parameters_path'], biophysical_params)

    # Read the biophysical global parameters into a dictionary
    bio_global_params_dict = _read_csv_wind_parameters(
        args['global_wind_parameters_path'], biophysical_params)

    # Combine the turbine and global parameters into one dictionary
    bio_parameters_dict = bio_global_params_dict.copy()
    bio_parameters_dict.update(bio_turbine_dict)

    LOGGER.debug('Biophysical Turbine Parameters: %s', bio_parameters_dict)

    if ('valuation_container' not in args or
            args['valuation_container'] is False):
        LOGGER.info('Valuation Not Selected')
    else:
        LOGGER.info(
            'Valuation Selected. Checking required parameters from CSV files.')

        # Create a list of the valuation parameters we are looking for from the
        # input files
        valuation_turbine_params = ['turbine_cost', 'turbine_rated_pwr']
        # Read the biophysical turbine parameters into a dictionary
        val_turbine_dict = _read_csv_wind_parameters(
            args['turbine_parameters_path'], valuation_turbine_params)

        valuation_global_params = [
            'carbon_coefficient', 'time_period', 'infield_cable_cost',
            'infield_cable_length', 'installation_cost',
            'miscellaneous_capex_cost', 'operation_maintenance_cost',
            'decommission_cost', 'ac_dc_distance_break', 'mw_coef_ac',
            'mw_coef_dc', 'cable_coef_ac', 'cable_coef_dc'
        ]
        # Read the biophysical global parameters into a dictionary
        val_global_param_dict = _read_csv_wind_parameters(
            args['global_wind_parameters_path'], valuation_global_params)

        # Combine the turbine and global parameters into one dictionary
        val_parameters_dict = val_global_param_dict.copy()
        val_parameters_dict.update(val_turbine_dict)

        # If Price Table provided use that for price of energy, validate inputs
        time = int(val_parameters_dict['time_period'])
        if args['price_table']:
            wind_price_df = utils.read_csv_to_dataframe(
                args['wind_schedule'], to_lower=True)

            year_count = len(wind_price_df['year'])
            if year_count != time + 1:
                raise ValueError(
                    "The 'time' argument in the Global Wind Energy Parameters "
                    "file must equal the number of years provided in the price"
                    " table.")

            # Save the price values into a list where the indices of the list
            # indicate the time steps for the lifespan of the wind farm
            wind_price_df.sort_values('year', inplace=True)
            price_list = wind_price_df['price'].tolist()
        else:
            change_rate = float(args["rate_change"])
            wind_price = float(args["wind_price"])
            # Build up a list of price values where the indices of the list
            # are the time steps for the lifespan of the farm and values
            # are adjusted based on the rate of change
            price_list = []
            for time_step in range(time + 1):
                price_list.append(wind_price * (1 + change_rate)**(time_step))

    # Hub Height to use for setting Weibull parameters
    hub_height = int(bio_parameters_dict['hub_height'])

    LOGGER.debug('hub_height : %s', hub_height)

    # Read the wind energy data into a dictionary
    LOGGER.info('Reading in Wind Data into a dictionary')
    wind_data = _read_csv_wind_data(args['wind_data_path'], hub_height)

    # Compute Wind Density and Harvested Wind Energy, adding the values to the
    # points to the dictionary, and pickle the dictionary
    wind_data_pickle_path = os.path.join(
        inter_dir, 'wind_data%s.pickle' % suffix)
    compute_density_harvested_task = task_graph.add_task(
        func=_compute_density_harvested_fields,
        args=(wind_data, bio_parameters_dict, number_of_turbines,
              wind_data_pickle_path),
        target_path_list=[wind_data_pickle_path],
        task_name='compute_density_harvested_fields')

    if 'aoi_vector_path' in args:
        LOGGER.info('AOI Provided')
        aoi_vector_path = args['aoi_vector_path']

        # Get suitable projection parameters for clipping and reprojecting
        # bathymetry layers
        proj_params_pickle_path = os.path.join(
            inter_dir, 'projection_params%s.pickle' % suffix)
        task_graph.add_task(
            func=_get_suitable_projection_params,
            args=(bathymetry_path, aoi_vector_path, proj_params_pickle_path),
            target_path_list=[proj_params_pickle_path],
            task_name='get_suitable_projection_params_from_bathy',
            dependent_task_list=bathy_dependent_task_list)

        # Clip and project the bathymetry shapefile to AOI
        LOGGER.info('Clip and project bathymetry to AOI')
        bathymetry_proj_raster_path = os.path.join(
            inter_dir, 'bathymetry_projected%s.tif' % suffix)

        # Join here because all the following tasks need to unpickle parameters
        # from ``get_suitable_projection_params`` task first
        task_graph.join()
        with open(proj_params_pickle_path, 'rb') as pickle_file:
            target_sr_wkt, target_pixel_size, target_bounding_box = pickle.load(
                pickle_file)
        LOGGER.debug('target_sr_wkt: %s\ntarget_pixel_size: %s\n' +
                     'target_bounding_box: %s\n', target_sr_wkt,
                     (target_pixel_size,), target_bounding_box)

        clip_bathy_to_projection_task = task_graph.add_task(
            func=_clip_to_projection_with_square_pixels,
            args=(bathymetry_path, aoi_vector_path,
                  bathymetry_proj_raster_path, target_sr_wkt,
                  target_pixel_size, target_bounding_box),
            target_path_list=[bathymetry_proj_raster_path],
            task_name='clip_to_projection_with_square_pixels')

        # Creation of depth mask raster is dependent on the final bathymetry
        depth_mask_dependent_task_list = [clip_bathy_to_projection_task]

        # Since an AOI was provided the wind energy points shapefile will need
        # to be clipped and projected. Thus save the construction of the
        # shapefile from dictionary in the intermediate directory. The final
        # projected shapefile will be written to the output directory
        wind_point_vector_path = os.path.join(
            inter_dir, 'wind_energy_points_from_data%s.shp' % suffix)

        # Create point shapefile from wind data
        LOGGER.info('Create point shapefile from wind data')
        # Use the projection from the projected bathymetry as reference to
        # create wind point vector from wind data dictionary
        wind_data_to_vector_task = task_graph.add_task(
            func=_wind_data_to_point_vector,
            args=(wind_data_pickle_path, 'wind_data', wind_point_vector_path),
            kwargs={'ref_projection_wkt': target_sr_wkt},
            target_path_list=[wind_point_vector_path],
            task_name='wind_data_to_vector',
            dependent_task_list=[compute_density_harvested_task])

        # Clip the wind energy point shapefile to AOI
        LOGGER.info('Clip and project wind points to AOI')
        clipped_wind_point_vector_path = os.path.join(
            out_dir, 'wind_energy_points%s.shp' % suffix)
        clip_wind_vector_task = task_graph.add_task(
            func=_clip_vector_by_vector,
            args=(wind_point_vector_path, aoi_vector_path,
                  clipped_wind_point_vector_path, inter_dir),
            target_path_list=[clipped_wind_point_vector_path],
            task_name='clip_wind_point_by_aoi',
            dependent_task_list=[wind_data_to_vector_task])

        # Creating density and harvested rasters depends on the clipped wind
        # vector
        density_harvest_rasters_dependent_task_list = [clip_wind_vector_task]

        # Set the bathymetry and points path to use in the rest of the model.
        # In this case these paths refer to the projected files. This may not
        # be the case if an AOI is not provided
        final_bathy_raster_path = bathymetry_proj_raster_path
        final_wind_point_vector_path = clipped_wind_point_vector_path

        # Try to handle the distance inputs and land datasource if they
        # are present
        try:
            min_distance = float(args['min_distance'])
            max_distance = float(args['max_distance'])
            land_polygon_vector_path = args['land_polygon_vector_path']
        except KeyError:
            LOGGER.info('Distance information not provided')
        else:
            # Clip and project the land polygon shapefile to AOI
            LOGGER.info('Clip and project land polygon to AOI')
            land_poly_proj_vector_path = os.path.join(
                inter_dir, 'projected_clipped_land_poly%s.shp' % suffix)
            clip_reproject_land_poly_task = task_graph.add_task(
                func=_clip_and_reproject_vector,
                args=(land_polygon_vector_path, aoi_vector_path,
                      land_poly_proj_vector_path, inter_dir),
                target_path_list=[land_poly_proj_vector_path],
                task_name='clip_and_reproject_land_poly_to_aoi')

            # If the distance inputs are present create a mask for the output
            # area that restricts where the wind energy farms can be based
            # on distance
            aoi_raster_path = os.path.join(inter_dir,
                                           'aoi_raster%s.tif' % suffix)

            # Make a raster from AOI using the reprojected bathymetry raster's
            # pixel size
            LOGGER.info('Create Raster From AOI')
            create_aoi_raster_task = task_graph.add_task(
                func=_create_aoi_raster,
                args=(aoi_vector_path, aoi_raster_path, target_pixel_size,
                      target_sr_wkt, inter_dir),
                target_path_list=[aoi_raster_path],
                task_name='create_aoi_raster_from_vector')

            # Rasterize land polygon onto AOI and calculate distance transform
            dist_trans_path = os.path.join(
                inter_dir, 'distance_trans%s.tif' % suffix)
            create_distance_raster_task = task_graph.add_task(
                func=_create_distance_raster,
                args=(aoi_raster_path, land_poly_proj_vector_path,
                      dist_trans_path, inter_dir),
                target_path_list=[dist_trans_path],
                task_name='create_distance_raster',
                dependent_task_list=[
                    create_aoi_raster_task, clip_reproject_land_poly_task])

            # Mask the distance raster by the min and max distances
            dist_mask_path = os.path.join(inter_dir,
                                          'distance_mask%s.tif' % suffix)
            mask_by_distance_task = task_graph.add_task(
                func=_mask_by_distance,
                args=(dist_trans_path, min_distance, max_distance,
                      _TARGET_NODATA, dist_mask_path),
                target_path_list=[dist_mask_path],
                task_name='mask_raster_by_distance',
                dependent_task_list=[create_distance_raster_task])

    else:
        LOGGER.info("AOI argument was not selected")

        # Since no AOI was provided the wind energy points shapefile that is
        # created directly from dictionary will be the final output, so set the
        # path to point to the output folder
        wind_point_vector_path = os.path.join(
            out_dir, 'wind_energy_points%s.shp' % suffix)

        # Create point shapefile from wind data dictionary
        LOGGER.info('Create point shapefile from wind data')
        wind_data_to_vector_task = task_graph.add_task(
            func=_wind_data_to_point_vector,
            args=(wind_data_pickle_path, 'wind_data', wind_point_vector_path),
            target_path_list=[wind_point_vector_path],
            task_name='wind_data_to_vector_without_aoi',
            dependent_task_list=[compute_density_harvested_task])

        # Creating density and harvested rasters depends on the wind vector
        density_harvest_rasters_dependent_task_list = [
            wind_data_to_vector_task]

        # Set the bathymetry and points path to use in the rest of the model.
        # In this case these paths refer to the unprojected files. This may not
        # be the case if an AOI is provided
        final_wind_point_vector_path = wind_point_vector_path
        final_bathy_raster_path = bathymetry_path

        # Creation of depth mask is not dependent on creating additional
        # bathymetry mask
        depth_mask_dependent_task_list = None

    # Get the min and max depth values from the arguments and set to a negative
    # value indicating below sea level
    min_depth = abs(float(args['min_depth'])) * -1
    max_depth = abs(float(args['max_depth'])) * -1

    # Create a mask for any values that are out of the range of the depth
    # values
    LOGGER.info('Creating Depth Mask')
    depth_mask_path = os.path.join(inter_dir, 'depth_mask%s.tif' % suffix)

    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(final_bathy_raster_path, 1), (min_depth, 'raw'),
               (max_depth, 'raw')], _depth_op, depth_mask_path,
              _TARGET_DATA_TYPE, _TARGET_NODATA),
        target_path_list=[depth_mask_path],
        task_name='mask_depth_on_bathymetry',
        dependent_task_list=depth_mask_dependent_task_list)

    # Set paths for creating density and harvested rasters
    temp_density_raster_path = os.path.join(
        inter_dir, 'temp_density%s.tif' % suffix)
    temp_harvested_raster_path = os.path.join(
        inter_dir, 'temp_harvested%s.tif' % suffix)

    # Create rasters for density and harvested values
    LOGGER.info('Create Density Raster')
    create_density_raster_task = task_graph.add_task(
        func=pygeoprocessing.create_raster_from_vector_extents,
        args=(final_wind_point_vector_path, temp_density_raster_path,
              target_pixel_size, _TARGET_DATA_TYPE, _TARGET_NODATA),
        target_path_list=[temp_density_raster_path],
        task_name='create_density_raster',
        dependent_task_list=density_harvest_rasters_dependent_task_list)

    LOGGER.info('Create Harvested Raster')
    create_harvested_raster_task = task_graph.add_task(
        func=pygeoprocessing.create_raster_from_vector_extents,
        args=(final_wind_point_vector_path, temp_harvested_raster_path,
              target_pixel_size, _TARGET_DATA_TYPE, _TARGET_NODATA),
        target_path_list=[temp_harvested_raster_path],
        task_name='create_harvested_raster',
        dependent_task_list=density_harvest_rasters_dependent_task_list)

    # Interpolate points onto raster for density values and harvested values:
    LOGGER.info('Interpolate Density Points')
    interpolate_density_task = task_graph.add_task(
        func=pygeoprocessing.interpolate_points,
        args=(final_wind_point_vector_path, _DENSITY_FIELD_NAME,
              (temp_density_raster_path, 1)),
        kwargs={'interpolation_mode': 'linear'},
        task_name='interpolate_density_points',
        dependent_task_list=[create_density_raster_task])

    LOGGER.info('Interpolate Harvested Points')
    interpolate_harvested_task = task_graph.add_task(
        func=pygeoprocessing.interpolate_points,
        args=(final_wind_point_vector_path, _HARVESTED_FIELD_NAME,
              (temp_harvested_raster_path, 1)),
        kwargs={'interpolation_mode': 'linear'},
        task_name='interpolate_harvested_points',
        dependent_task_list=[create_harvested_raster_task])

    # Output paths for final Density and Harvested rasters after they've been
    # masked by depth and distance
    density_masked_path = os.path.join(
        out_dir, 'density_W_per_m2%s.tif' % suffix)
    harvested_masked_path = os.path.join(
        out_dir, 'harvested_energy_MWhr_per_yr%s.tif' % suffix)

    # List of paths to pass to raster_calculator for operations
    density_mask_list = [temp_density_raster_path, depth_mask_path]
    harvested_mask_list = [temp_harvested_raster_path, depth_mask_path]

    # If a distance mask was created then add it to the raster list to pass in
    # for masking out the output datasets
    try:
        density_mask_list.append(dist_mask_path)
        harvested_mask_list.append(dist_mask_path)

        # The align_and_resize_density_and_harvest_task will be dependent on
        # the density, harvested raster interpolation tasks, as well as
        # masking by distance task
        align_and_resize_dependent_task_list = [
            interpolate_density_task, interpolate_harvested_task,
            mask_by_distance_task]
    except NameError:
        # No mask_by_distance_task is added to taskgraph
        align_and_resize_dependent_task_list = [
            interpolate_density_task, interpolate_harvested_task]
        LOGGER.info('NO Distance Mask to add to list')

    # Align and resize the mask rasters
    LOGGER.info('Align and resize the Density rasters')
    aligned_density_mask_list = [
        path.replace('%s.tif' % suffix, '_aligned%s.tif' % suffix)
        for path in density_mask_list
    ]

    aligned_harvested_mask_list = [
        path.replace('%s.tif' % suffix, '_aligned%s.tif' % suffix)
        for path in harvested_mask_list
    ]

    # Merge the density and harvest lists, and remove duplicates
    merged_mask_list = density_mask_list + [
        mask for mask in harvested_mask_list if mask not in density_mask_list
    ]
    merged_aligned_mask_list = aligned_density_mask_list + [
        mask for mask in aligned_harvested_mask_list
        if mask not in aligned_density_mask_list
    ]
    # Align and resize rasters in the density and harvest lists
    align_and_resize_density_and_harvest_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(merged_mask_list, merged_aligned_mask_list,
              [_TARGET_RESAMPLE_METHOD] * len(merged_mask_list),
              target_pixel_size, 'intersection'),
        task_name='align_and_resize_density_and_harvest_list',
        target_path_list=merged_aligned_mask_list,
        dependent_task_list=align_and_resize_dependent_task_list)

    # Mask out any areas where distance or depth has determined that wind farms
    # cannot be located
    LOGGER.info('Mask out depth and [distance] areas from Density raster')
    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(path, 1) for path in aligned_density_mask_list],
              _mask_out_depth_dist, density_masked_path, _TARGET_DATA_TYPE,
              _TARGET_NODATA),
        task_name='mask_density_raster',
        target_path_list=[density_masked_path],
        dependent_task_list=[align_and_resize_density_and_harvest_task])

    LOGGER.info('Mask out depth and [distance] areas from Harvested raster')
    masked_harvested_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(path, 1) for path in aligned_harvested_mask_list],
              _mask_out_depth_dist, harvested_masked_path, _TARGET_DATA_TYPE,
              _TARGET_NODATA),
        task_name='mask_harvested_raster',
        target_path_list=[harvested_masked_path],
        dependent_task_list=[align_and_resize_density_and_harvest_task])

    LOGGER.info('Wind Energy Biophysical Model completed')

    if 'valuation_container' in args and args['valuation_container'] is True:
        LOGGER.info('Starting Wind Energy Valuation Model')

        # path for final distance transform used in valuation calculations
        final_dist_raster_path = os.path.join(
            inter_dir, 'val_distance_trans%s.tif' % suffix)
    else:
        task_graph.close()
        task_graph.join()
        LOGGER.info('Valuation Not Selected. Model completed')
        return

    if 'grid_points_path' in args:
        # Handle Grid Points
        LOGGER.info('Grid Points Provided. Reading in the grid points')

        # Read the grid points csv, and convert it to land and grid dictionary
        grid_land_df = utils.read_csv_to_dataframe(
            args['grid_points_path'], to_lower=True)

        # Make separate dataframes based on 'TYPE'
        grid_df = grid_land_df.loc[(
            grid_land_df['type'].str.upper() == 'GRID')]
        land_df = grid_land_df.loc[(
            grid_land_df['type'].str.upper() == 'LAND')]

        # Convert the dataframes to dictionaries, using 'ID' (the index) as key
        grid_df.set_index('id', inplace=True)
        grid_dict = grid_df.to_dict('index')
        land_df.set_index('id', inplace=True)
        land_dict = land_df.to_dict('index')

        grid_vector_path = os.path.join(
            inter_dir, 'val_grid_points%s.shp' % suffix)

        # Create a point shapefile from the grid point dictionary.
        # This makes it easier for future distance calculations and provides a
        # nice intermediate output for users
        grid_dict_to_vector_task = task_graph.add_task(
            func=_dictionary_to_point_vector,
            args=(grid_dict, 'grid_points', grid_vector_path),
            target_path_list=[grid_vector_path],
            task_name='grid_dictionary_to_vector')

        # In case any of the above points lie outside the AOI, clip the
        # shapefiles and then project them to the AOI as well.
        grid_projected_vector_path = os.path.join(
            inter_dir, 'grid_point_projected_clipped%s.shp' % suffix)
        task_graph.add_task(
            func=_clip_and_reproject_vector,
            args=(grid_vector_path, aoi_vector_path,
                  grid_projected_vector_path, inter_dir),
            target_path_list=[grid_projected_vector_path],
            task_name='clip_and_reproject_grid_vector',
            dependent_task_list=[grid_dict_to_vector_task])

        # It is possible that NO grid points lie within the AOI, so we need to
        # handle both cases
        task_graph.join()  # need to join to get grid feature count
        grid_feature_count = _get_feature_count(grid_projected_vector_path)
        if grid_feature_count > 0:
            LOGGER.debug('There are %s grid point(s) within AOI.' %
                         grid_feature_count)
            # It's possible that no land points were provided, and we need to
            # handle both cases
            if land_dict:
                # A bool used to determine if the final distance raster should
                # be calculated without land points later
                calc_grid_dist_without_land = False

                land_point_vector_path = os.path.join(
                    inter_dir, 'val_land_points%s.shp' % suffix)
                # Create a point shapefile from the land point dictionary.
                # This makes it easier for future distance calculations and
                # provides a nice intermediate output for users
                land_dict_to_vector_task = task_graph.add_task(
                    func=_dictionary_to_point_vector,
                    args=(land_dict, 'land_points', land_point_vector_path),
                    target_path_list=[land_point_vector_path],
                    task_name='land_dictionary_to_vector')

                # In case any of the above points lie outside the AOI, clip the
                # shapefiles and then project them to the AOI as well.
                land_projected_vector_path = os.path.join(
                    inter_dir, 'land_point_projected_clipped%s.shp' % suffix)
                task_graph.add_task(
                    func=_clip_and_reproject_vector,
                    args=(land_point_vector_path, aoi_vector_path,
                          land_projected_vector_path, inter_dir),
                    target_path_list=[land_projected_vector_path],
                    task_name='clip_and_reproject_land_vector',
                    dependent_task_list=[land_dict_to_vector_task])

                # It is possible that NO land point lie within the AOI, so we
                # need to handle both cases
                task_graph.join()  # need to join to get land feature count
                land_feature_count = _get_feature_count(
                    land_projected_vector_path)
                if land_feature_count > 0:
                    LOGGER.debug('There are %d land point(s) within AOI.' %
                                 land_feature_count)

                    # Calculate and add the shortest distances from each land
                    # point to the grid points and add them to the new field
                    LOGGER.info(
                        'Adding land to grid distances ("L2G") field to land '
                        'point shapefile.')

                    # Make a path for the grid vector, so Taskgraph can keep
                    # track of the correct timestamp of file being modified
                    land_to_grid_vector_path = os.path.join(
                        inter_dir,
                        'land_point_to_grid%s.shp' % suffix)

                    land_to_grid_task = task_graph.add_task(
                        func=_calculate_land_to_grid_distance,
                        args=(land_projected_vector_path,
                              grid_projected_vector_path,
                              _LAND_TO_GRID_FIELD, land_to_grid_vector_path),
                        target_path_list=[land_to_grid_vector_path],
                        task_name='calculate_grid_point_to_land_poly')

                    # Calculate distance raster
                    final_dist_task = task_graph.add_task(
                        func=_calculate_distances_land_grid,
                        args=(land_to_grid_vector_path,
                              harvested_masked_path,
                              final_dist_raster_path,
                              inter_dir),
                        target_path_list=[final_dist_raster_path],
                        task_name='calculate_distances_land_grid',
                        dependent_task_list=[land_to_grid_task])
                else:
                    LOGGER.debug(
                        'No land point lies within AOI. Energy transmission '
                        'cable distances are calculated from grid data.')
                    calc_grid_dist_without_land = True

            else:
                LOGGER.info(
                    'No land points provided in the Grid Connection Points '
                    'CSV file. Energy transmission cable distances are '
                    'calculated from grid data.')
                calc_grid_dist_without_land = True

            if calc_grid_dist_without_land:
                # Calculate distance raster without land points provided
                final_dist_task = task_graph.add_task(
                    func=_create_distance_raster,
                    args=(harvested_masked_path,
                          grid_projected_vector_path,
                          final_dist_raster_path,
                          inter_dir),
                    target_path_list=[final_dist_raster_path],
                    task_name='calculate_grid_distance')

        else:
            LOGGER.debug(
                'No grid or land point lies in AOI. Energy transmission '
                'cable distances are not calculated.')

    else:
        LOGGER.info('Grid points not provided')
        LOGGER.debug(
            'No grid points, calculating distances using land polygon')
        # Since the grid points were not provided use the land polygon to get
        # near shore distances
        # The average land cable distance in km converted to meters
        avg_grid_distance = float(args['avg_grid_distance']) * 1000

        land_poly_dist_raster_path = os.path.join(
            inter_dir, 'land_poly_dist%s.tif' % suffix)

        land_poly_dist_raster_task = task_graph.add_task(
            func=_create_distance_raster,
            args=(harvested_masked_path, land_poly_proj_vector_path,
                  land_poly_dist_raster_path, inter_dir),
            target_path_list=[land_poly_dist_raster_path],
            dependent_task_list=[masked_harvested_task],
            task_name='create_land_poly_dist_raster')

        final_dist_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(land_poly_dist_raster_path, 1), (avg_grid_distance, 'raw')],
                _add_avg_dist_op, final_dist_raster_path, _TARGET_DATA_TYPE,
                _TARGET_NODATA),
            target_path_list=[final_dist_raster_path],
            task_name='calculate_final_distance_in_meters',
            dependent_task_list=[land_poly_dist_raster_task])

    # Create output NPV and levelized rasters
    npv_raster_path = os.path.join(out_dir, 'npv%s.tif' % suffix)
    levelized_raster_path = os.path.join(
        out_dir, 'levelized_cost_price_per_kWh%s.tif' % suffix)

    task_graph.add_task(
        func=_calculate_npv_levelized_rasters,
        args=(harvested_masked_path, final_dist_raster_path, npv_raster_path,
              levelized_raster_path, val_parameters_dict, args, price_list),
        target_path_list=[npv_raster_path, levelized_raster_path],
        task_name='calculate_npv_levelized_rasters',
        dependent_task_list=[final_dist_task])

    # Creating output carbon offset raster
    carbon_path = os.path.join(out_dir, 'carbon_emissions_tons%s.tif' % suffix)

    # The amount of CO2 not released into the atmosphere, with the constant
    # conversion factor provided in the users guide by Rob Griffin
    carbon_coef = float(val_parameters_dict['carbon_coefficient'])

    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(harvested_masked_path, 1), (carbon_coef, 'raw')],
              _calculate_carbon_op, carbon_path, _TARGET_DATA_TYPE,
              _TARGET_NODATA),
        target_path_list=[carbon_path],
        dependent_task_list=[masked_harvested_task],
        task_name='calculate_carbon_raster')

    task_graph.close()
    task_graph.join()
    LOGGER.info('Wind Energy Valuation Model Completed')


def _calculate_npv_levelized_rasters(
        base_harvested_raster_path, base_dist_raster_path,
        target_npv_raster_path, target_levelized_raster_path,
        val_parameters_dict, args, price_list):
    """Calculate NPV and levelized rasters from harvested and dist rasters.

    Args:
        base_harvested_raster_path (str): a path to the raster that indicates
            the averaged energy output for a given period

        base_dist_raster_path (str): a path to the raster that indicates the
            distance from wind turbines to the land.

        target_npv_raster_path (str): a path to the target raster to store
            the net present value of a farm centered on each pixel.

        target_levelized_raster_path (str): a path to the target raster to
            store the unit price of energy that would be required to set the
            present value of the farm centered at each pixel equal to zero.

        val_parameters_dict (dict): a dictionary of the turbine and biophysical
            global parameters.

        args (dict): a dictionary that contains information on
            ``foundation_cost``, ``discount_rate``, ``number_of_turbines``.

        price_list (list): a list of wind energy prices for a period of time.


    Returns:
        None

    """
    LOGGER.info('Creating output NPV and levelized rasters.')

    pygeoprocessing.new_raster_from_base(
        base_harvested_raster_path, target_npv_raster_path, _TARGET_DATA_TYPE,
        [_TARGET_NODATA])

    pygeoprocessing.new_raster_from_base(
        base_harvested_raster_path, target_levelized_raster_path,
        _TARGET_DATA_TYPE, [_TARGET_NODATA])

    # Open raster bands for writing
    npv_raster = gdal.OpenEx(
        target_npv_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    npv_band = npv_raster.GetRasterBand(1)
    levelized_raster = gdal.OpenEx(
        target_levelized_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    levelized_band = levelized_raster.GetRasterBand(1)

    # Get constants from val_parameters_dict to make it more readable
    # The length of infield cable in km
    infield_length = float(val_parameters_dict['infield_cable_length'])
    # The cost of infield cable in currency units per km
    infield_cost = float(val_parameters_dict['infield_cable_cost'])
    # The cost of the foundation in currency units
    foundation_cost = float(args['foundation_cost'])
    # The cost of each turbine unit in currency units
    unit_cost = float(val_parameters_dict['turbine_cost'])
    # The installation cost as a decimal
    install_cost = float(val_parameters_dict['installation_cost'])
    # The miscellaneous costs as a decimal factor of capex_arr
    misc_capex_cost = float(val_parameters_dict['miscellaneous_capex_cost'])
    # The operations and maintenance costs as a decimal factor of capex_arr
    op_maint_cost = float(val_parameters_dict['operation_maintenance_cost'])
    # The discount rate as a decimal
    discount_rate = float(args['discount_rate'])
    # The cost to decommission the farm as a decimal factor of capex_arr
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

    # The total mega watt capacity of the wind farm where mega watt is the
    # turbines rated power
    total_mega_watt = mega_watt * int(args['number_of_turbines'])

    # Total infield cable cost
    infield_cable_cost = infield_length * infield_cost * int(
        args['number_of_turbines'])
    LOGGER.debug('infield_cable_cost : %s', infield_cable_cost)

    # Total foundation cost
    total_foundation_cost = (foundation_cost + unit_cost) * int(
        args['number_of_turbines'])
    LOGGER.debug('total_foundation_cost : %s', total_foundation_cost)

    # Nominal Capital Cost (CAP) minus the cost of cable which needs distances
    cap_less_dist = infield_cable_cost + total_foundation_cost
    LOGGER.debug('cap_less_dist : %s', cap_less_dist)

    # Discount rate plus one to get that constant
    disc_const = discount_rate + 1
    LOGGER.debug('discount_rate : %s', disc_const)

    # Discount constant raised to the total time, a constant found in the NPV
    # calculation (1+i)^T
    disc_time = disc_const**int(val_parameters_dict['time_period'])
    LOGGER.debug('disc_time : %s', disc_time)

    for (harvest_block_info, harvest_block_data), (_, dist_block_data) in zip(
            pygeoprocessing.iterblocks((base_harvested_raster_path, 1)),
            pygeoprocessing.iterblocks((base_dist_raster_path, 1))):

        target_arr_shape = harvest_block_data.shape
        target_nodata_mask = utils.array_equals_nodata(
            harvest_block_data, _TARGET_NODATA)

        # Total cable distance converted to kilometers
        cable_dist_arr = dist_block_data / 1000

        # The energy value converted from MWhr/yr (Mega Watt hours as output
        # from CK's biophysical model equations) to kWhr/yr for the
        # valuation model
        energy_val_arr = harvest_block_data * 1000

        # Calculate cable cost. The break at 'circuit_break' indicates the
        # difference in using AC and DC current systems
        circuit_mask = (cable_dist_arr <= circuit_break)
        cable_cost_arr = numpy.full(target_arr_shape, 0, dtype=numpy.float32)

        # Calculate AC cable cost
        cable_cost_arr[circuit_mask] = cable_dist_arr[
            circuit_mask] * cable_coef_ac + (mw_coef_ac * total_mega_watt)
        # Calculate DC cable cost
        cable_cost_arr[~circuit_mask] = cable_dist_arr[
            ~circuit_mask] * cable_coef_dc + (mw_coef_dc * total_mega_watt)
        # Mask out nodata values
        cable_cost_arr[target_nodata_mask] = _TARGET_NODATA

        # Compute the total CAP
        cap_arr = cap_less_dist + cable_cost_arr

        # Nominal total capital costs including installation and
        # miscellaneous costs (capex_arr)
        capex_arr = cap_arr / (1 - install_cost - misc_capex_cost)

        # The ongoing cost of the farm
        ongoing_capex_arr = op_maint_cost * capex_arr

        # The cost to decommission the farm
        decommish_capex_arr = decom * capex_arr / disc_time

        # Initialize the summation of the revenue less the ongoing costs,
        # adjusted for discount rate
        npv_arr = numpy.full(
            target_arr_shape, 0, dtype=numpy.float32)

        # Initialize the numerator summation part of the levelized cost
        levelized_num_arr = numpy.full(
            target_arr_shape, 0, dtype=numpy.float32)

        # Initialize and calculate the denominator summation value for
        # levelized cost of energy at year 0
        levelized_denom_arr = numpy.full(
            target_arr_shape, 0, dtype=numpy.float32)
        levelized_denom_arr = energy_val_arr / disc_const**0

        # Calculate the total NPV and the levelized cost over the lifespan of
        # the wind farm. Starting at year 1, because year 0 yields no revenue
        for year in range(1, len(price_list)):
            # currency units per kilowatt-hour of that year
            currency_per_kwh = float(price_list[year])

            # The revenue for the wind farm. The energy_val_arr is in kWh/yr
            rev_arr = energy_val_arr * currency_per_kwh

            # Calculate the net present value (NPV), the summation of the net
            # revenue from power generation, adjusted for discount rate
            npv_arr = (
                npv_arr + (rev_arr - ongoing_capex_arr) / disc_const**year)

            # Calculate the cumulative numerator summation value
            levelized_num_arr = levelized_num_arr + (
                (ongoing_capex_arr / disc_const**year))

            # Calculate the cumulative denominator summation value
            levelized_denom_arr = levelized_denom_arr + (
                energy_val_arr / disc_const**year)

        # Calculate the final NPV by subtracting other costs from the NPV
        npv_arr[target_nodata_mask] = _TARGET_NODATA
        npv_arr[~target_nodata_mask] = (
            npv_arr[~target_nodata_mask] -
            decommish_capex_arr[~target_nodata_mask] -
            capex_arr[~target_nodata_mask])

        # Calculate the levelized cost of energy
        levelized_arr = (
            (levelized_num_arr + decommish_capex_arr + capex_arr) /
            levelized_denom_arr)
        levelized_arr[target_nodata_mask] = _TARGET_NODATA

        npv_band.WriteArray(npv_arr,
                            xoff=harvest_block_info['xoff'],
                            yoff=harvest_block_info['yoff'])
        npv_band.FlushCache()

        levelized_band.WriteArray(levelized_arr,
                                  xoff=harvest_block_info['xoff'],
                                  yoff=harvest_block_info['yoff'])
        levelized_band.FlushCache()

    npv_band = None
    npv_raster.FlushCache()
    npv_raster = None

    levelized_band = None
    levelized_raster.FlushCache()
    levelized_raster = None


def _get_feature_count(base_vector_path):
    """Get feature count from vector and return it.

    Args:
        base_vector_path (str): a path to the vector to get feature
            count from.

    Returns:
        feature_count (float): the feature count in the base vector.

    """
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    feature_count = layer.GetFeatureCount()
    layer = None
    vector = None

    return feature_count


def _get_file_ext_and_driver_name(base_vector_path):
    """Get file extension and GDAL driver name from the vector path.

    Args:
        base_path (str): a path to the vector file to get extension from.

    Returns:
        file_ext (str): the file extension of the base file path.
        driver_name (str): the GDAL driver name used to create data.

    """
    file_ext = os.path.splitext(base_vector_path)[1]

    # A dictionary of file extensions/GDAL vector driver names pairs
    vector_formats = {
        '.shp': 'ESRI Shapefile',
        '.gpkg': 'GPKG',
        '.geojson': 'GeoJSON',
        '.gmt': 'GMT'
    }

    try:
        driver_name = vector_formats[file_ext]
    except KeyError:
        raise KeyError(
            'Unknown file extension for vector file %s' % base_vector_path)

    return file_ext, driver_name


def _depth_op(bath, min_depth, max_depth):
    """Determine if a value falls within the range.

    The function takes a value and uses a range to determine if that falls
    within the range.

    Args:
        bath (int): a value of either positive or negative
        min_depth (float): a value specifying the lower limit of the
            range. This value is set above
        max_depth (float): a value specifying the upper limit of the
            range. This value is set above
        _TARGET_NODATA (int or float): a nodata value set above

    Returns:
        out_array (numpy.array): an array where values are _TARGET_NODATA
            if 'bath' does not fall within the range, or 'bath' if it does.

    """
    out_array = numpy.full(
        bath.shape, _TARGET_NODATA, dtype=numpy.float32)
    valid_pixels_mask = ((bath >= max_depth) & (bath <= min_depth) &
                         ~utils.array_equals_nodata(bath, _TARGET_NODATA))
    out_array[
        valid_pixels_mask] = bath[valid_pixels_mask]
    return out_array


def _add_avg_dist_op(tmp_dist, avg_grid_distance):
    """Add in avg_grid_distance.

    Args:
        tmp_dist (numpy.array): an array of distances in meters
        avg_grid_distance (float): the average land cable distance in meters

    Returns:
        out_array (numpy.array): distance values in meters with average
            grid to land distance factored in

    """
    out_array = numpy.full(
        tmp_dist.shape, _TARGET_NODATA, dtype=numpy.float32)
    valid_pixels_mask = ~utils.array_equals_nodata(tmp_dist, _TARGET_NODATA)
    out_array[valid_pixels_mask] = (
        tmp_dist[valid_pixels_mask] + avg_grid_distance)
    return out_array


def _create_aoi_raster(base_aoi_vector_path, target_aoi_raster_path,
                       target_pixel_size, target_sr_wkt, work_dir):
    """Create an AOI raster from a vector w/ target pixel size and projection.

    Args:
        base_aoi_vector_path (str): a path to the base AOI vector to create
            AOI raster from.
        target_aoi_raster_path (str): a path to the target AOI raster.
        target_pixel_size (tuple): a tuple of x, y pixel sizes for the target
            AOI raster.
        target_sr_wkt (str): a projection string used as the target projection
            for the AOI raster.
        work_dir (str): path to create a temp folder for saving temp files.

    Returns:
        None

    """
    base_sr_wkt = pygeoprocessing.get_vector_info(
        base_aoi_vector_path)['projection_wkt']
    if base_sr_wkt != target_sr_wkt:
        # Reproject clip vector to the spatial reference of the base vector.
        # Note: reproject_vector can be expensive if vector has many features.
        temp_dir = tempfile.mkdtemp(dir=work_dir, prefix='clip-')
        file_ext, driver_name = _get_file_ext_and_driver_name(
            base_aoi_vector_path)
        reprojected_aoi_vector_path = os.path.join(
            temp_dir, 'reprojected_aoi' + file_ext)
        pygeoprocessing.reproject_vector(
            base_aoi_vector_path, target_sr_wkt,
            reprojected_aoi_vector_path, driver_name=driver_name)
        pygeoprocessing.create_raster_from_vector_extents(
            reprojected_aoi_vector_path, target_aoi_raster_path,
            target_pixel_size, gdal.GDT_Byte, _TARGET_NODATA)
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        pygeoprocessing.create_raster_from_vector_extents(
            base_aoi_vector_path, target_aoi_raster_path, target_pixel_size,
            gdal.GDT_Byte, _TARGET_NODATA)


def _mask_out_depth_dist(*rasters):
    """Return the value of an item in the list based on some condition.

    Return the value of an item in the list if and only if all other values
    are not a nodata value.

    Args:
        *rasters (list): a list of values as follows:
            rasters[0] - the density value (required)
            rasters[1] - the depth mask value (required)
            rasters[2] - the distance mask value (optional)

    Returns:
        out_array (numpy.array): an array of either _TARGET_NODATA or density
            values from rasters[0]

    """
    out_array = numpy.full(rasters[0].shape, _TARGET_NODATA, dtype=numpy.float32)
    nodata_mask = numpy.full(rasters[0].shape, False, dtype=bool)
    for array in rasters:
        nodata_mask = nodata_mask | utils.array_equals_nodata(
                array, _TARGET_NODATA)
    out_array[~nodata_mask] = rasters[0][~nodata_mask]
    return out_array


def _calculate_carbon_op(harvested_arr, carbon_coef):
    """Calculate the carbon offset from harvested array.

    Args:
        harvested_arr (numpy.array): an array of harvested energy values
        carbon_coef (float): the amount of CO2 not released into the
                atmosphere

    Returns:
        out_array (numpy.array): an array of carbon offset values

    """
    out_array = numpy.full(
        harvested_arr.shape, _TARGET_NODATA, dtype=numpy.float32)
    valid_pixels_mask = ~utils.array_equals_nodata(
        harvested_arr, _TARGET_NODATA)

    # The energy value converted from MWhr/yr (Mega Watt hours as output
    # from CK's biophysical model equations) to kWhr for the
    # valuation model
    out_array[valid_pixels_mask] = (
        harvested_arr[valid_pixels_mask] * carbon_coef * 1000)
    return out_array


def _calculate_land_to_grid_distance(
        base_land_vector_path, base_grid_vector_path, dist_field_name,
        target_land_vector_path):
    """Calculate the distances from points to the nearest polygon.

    Distances are calculated from points in a point geometry shapefile to the
    nearest polygon from a polygon shapefile. Both shapefiles must be
    projected in meters

    Args:
        base_land_vector_path (str): a path to an OGR point geometry shapefile
            projected in meters
        base_grid_vector_path (str): a path to an OGR polygon shapefile
            projected in meters
        dist_field_name (str): the name of the new distance field to be added
            to the attribute table of base_point_vector
        copied_point_vector_path (str): if a path is provided, make a copy of
            the base point vector on this path after computing the distance
            field. (optional)

    Returns:
        None.

    """
    LOGGER.info('Starting _calculate_land_to_grid_distance.')

    # Copy the point vector
    _, driver_name = _get_file_ext_and_driver_name(
        target_land_vector_path)
    base_land_vector = ogr.Open(base_land_vector_path, gdal.OF_VECTOR)
    driver = ogr.GetDriverByName(driver_name)
    driver.CopyDataSource(base_land_vector, target_land_vector_path)
    base_land_vector = None

    target_land_vector = gdal.OpenEx(
        target_land_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    base_grid_vector = gdal.OpenEx(
        base_grid_vector_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)

    base_grid_layer = base_grid_vector.GetLayer()
    # List to store the grid point geometries as shapely objects
    grid_point_list = []

    LOGGER.info('Loading the polygons into Shapely')
    for grid_point_feat in base_grid_layer:
        # Get the geometry of the grid point in WKT format
        grid_point_wkt = grid_point_feat.GetGeometryRef().ExportToWkt()
        # Load the geometry into shapely making it a shapely object
        shapely_grid_point = shapely.wkt.loads(grid_point_wkt)
        # Add the shapely point geometry to a list
        grid_point_list.append(shapely_grid_point)

    # Take the union over the list of points to get one point collection object
    LOGGER.info('Get the collection of polygon geometries by taking the union')
    grid_point_collection = shapely.ops.unary_union(grid_point_list)

    target_land_layer = target_land_vector.GetLayer()
    # Create a new distance field based on the name given
    dist_field_defn = ogr.FieldDefn(dist_field_name, ogr.OFTReal)
    target_land_layer.CreateField(dist_field_defn)

    LOGGER.info('Loading the points into shapely')
    for land_point_feat in target_land_layer:
        # Get the geometry of the point in WKT format
        land_point_wkt = land_point_feat.GetGeometryRef().ExportToWkt()
        # Load the geometry into shapely making it a shapely object
        shapely_land_point = shapely.wkt.loads(land_point_wkt)
        # Get the distance in meters and convert to km
        land_to_grid_dist = shapely_land_point.distance(
            grid_point_collection) / 1000
        # Add the distance value to the new field and set to the feature
        land_point_feat.SetField(dist_field_name, land_to_grid_dist)
        target_land_layer.SetFeature(land_point_feat)

    target_land_layer = None
    target_land_vector = None
    base_grid_layer = None
    base_grid_vector = None

    LOGGER.info('Finished _calculate_land_to_grid_distance.')


def _read_csv_wind_parameters(csv_path, parameter_list):
    """Construct a dictionary from a csv file given a list of keys.

    The list of keys corresponds to the parameters names in 'csv_path' which
    are represented in the first column of the file.

    Args:
        csv_path (str): a path to a CSV file where every row is a parameter
            with the parameter name in the first column followed by the value
            in the second column
        parameter_list (list) : a List of strs that represent the parameter
            names to be found in 'csv_path'. These strs will be the keys in
            the returned dictionary

    Returns: a Dictionary where the 'parameter_list' strs are the
            keys that have values pulled from 'csv_path'

    """
    # use the parameters in the first column as indices for the dataframe
    # this doesn't benefit from `utils.read_csv_to_dataframe` because there
    # is no header to strip whitespace
    # use sep=None, engine='python' to infer what the separator is
    wind_param_df = pandas.read_csv(
        csv_path, header=None, index_col=0, sep=None, engine='python')
    # only get the required parameters and leave out the rest
    wind_param_df = wind_param_df[wind_param_df.index.isin(parameter_list)]
    wind_dict = wind_param_df.to_dict()[1]

    return wind_dict


def _mask_by_distance(base_raster_path, min_dist, max_dist, out_nodata,
                      target_raster_path):
    """Create a raster whose pixel values are bound by min and max distances.

    Args:
        base_raster_path (str): path to a raster with euclidean distance values in meters
        min_dist (int): the minimum distance allowed in meters.
        max_dist (int): the maximum distance allowed in meters.
        target_raster_path (str): path output to the raster masked by distance
            values.
        out_nodata (float): the nodata value of the raster.

    Returns:
        None.

    """
    raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    raster_nodata = raster_info['nodata'][0]

    def _dist_mask_op(dist_arr):
        """Mask distance values by min/max values."""
        out_array = numpy.full(dist_arr.shape, out_nodata, dtype=numpy.float32)
        valid_pixels_mask = (
            ~utils.array_equals_nodata(dist_arr, raster_nodata) &
            (dist_arr >= min_dist) & (dist_arr <= max_dist))
        out_array[
            valid_pixels_mask] = dist_arr[valid_pixels_mask]
        return out_array

    pygeoprocessing.raster_calculator([(base_raster_path, 1)], _dist_mask_op,
                                      target_raster_path, _TARGET_DATA_TYPE,
                                      out_nodata)


def _create_distance_raster(base_raster_path, base_vector_path,
                            target_dist_raster_path, work_dir):
    """Create and rasterize vector onto a raster, and calculate dist transform.

    Create a raster where the pixel values represent the euclidean distance to
    the vector. The distance inherits units from ``base_raster_path`` pixel
    dimensions.

    Args:
        base_raster_path (str): path to raster to create a new raster from.
        base_vector_path (str): path to vector to be rasterized.
        target_dist_raster_path (str): path to raster with distance transform.
        work_dir (str): path to create a temp folder for saving files.

    Returns:
        None

    """
    LOGGER.info("Starting _create_distance_raster")
    temp_dir = tempfile.mkdtemp(dir=work_dir, prefix='dist-raster-')

    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    pixel_xy_scale = tuple([abs(p) for p in base_raster_info['pixel_size']])

    rasterized_raster_path = os.path.join(temp_dir, 'rasterized_raster.tif')

    # Create a new raster based on the given base raster and fill with 0's
    # to set up for distance transform
    pygeoprocessing.new_raster_from_base(
        base_raster_path,
        rasterized_raster_path,
        gdal.GDT_Byte,
        band_nodata_list=[255],
        fill_value_list=[0])

    # Burn vector onto the raster to set up for distance transform
    pygeoprocessing.rasterize(
        base_vector_path,
        rasterized_raster_path,
        burn_values=[1],
        option_list=["ALL_TOUCHED=TRUE"])

    # Calculate euclidean distance transform
    pygeoprocessing.distance_transform_edt(
        (rasterized_raster_path, 1), target_dist_raster_path,
        sampling_distance=pixel_xy_scale)

    # Set the nodata value of output raster to _TARGET_NODATA
    target_dist_raster = gdal.OpenEx(
        target_dist_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    for band in range(1, target_dist_raster.RasterCount+1):
        target_band = target_dist_raster.GetRasterBand(band)
        target_band.SetNoDataValue(_TARGET_NODATA)
        target_band.FlushCache()
        target_band = None
    target_dist_raster.FlushCache()
    target_dist_raster = None

    shutil.rmtree(temp_dir, ignore_errors=True)
    LOGGER.info("Finished _create_distance_raster")


def _read_csv_wind_data(wind_data_path, hub_height):
    """Unpack the csv wind data into a dictionary.

    Args:
        wind_data_path (str): a path for the csv wind data file with header
            of: "LONG","LATI","LAM","K","REF"
        hub_height (int): the hub height to use for calculating Weibull
            parameters and wind energy values

    Returns:
        A dictionary where the keys are lat/long tuples which point
            to dictionaries that hold wind data at that location.

    """
    wind_point_df = utils.read_csv_to_dataframe(wind_data_path, to_lower=False)

    # Calculate scale value at new hub height given reference values.
    # See equation 3 in users guide
    wind_point_df.rename(columns={'LAM': 'REF_LAM'}, inplace=True)
    wind_point_df['LAM'] = wind_point_df.apply(
        lambda row: row.REF_LAM * (hub_height / row.REF)**_ALPHA, axis=1)
    wind_point_df.drop(['REF'], axis=1)  # REF is not needed after calculation
    wind_dict = wind_point_df.to_dict('index')  # so keys will be 0, 1, 2, ...

    return wind_dict


def _compute_density_harvested_fields(
        wind_dict, bio_parameters_dict, number_of_turbines,
        target_pickle_path):
    """Compute the density and harvested energy based on scale and shape keys.

    Args:
        wind_dict (dict): a dictionary whose values are a dictionary with
            keys ``LAM``, ``LATI``, ``K``, ``LONG``, ``REF_LAM``, and ``REF``,
            and numbers indicating their corresponding values.

        bio_parameters_dict (dict): a dictionary where the 'parameter_list'
            strings are the keys that have values pulled from bio-parameters
            CSV.

        number_of_turbines (int): an integer value for the number of machines
            for the wind farm.

        target_pickle_path (str): a path to the pickle file that has
            wind_dict_copy, a modified dictionary with new fields computed
            from the existing fields and bio-parameters.

    Returns:
        None

    """
    wind_dict_copy = wind_dict.copy()

    # The rated power is expressed in units of MW but the harvested energy
    # equation calls for it in terms of Wh. Thus we multiply by a million to
    # get to Wh.
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

    # Hub Height to use for setting Weibull parameters
    hub_height = int(bio_parameters_dict['hub_height'])

    # Compute the mean air density, given by CKs formulas
    mean_air_density = air_density_standard - air_density_coef * hub_height

    # Fractional coefficient that lives outside the intregation for computing
    # the harvested wind energy
    fract_coef = rated_power * (mean_air_density / air_density_standard)

    # The coefficient that is multiplied by the integration portion of the
    # harvested wind energy equation
    scalar = _NUM_DAYS * 24 * fract_coef

    # Weibull probability function to integrate over
    def _calc_weibull_probability(v_speed, k_shape, l_scale):
        """Calculate the Weibull probability function of variable v_speed.

        Args:
            v_speed (int or float): a number representing wind speed
            k_shape (float): the shape parameter
            l_scale (float): the scale parameter of the distribution

        Returns:
            a float

        """
        return ((k_shape / l_scale) * (v_speed / l_scale)**(k_shape - 1) *
                (math.exp(-1 * (v_speed / l_scale)**k_shape)))

    # Density wind energy function to integrate over
    def _calc_density_wind_energy(v_speed, k_shape, l_scale):
        """Calculate the probability density function of a Weibull variable.

        Args:
            v_speed (int or float): a number representing wind speed
            k_shape (float): the shape parameter
            l_scale (float): the scale parameter of the distribution

        Returns:
            a float

        """
        return ((k_shape / l_scale) * (v_speed / l_scale)**(k_shape - 1) *
                (math.exp(-1 * (v_speed / l_scale)**k_shape))) * v_speed**3

    # Harvested wind energy function to integrate over
    def _calc_harvested_wind_energy(v_speed, k_shape, l_scale):
        """Calculate the harvested wind energy.

        Args:
            v_speed (int or float): a number representing wind speed
            k_shape (float): the shape parameter
            l_scale (float): the scale parameter of the distribution

        Returns:
            a float

        """
        fract = ((v_speed**exp_pwr_curve - v_in**exp_pwr_curve) /
                 (v_rate**exp_pwr_curve - v_in**exp_pwr_curve))

        return fract * _calc_weibull_probability(v_speed, k_shape, l_scale)

    for key, value_fields in wind_dict.items():
        # Get the indexes for the scale and shape parameters
        scale_value = value_fields[_SCALE_KEY]
        shape_value = value_fields[_SHAPE_KEY]

        # Integrate over the probability density function. 0 and 50 are
        # hard coded values set in CKs documentation
        density_results = integrate.quad(_calc_density_wind_energy, 0, 50,
                                         (shape_value, scale_value))

        # Compute the mean air density, given by CKs formulas
        mean_air_density = air_density_standard - air_density_coef * hub_height

        # Compute the final wind power density value
        density_results = 0.5 * mean_air_density * density_results[0]

        # Integrate over the harvested wind energy function
        harv_results = integrate.quad(_calc_harvested_wind_energy, v_in,
                                      v_rate, (shape_value, scale_value))

        # Integrate over the Weibull probability function
        weibull_results = integrate.quad(_calc_weibull_probability, v_rate,
                                         v_out, (shape_value, scale_value))

        # Compute the final harvested wind energy value
        harvested_wind_energy = (
            scalar * (harv_results[0] + weibull_results[0]))

        # Convert harvested energy from Whr/yr to MWhr/yr by dividing by
        # 1,000,000
        harvested_wind_energy = harvested_wind_energy / 1000000

        # Now factor in the percent losses due to turbine
        # downtime (mechanical failure, storm damage, etc.)
        # and due to electrical resistance in the cables
        harvested_wind_energy = (1 - losses) * harvested_wind_energy

        # Finally, multiply the harvested wind energy by the number of
        # turbines to get the amount of energy generated for the entire farm
        harvested_wind_energy = harvested_wind_energy * number_of_turbines

        # Append calculated results to the dictionary
        wind_dict_copy[key][_DENSITY_FIELD_NAME] = density_results
        wind_dict_copy[key][_HARVESTED_FIELD_NAME] = harvested_wind_energy

    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(wind_dict_copy, pickle_file)


def _dictionary_to_point_vector(
        base_dict_data, layer_name, target_vector_path):
    """Create a point shapefile from a dictionary.

    The point shapefile created is not projected and uses latitude and
        longitude for its geometry.

    Args:
        base_dict_data (dict): a python dictionary with keys being unique id's
            that point to sub-dictionaries that have key-value pairs. These
            inner key-value pairs will represent the field-value pair for the
            point features. At least two fields are required in the sub-
            dictionaries. All the keys in the sub dictionary should have the
            same name and order. All the values in the sub dictionary should
            have the same type 'LATI' and 'LONG'. These fields determine the
            geometry of the point.
            0 : {'TYPE':GRID, 'LATI':41, 'LONG':-73, ...},
            1 : {'TYPE':GRID, 'LATI':42, 'LONG':-72, ...},
            2 : {'TYPE':GRID, 'LATI':43, 'LONG':-72, ...},
        layer_name (str): a python str for the name of the layer
        target_vector_path (str): a path to the output path of the point
            vector.

    Returns:
        None

    """
    # If the target_vector_path exists delete it
    _, driver_name = _get_file_ext_and_driver_name(target_vector_path)
    output_driver = ogr.GetDriverByName(driver_name)
    if os.path.exists(target_vector_path):
        output_driver.DeleteDataSource(target_vector_path)

    target_vector = output_driver.CreateDataSource(target_vector_path)

    # Set the spatial reference to WGS84 (lat/long)
    source_sr = osr.SpatialReference()
    source_sr.SetWellKnownGeogCS("WGS84")

    output_layer = target_vector.CreateLayer(layer_name, source_sr,
                                             ogr.wkbPoint)

    # Outer unique keys
    outer_keys = list(base_dict_data.keys())

    # Construct a list of fields to add from the keys of the inner dictionary
    field_list = list(base_dict_data[outer_keys[0]])

    # Create a dictionary to store what variable types the fields are
    type_dict = {}
    for field in field_list:
        field_type = None
        # Get a value from the field
        val = base_dict_data[outer_keys[0]][field]
        # Check to see if the value is a str of characters or a number. This
        # will determine the type of field created in the shapefile
        if isinstance(val, str):
            type_dict[field] = 'str'
            field_type = ogr.OFTString
        else:
            type_dict[field] = 'number'
            field_type = ogr.OFTReal
        output_field = ogr.FieldDefn(field, field_type)
        output_layer.CreateField(output_field)

    # For each inner dictionary (for each point) create a point and set its
    # fields
    for point_dict in base_dict_data.values():
        latitude = float(point_dict['lati'])
        longitude = float(point_dict['long'])

        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint_2D(longitude, latitude)

        output_feature = ogr.Feature(output_layer.GetLayerDefn())

        for field_name in point_dict:
            field_index = output_feature.GetFieldIndex(field_name)
            output_feature.SetField(field_index, point_dict[field_name])

        output_feature.SetGeometryDirectly(geom)
        output_layer.CreateFeature(output_feature)
        output_feature = None

    output_layer.SyncToDisk()


def _get_suitable_projection_params(
        base_raster_path, aoi_vector_path, target_pickle_path):
    """Choose projection, pixel size and bounding box for clipping a raster.

    If base raster is not already projected, choose a suitable UTM zone.
    The target_pickle_path contains a tuple of three elements:
        target_sr_wkt (str): a projection string used as the target projection
            for warping the base raster later on.
        target_pixel_size (tuple): a tuple of equal x, y pixel sizes in minimum
            absolute value.
        target_bounding_box (list): a list of the form [xmin, ymin, xmax, ymax]
            that describes the largest fitting bounding box around the original
            warped bounding box in ````new_epsg```` coordinate system.

    Args:
        base_raster_path (str): path to base raster that might not be projected
        aoi_vector_path (str): path to base AOI vector that'll be used to
            clip the raster.
        target_pickle_path (str): a path to the pickle file for storing
            target_sr_wkt, target_pixel_size, and target_bounding_box.

    Returns:
        None.

    """
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    aoi_vector_info = pygeoprocessing.get_vector_info(aoi_vector_path)

    base_raster_srs = osr.SpatialReference()
    base_raster_srs.ImportFromWkt(base_raster_info['projection_wkt'])

    if not base_raster_srs.IsProjected():
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)
        aoi_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
            aoi_vector_info['bounding_box'], aoi_vector_info['projection_wkt'],
            wgs84_sr.ExportToWkt())

        base_raster_bounding_box = pygeoprocessing.transform_bounding_box(
            base_raster_info['bounding_box'],
            base_raster_info['projection_wkt'], wgs84_sr.ExportToWkt())

        target_bounding_box_wgs84 = pygeoprocessing.merge_bounding_box_list(
            [aoi_wgs84_bounding_box, base_raster_bounding_box], 'intersection')

        # Get the suitable UTM code
        centroid_x = (
            target_bounding_box_wgs84[2] + target_bounding_box_wgs84[0]) / 2
        centroid_y = (
            target_bounding_box_wgs84[3] + target_bounding_box_wgs84[1]) / 2

        # Get target pixel size in square meters used for resizing the base
        # raster later on
        target_pixel_size = _convert_degree_pixel_size_to_square_meters(
            base_raster_info['pixel_size'], centroid_y)

        utm_code = (math.floor((centroid_x + 180) / 6) % 60) + 1
        lat_code = 6 if centroid_y > 0 else 7
        epsg_code = int('32%d%02d' % (lat_code, utm_code))
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(epsg_code)

        # Transform the merged unprojected bounding box of base raster and clip
        # vector from WGS84 to the target UTM projection
        target_bounding_box = pygeoprocessing.transform_bounding_box(
            target_bounding_box_wgs84, wgs84_sr.ExportToWkt(),
            target_srs.ExportToWkt())

        target_sr_wkt = target_srs.ExportToWkt()
    else:
        # If the base raster is already projected, transform the bounding
        # box from base raster and aoi vector bounding boxes
        target_sr_wkt = base_raster_info['projection_wkt']

        aoi_bounding_box = pygeoprocessing.transform_bounding_box(
            aoi_vector_info['bounding_box'], aoi_vector_info['projection_wkt'],
            target_sr_wkt)

        target_bounding_box = pygeoprocessing.merge_bounding_box_list(
            [aoi_bounding_box, base_raster_info['bounding_box']],
            'intersection')

        # Get the minimum square pixel size
        min_pixel_size = numpy.min(numpy.absolute(base_raster_info['pixel_size']))
        target_pixel_size = (min_pixel_size, -min_pixel_size)

    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(
            (target_sr_wkt, target_pixel_size, target_bounding_box),
            pickle_file)


def _clip_to_projection_with_square_pixels(
        base_raster_path, clip_vector_path, target_raster_path,
        target_projection_wkt, target_pixel_size, target_bounding_box):
    """Clip raster with vector into target projected coordinate system.

    If pixel size of target raster is not square, the minimum absolute value
    will be used for target_pixel_size.

    Args:
        base_raster_path (str): path to base raster.
        clip_vector_path (str): path to base clip vector.
        target_projection_wkt (str): a projection string used as the target
            projection for warping the base raster.
        target_pixel_size (tuple): a tuple of equal x, y pixel sizes in minimum
            absolute value.
        target_bounding_box (list): a list of the form [xmin, ymin, xmax, ymax]
            that describes the largest fitting bounding box around the original
            warped bounding box in ````new_epsg```` coordinate system.

    Returns:
        None.

    """
    pygeoprocessing.warp_raster(
        base_raster_path,
        target_pixel_size,
        target_raster_path,
        _TARGET_RESAMPLE_METHOD,
        target_bb=target_bounding_box,
        target_projection_wkt=target_projection_wkt,
        vector_mask_options={'mask_vector_path': clip_vector_path})


def _convert_degree_pixel_size_to_square_meters(pixel_size, center_lat):
    """Calculate meter size of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397. If the pixel
    is not square, this function will use the minimum absolute value from the
    pixel dimension in the output meter_pixel_size_tuple.

    Args:
        pixel_size (tuple): [xsize, ysize] in degrees (float).
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the ``pixel-size`` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        ``meter_pixel_size_tuple`` with minimum absolute value in the pixel
            sizes.

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
    x_meter_size = longlen * pixel_size[0]
    y_meter_size = latlen * pixel_size[1]
    meter_pixel_size_tuple = (x_meter_size, y_meter_size)
    if not numpy.isclose(x_meter_size, y_meter_size):
        min_meter_size = numpy.min(numpy.absolute(meter_pixel_size_tuple))
        meter_pixel_size_tuple = (min_meter_size, -min_meter_size)

    return meter_pixel_size_tuple


def _wind_data_to_point_vector(wind_data_pickle_path,
                               layer_name,
                               target_vector_path,
                               ref_projection_wkt=None):
    """Create a point shapefile given a dictionary of the wind data fields.

    Args:
        wind_data_pickle_path (str): a path to the pickle file that has
            wind_dict_copy, where the keys are tuples of the lat/long
            coordinates:
                {
                1 : {'LATI':97, 'LONG':43, 'LAM':6.3, 'K':2.7, 'REF':10, ...},
                2 : {'LATI':55, 'LONG':51, 'LAM':6.2, 'K':2.4, 'REF':10, ...},
                3 : {'LATI':73, 'LONG':47, 'LAM':6.5, 'K':2.3, 'REF':10, ...}
                }
        layer_name (str): the name of the layer.
        target_vector_path (str): path to the output destination of the
            shapefile.
        ref_projection_wkt (str): reference projection of in Well Known Text.

    Returns:
        None

    """
    LOGGER.info('Entering _wind_data_to_point_vector')

    # Unpickle the wind data dictionary
    with open(wind_data_pickle_path, 'rb') as pickle_file:
        wind_data = pickle.load(pickle_file)

    # Get driver based on file extension
    _, driver_name = _get_file_ext_and_driver_name(target_vector_path)

    # If the target_vector_path exists delete it
    if os.path.isfile(target_vector_path):
        driver = ogr.GetDriverByName(driver_name)
        driver.DeleteDataSource(target_vector_path)

    target_driver = ogr.GetDriverByName(driver_name)
    target_vector = target_driver.CreateDataSource(target_vector_path)
    target_sr = osr.SpatialReference()
    target_sr.SetWellKnownGeogCS("WGS84")

    need_geotranform = False
    if ref_projection_wkt:
        ref_sr = osr.SpatialReference(wkt=ref_projection_wkt)
        if ref_sr.IsProjected:
            # Get coordinate transformation between two projections
            coord_trans = utils.create_coordinate_transformer(
                target_sr, ref_sr)
            need_geotranform = True
    else:
        need_geotranform = False

    if need_geotranform:
        target_layer = target_vector.CreateLayer(
            layer_name, ref_sr, ogr.wkbPoint)
    else:
        target_layer = target_vector.CreateLayer(
            layer_name, target_sr, ogr.wkbPoint)

    # Construct a list of fields to add from the keys of the inner dictionary
    field_list = list(wind_data[list(wind_data.keys())[0]])

    # For the two fields that we computed and added to the dictionary, move
    # them to the last
    for field in [_DENSITY_FIELD_NAME, _HARVESTED_FIELD_NAME]:
        if field in field_list:
            field_list.remove(field)
            field_list.append(field)

    LOGGER.debug('field_list : %s', field_list)

    LOGGER.info('Creating fields for the target vector')
    for field in field_list:
        target_field = ogr.FieldDefn(field, ogr.OFTReal)
        target_layer.CreateField(target_field)

    LOGGER.info('Entering iteration to create and set the features')
    # For each inner dictionary (for each point) create a point
    for point_dict in wind_data.values():
        geom = ogr.Geometry(ogr.wkbPoint)
        latitude = float(point_dict['LATI'])
        longitude = float(point_dict['LONG'])
        # When projecting to WGS84, extents -180 to 180 are used for
        # longitude. In case input longitude is from -360 to 0 convert
        if longitude < -180:
            longitude += 360
        if need_geotranform:
            point_x, point_y, _ = coord_trans.TransformPoint(
                longitude, latitude)
            geom.AddPoint(point_x, point_y)
        else:
            geom.AddPoint_2D(longitude, latitude)

        target_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_layer.CreateFeature(target_feature)

        for field_name in point_dict:
            field_index = target_feature.GetFieldIndex(field_name)
            target_feature.SetField(field_index, point_dict[field_name])

        target_feature.SetGeometryDirectly(geom)
        target_layer.SetFeature(target_feature)
        target_feature = None

    LOGGER.info('Finished _wind_data_to_point_vector')
    target_vector = None


def _clip_and_reproject_vector(base_vector_path, clip_vector_path,
                               target_vector_path, work_dir):
    """Clip a vector against an AOI and output result in AOI coordinates.

    Args:
        base_vector_path (str): path to a base vector
        clip_vector_path (str): path to an AOI vector
        target_vector_path (str): desired output path to write the clipped
            base against AOI in AOI's coordinate system.
        work_dir (str): path to create a temp folder for saving temporary
            files. The temp folder will be deleted when function finishes.

    Returns:
        None.
    """
    LOGGER.info('Entering _clip_and_reproject_vector')
    temp_dir = tempfile.mkdtemp(dir=work_dir, prefix='clip-reproject-')
    file_ext, driver_name = _get_file_ext_and_driver_name(target_vector_path)

    # Get the base and target spatial reference in Well Known Text
    base_sr_wkt = pygeoprocessing.get_vector_info(base_vector_path)[
        'projection_wkt']
    target_sr_wkt = pygeoprocessing.get_vector_info(clip_vector_path)[
        'projection_wkt']

    # Create path for the reprojected shapefile
    clipped_vector_path = os.path.join(
        temp_dir, 'clipped_vector' + file_ext)
    reprojected_clip_path = os.path.join(
        temp_dir, 'reprojected_clip_vector' + file_ext)

    if base_sr_wkt != target_sr_wkt:
        # Reproject clip vector to the spatial reference of the base vector.
        # Note: reproject_vector can be expensive if vector has many features.
        pygeoprocessing.reproject_vector(
            clip_vector_path, base_sr_wkt, reprojected_clip_path,
            driver_name=driver_name)

    # Clip the base vector to the AOI
    _clip_vector_by_vector(
        base_vector_path, reprojected_clip_path, clipped_vector_path, temp_dir)

    # Reproject the clipped base vector to the spatial reference of clip vector
    pygeoprocessing.reproject_vector(
        clipped_vector_path, target_sr_wkt, target_vector_path,
        driver_name=driver_name)

    shutil.rmtree(temp_dir, ignore_errors=True)
    LOGGER.info('Finished _clip_and_reproject_vector')


def _clip_vector_by_vector(
        base_vector_path, clip_vector_path, target_vector_path, work_dir):
    """Clip a vector from another vector keeping features.

    Create a new target vector where base features are contained in the
        polygon in clip_vector_path. Assumes all data are in the same
        projection.

    Args:
        base_vector_path (str): path to a vector to clip.
        clip_vector_path (str): path to a polygon vector for clipping.
        target_vector_path (str): output path for the clipped vector.
        work_dir (str): path to create a temp folder for saving temporary
            files. The temp folder will be deleted when function finishes.

    Returns:
        None.

    """
    LOGGER.info('Entering _clip_vector_by_vector')

    file_ext, driver_name = _get_file_ext_and_driver_name(target_vector_path)

    # Get the base and target spatial reference in Well Known Text
    base_sr_wkt = pygeoprocessing.get_vector_info(base_vector_path)[
        'projection_wkt']
    target_sr_wkt = pygeoprocessing.get_vector_info(clip_vector_path)[
        'projection_wkt']

    if base_sr_wkt != target_sr_wkt:
        # Reproject clip vector to the spatial reference of the base vector.
        # Note: reproject_vector can be expensive if vector has many features.
        temp_dir = tempfile.mkdtemp(dir=work_dir, prefix='clip-')
        reprojected_clip_vector_path = os.path.join(
            temp_dir, 'reprojected_clip_vector' + file_ext)
        pygeoprocessing.reproject_vector(
            clip_vector_path, base_sr_wkt, reprojected_clip_vector_path,
            driver_name=driver_name)
        clip_vector_path = reprojected_clip_vector_path

    # Get layer and geometry informations from base vector path
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()
    base_layer_defn = base_layer.GetLayerDefn()
    base_geom_type = base_layer.GetGeomType()

    clip_vector = gdal.OpenEx(clip_vector_path, gdal.OF_VECTOR)
    clip_layer = clip_vector.GetLayer()

    target_driver = gdal.GetDriverByName(driver_name)
    target_vector = target_driver.Create(
        target_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    target_layer = target_vector.CreateLayer(
        base_layer_defn.GetName(), base_layer.GetSpatialRef(), base_geom_type)
    base_layer.Clip(clip_layer, target_layer)

    empty_clip = False
    # Check if the feature count is less than 1, indicating the two vectors
    # did not intersect. This will raise a ValueError below. GetFeatureCount
    # can return -1 if the count is not known or too expensive to compute.
    if target_layer.GetFeatureCount() <= 0:
        empty_clip = True

    # Allow function to clean up resources
    target_layer = None
    target_vector = None
    clip_vector = None
    clip_vector = None
    base_layer = None
    base_vector = None

    if base_sr_wkt != target_sr_wkt:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if empty_clip:
        raise ValueError(
            f"Clipping {base_vector_path} by {clip_vector_path} returned 0"
            " features. If an AOI was provided this could mean the AOI and"
            " Wind Data do not intersect spatially.")

    LOGGER.info('Finished _clip_vector_by_vector')


def _calculate_distances_land_grid(base_point_vector_path, base_raster_path,
                                   target_dist_raster_path, work_dir):
    """Creates a distance transform raster.

    The distances are calculated based on the shortest distances of each point
    feature in 'base_point_vector_path' and each feature's 'L2G' field.

    Args:
        base_point_vector_path (str): path to an OGR shapefile that has
            the desired features to get the distance from.
        base_raster_path (str): path to a GDAL raster that is used to
            get the proper extents and configuration for the new raster.
        target_dist_raster_path (str): path to a GDAL raster for the final
            distance transform raster output.
        work_dir (str): path to create a temp folder for saving files.

    Returns:
        None.

    """
    LOGGER.info('Starting _calculate_distances_land_grid.')
    temp_dir = tempfile.mkdtemp(dir=work_dir, prefix='calc-dist-land')

    # Open the point shapefile and get the layer
    base_point_vector = gdal.OpenEx(base_point_vector_path, gdal.OF_VECTOR)
    base_point_layer = base_point_vector.GetLayer()
    # A list to hold the land to grid distances in order for each point
    # features 'L2G' field
    l2g_dist = []
    # A list to hold the individual distance transform path's in order
    land_point_dist_raster_path_list = []

    # Get the original layer definition which holds needed attribute values
    base_layer_defn = base_point_layer.GetLayerDefn()
    file_ext, driver_name = _get_file_ext_and_driver_name(
        base_point_vector_path)
    output_driver = ogr.GetDriverByName(driver_name)
    single_feature_vector_path = os.path.join(
        temp_dir, 'single_feature' + file_ext)
    target_vector = output_driver.CreateDataSource(single_feature_vector_path)

    # Create the new layer for target_vector using same name and
    # geometry type from base_vector as well as spatial reference
    target_layer = target_vector.CreateLayer(base_layer_defn.GetName(),
                                             base_point_layer.GetSpatialRef(),
                                             base_layer_defn.GetGeomType())

    # Get the number of fields in original_layer
    base_field_count = base_layer_defn.GetFieldCount()

    # For every field, create a duplicate field and add it to the new
    # shapefiles layer
    for fld_index in range(base_field_count):
        base_field = base_layer_defn.GetFieldDefn(fld_index)
        target_field = ogr.FieldDefn(base_field.GetName(),
                                     base_field.GetType())
        # NOT setting the WIDTH or PRECISION because that seems to be
        # unneeded and causes interesting OGR conflicts
        target_layer.CreateField(target_field)

    # Create a new shapefile with only one feature to burn onto a raster
    # in order to get the distance transform based on that one feature
    for feature_index, point_feature in enumerate(base_point_layer):
        # Get the point features land to grid value and add it to the list
        field_index = point_feature.GetFieldIndex('L2G')
        l2g_dist.append(float(point_feature.GetField(field_index)))

        # Copy original_datasource's feature and set as new shapes feature
        output_feature = ogr.Feature(feature_def=target_layer.GetLayerDefn())

        # Since the original feature is of interest add its fields and
        # Values to the new feature from the intersecting geometries
        # The False in SetFrom() signifies that the fields must match
        # exactly
        output_feature.SetFrom(point_feature, False)
        target_layer.CreateFeature(output_feature)
        target_vector.SyncToDisk()
        target_layer.DeleteFeature(point_feature.GetFID())

        dist_raster_path = os.path.join(temp_dir,
                                        'dist_%s.tif' % feature_index)
        _create_distance_raster(base_raster_path, single_feature_vector_path,
                                dist_raster_path, work_dir)
        # Add each features distance transform result to list
        land_point_dist_raster_path_list.append(dist_raster_path)

    target_layer = None
    target_vector = None
    base_point_layer = None
    base_point_vector = None
    l2g_dist_array = numpy.array(l2g_dist)

    def _min_land_ocean_dist(*grid_distances):
        """Aggregate each features distance transform output and create one
            distance output that has the shortest distances combined with each
            features land to grid distance

        Args:
            *grid_distances (numpy.ndarray): a variable number of numpy.ndarray

        Returns:
            a numpy.ndarray of the shortest distances

        """
        # Get the shape of the incoming numpy arrays
        # Initialize with land to grid distances from the first array
        min_distances = numpy.min(grid_distances, axis=0)
        min_land_grid_dist = l2g_dist_array[numpy.argmin(grid_distances, axis=0)]
        return min_distances + min_land_grid_dist

    pygeoprocessing.raster_calculator(
        [(path, 1)
         for path in land_point_dist_raster_path_list], _min_land_ocean_dist,
        target_dist_raster_path, _TARGET_DATA_TYPE, _TARGET_NODATA)

    shutil.rmtree(temp_dir, ignore_errors=True)

    LOGGER.info('Finished _calculate_distances_land_grid.')


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Wind Energy.

    Args:
        args (dict): The args dictionary.

        limit_to=None (str or None): If a str key, only this args parameter
            will be validated.  If ````None````, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the str validation warning.

    """
    return validation.validate(args, ARGS_SPEC['args'],
                               ARGS_SPEC['args_with_spatial_overlap'])
