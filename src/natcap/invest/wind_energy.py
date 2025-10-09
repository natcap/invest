"""InVEST Wind Energy model."""
import logging
import math
import os
import pickle
import shutil
import tempfile

import numpy
import pygeoprocessing
import shapely.ops
import shapely.prepared
import shapely.wkb
import shapely.wkt
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from scipy import integrate
from shapely import speedups
from rtree import index

from . import gettext
from . import spec
from . import utils
from . import validation
from .file_registry import FileRegistry
from .unit_registry import u

LOGGER = logging.getLogger(__name__)
speedups.enable()

WIND_DATA_FIELDS_FROM_INPUT = [
    spec.NumberOutput(
        id="long",
        about=gettext("Longitude of the data point."),
        units=u.degree
    ),
    spec.NumberOutput(
        id="lati",
        about=gettext("Latitude of the data point."),
        units=u.degree
    ),
    spec.NumberOutput(
        id="lam",
        about=gettext(
            "Weibull scale factor at the reference hub height at this"
            " point."
        ),
        units=u.none
    ),
    spec.NumberOutput(
        id="k",
        about=gettext("Weibull shape factor at this point."),
        units=u.none
    ),
    spec.NumberOutput(
        id="ref",
        about=gettext(
            "The reference hub height at this point, at which wind"
            " speed data was collected and LAM was estimated."
        ),
        units=u.meter
    ),
    spec.NumberOutput(
        id="ref_lam",
        about=gettext(
            "Weibull scale factor at the reference hub height at this"
            " point."
        ),
        units=u.degree
    ),
    spec.NumberOutput(
        id="Dens_W/m2",
        about=gettext("Power density at this point."),
        units=u.watt / u.meter**2
    ),
    spec.NumberOutput(
        id="Harv_MWhr",
        about=gettext(
            "Predicted energy harvested from a wind farm centered on"
            " this point."
        ),
        units=u.megawatt_hour / u.year
    )
]

VALUATION_WIND_DATA_FIELDS = [
    spec.NumberOutput(
        id="DepthVal",
        about=gettext("Ocean depth at this point."),
        units=u.meter
    ),
    spec.NumberOutput(
        id="DistVal",
        about=gettext("Distance to shore from this point."),
        units=u.meter,
    ),
    spec.NumberOutput(
        id="CO2_Tons",
        about=gettext(
            "Offset carbon emissions for a farm centered on this"
            " point. Included only if Valuation is run."
        ),
        units=u.metric_ton / u.year,
    ),
    spec.NumberOutput(
        id="Level_Cost",
        about=gettext(
            "Energy price that would be required to set the present"
            " value of a farm centered on this point equal to zero."
            " Included only if Valuation is run."
        ),
        units=u.currency / u.kilowatt_hour,
    ),
    spec.NumberOutput(
        id="NPV",
        about=gettext(
            "The net present value of a farm centered on this point."
            " Included only if Valuation is run."
        ),
        units=u.currency,
    )
]

MODEL_SPEC = spec.ModelSpec(
    model_id="wind_energy",
    model_title=gettext("Wind Energy Production"),
    userguide="wind_energy.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["wind_data_path", "aoi_vector_path", "bathymetry_path",
         "land_polygon_vector_path", "global_wind_parameters_path"],
        ["turbine_parameters_path", "number_of_turbines", "min_depth",
         "max_depth", "min_distance", "max_distance"],
        ["valuation_container", "foundation_cost", "discount_rate",
         "grid_points_path", "avg_grid_distance", "price_table",
         "wind_schedule", "wind_price", "rate_change"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.CSVInput(
            id="wind_data_path",
            name=gettext("wind data points"),
            about=gettext("Table of Weibull parameters for each wind data point."),
            columns=[
                spec.NumberInput(
                    id="long",
                    about=gettext("Longitude of the data point."),
                    units=u.degree
                ),
                spec.NumberInput(
                    id="lati",
                    about=gettext("Latitude of the data point."),
                    units=u.degree
                ),
                spec.NumberInput(
                    id="lam",
                    about=gettext(
                        "Weibull scale factor at the reference hub height at this point."
                    ),
                    units=u.none
                ),
                spec.NumberInput(
                    id="k",
                    about=gettext("Weibull shape factor at this point."),
                    units=u.none
                ),
                spec.NumberInput(
                    id="ref",
                    about=gettext(
                        "The reference hub height at this point, at which wind speed data"
                        " was collected and LAM was estimated."
                    ),
                    units=u.meter
                )
            ],
            index_col=None
        ),
        spec.AOI.model_copy(update=dict(
            id="aoi_vector_path",
            about=gettext(
                "Map of the area(s) of interest over which to run the model and aggregate"
                " valuation results."
            ),
            projected=True,
            projection_units=u.meter
        )),
        spec.SingleBandRasterInput(
            id="bathymetry_path",
            name=gettext("bathymetry"),
            about=gettext("Map of ocean depth. Values should be negative."),
            data_type=float,
            units=u.meter,
            projected=None
        ),
        spec.VectorInput(
            id="land_polygon_vector_path",
            name=gettext("land polygon"),
            about=gettext(
                "Map of the coastlines of landmasses in the area of interest."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[],
            projected=None
        ),
        spec.CSVInput(
            id="global_wind_parameters_path",
            name=gettext("global wind energy parameters"),
            about=gettext("A table of wind energy infrastructure parameters."),
            columns=None,
            rows=[
                spec.NumberInput(
                    id="air_density",
                    about=gettext("Standard atmosphere air density."),
                    units=u.kilogram / u.meter**3
                ),
                spec.NumberInput(
                    id="exponent_power_curve",
                    about=gettext("Exponent to use in the power curve function."),
                    units=u.none
                ),
                spec.RatioInput(
                    id="decommission_cost",
                    about=gettext(
                        "Cost to decommission a turbine as a proportion of the total"
                        " upfront costs (cables, foundations, installation?)"
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="operation_maintenance_cost",
                    about=gettext(
                        "The operations and maintenance costs as a proportion of"
                        " capex_arr"
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="miscellaneous_capex_cost",
                    about=gettext("The miscellaneous costs as a proportion of capex_arr"),
                    units=None
                ),
                spec.RatioInput(
                    id="installation_cost",
                    about=gettext("The installation costs as a proportion of capex_arr"),
                    units=None
                ),
                spec.NumberInput(
                    id="infield_cable_length",
                    about=gettext("The length of infield cable."),
                    units=u.kilometer
                ),
                spec.NumberInput(
                    id="infield_cable_cost",
                    about=gettext("The cost of infield cable."),
                    units=u.currency / u.kilometer
                ),
                spec.NumberInput(
                    id="mw_coef_ac",
                    about=gettext("Cost of AC cable that scales with capacity."),
                    units=u.currency / u.megawatt
                ),
                spec.NumberInput(
                    id="mw_coef_dc",
                    about=gettext("Cost of DC cable that scales with capacity."),
                    units=u.currency / u.megawatt
                ),
                spec.NumberInput(
                    id="cable_coef_ac",
                    about=gettext("Cost of AC cable that scales with length."),
                    units=u.currency / u.kilometer
                ),
                spec.NumberInput(
                    id="cable_coef_dc",
                    about=gettext("Cost of DC cable that scales with length."),
                    units=u.currency / u.kilometer
                ),
                spec.NumberInput(
                    id="ac_dc_distance_break",
                    about=gettext(
                        "The threshold above which a wind farm’s distance from the grid"
                        " requires a switch from AC to DC power to overcome line losses"
                        " which reduce the amount of energy delivered"
                    ),
                    units=u.kilometer
                ),
                spec.IntegerInput(
                    id="time_period",
                    about=gettext("The expected lifetime of the facility"),
                    units=u.year
                ),
                spec.NumberInput(
                    id="carbon_coefficient",
                    about=gettext(
                        "Factor that translates carbon-free wind power to a corresponding"
                        " amount of avoided CO2 emissions"
                    ),
                    units=u.metric_ton / u.kilowatt_hour
                ),
                spec.NumberInput(
                    id="air_density_coefficient",
                    about=gettext(
                        "The reduction in air density per meter above sea level"
                    ),
                    units=u.kilogram / u.meter**4
                ),
                spec.RatioInput(
                    id="loss_parameter",
                    about=gettext(
                        "The fraction of energy lost due to downtime, power conversion"
                        " inefficiency, and electrical grid losses"
                    ),
                    units=None
                )
            ],
            index_col=None
        ),
        spec.CSVInput(
            id="turbine_parameters_path",
            name=gettext("turbine parameters"),
            about=gettext("A table of parameters specific to the type of turbine."),
            columns=None,
            rows=[
                spec.NumberInput(
                    id="hub_height",
                    about=gettext("Height of the turbine hub above sea level."),
                    units=u.meter
                ),
                spec.NumberInput(
                    id="cut_in_wspd",
                    about=gettext(
                        "Wind speed at which the turbine begins producing power."
                    ),
                    units=u.meter / u.second
                ),
                spec.NumberInput(
                    id="rated_wspd",
                    about=gettext(
                        "Minimum wind speed at which the turbine reaches its rated power"
                        " output."
                    ),
                    units=u.meter / u.second
                ),
                spec.NumberInput(
                    id="cut_out_wspd",
                    about=gettext(
                        "Wind speed above which the turbine stops generating power for"
                        " safety reasons."
                    ),
                    units=u.meter / u.second
                ),
                spec.NumberInput(
                    id="turbine_rated_pwr",
                    about="The turbine's rated power output.",
                    units=u.megawatt
                ),
                spec.NumberInput(
                    id="turbine_cost",
                    about=gettext("The cost of one turbine."),
                    units=u.currency
                )
            ],
            index_col=None
        ),
        spec.NumberInput(
            id="number_of_turbines",
            name=gettext("number of turbines"),
            about=gettext("The number of wind turbines per wind farm."),
            units=u.none,
            expression="value > 0"
        ),
        spec.NumberInput(
            id="min_depth",
            name=gettext("minimum depth"),
            about=gettext("Minimum depth for offshore wind farm installation."),
            units=u.meter
        ),
        spec.NumberInput(
            id="max_depth",
            name=gettext("maximum depth"),
            about=gettext("Maximum depth for offshore wind farm installation."),
            units=u.meter
        ),
        spec.NumberInput(
            id="min_distance",
            name=gettext("minimum distance"),
            about=gettext(
                "Minimum distance from shore for offshore wind farm installation."
            ),
            units=u.meter
        ),
        spec.NumberInput(
            id="max_distance",
            name=gettext("maximum distance"),
            about=gettext(
                "Maximum distance from shore for offshore wind farm installation."
            ),
            units=u.meter
        ),
        spec.BooleanInput(
            id="valuation_container",
            name=gettext("run valuation"),
            about=gettext("Run the valuation component of the model."),
            required=False
        ),
        spec.NumberInput(
            id="foundation_cost",
            name=gettext("foundation cost"),
            about=gettext("The cost of the foundation for one turbine."),
            required="valuation_container",
            allowed="valuation_container",
            units=u.currency
        ),
        spec.RatioInput(
            id="discount_rate",
            name=gettext("discount rate"),
            about=gettext("Annual discount rate to apply to valuation."),
            required="valuation_container",
            allowed="valuation_container",
            units=None
        ),
        spec.CSVInput(
            id="grid_points_path",
            name=gettext("grid connection points"),
            about=gettext(
                "Table of grid and land connection points to which cables will connect."
                " Required if Run Valuation is selected and Average Shore-to-Grid"
                " Distance is not provided."
            ),
            required="valuation_container and not avg_grid_distance",
            allowed="valuation_container",
            columns=[
                spec.IntegerInput(
                    id="id", about=gettext("Unique identifier for each point.")
                ),
                spec.OptionStringInput(
                    id="type",
                    about=gettext("The type of connection at this point."),
                    options=[
                        spec.Option(key="LAND", about="This is a land connection point"),
                        spec.Option(key="GRID", about="This is a grid connection point")
                    ]
                ),
                spec.NumberInput(
                    id="lati",
                    about=gettext("Latitude of the connection point."),
                    units=u.degree
                ),
                spec.NumberInput(
                    id="long",
                    about=gettext("Longitude of the connection point."),
                    units=u.degree
                )
            ],
            index_col="id"
        ),
        spec.NumberInput(
            id="avg_grid_distance",
            name=gettext("average shore-to-grid distance"),
            about=gettext(
                "Average distance to the onshore grid from coastal cable landing points."
                " Required if Run Valuation is selected and the Grid Connection Points"
                " table is not provided."
            ),
            required="valuation_container and not grid_points_path",
            allowed="valuation_container",
            units=u.kilometer,
            expression="value > 0"
        ),
        spec.BooleanInput(
            id="price_table",
            name=gettext("use price table"),
            about=gettext(
                "Use a Wind Energy Price Table instead of calculating annual prices from"
                " the initial Energy Price and Rate of Price Change inputs."
            ),
            required="valuation_container",
            allowed="valuation_container"
        ),
        spec.CSVInput(
            id="wind_schedule",
            name=gettext("wind energy price table"),
            about=(
                "Table of yearly prices for wind energy. There must be a row for each"
                " year in the lifespan given in the 'time_period' column in the Global"
                " Wind Energy Parameters table. Required if Run Valuation and Use Price"
                " Table are selected."
            ),
            required="valuation_container and price_table",
            allowed="price_table",
            columns=[
                spec.NumberInput(
                    id="year",
                    about=gettext(
                        "Consecutive years for each year in the lifespan of the wind"
                        " farm. These may be the actual years: 2010, 2011, 2012..., or"
                        " the number of the years after the starting date: 1, 2, 3,..."
                    ),
                    units=u.year_AD
                ),
                spec.NumberInput(
                    id="price",
                    about=gettext("Price of energy for each year."),
                    units=u.currency / u.kilowatt_hour
                )
            ],
            index_col="year"
        ),
        spec.NumberInput(
            id="wind_price",
            name=gettext("price of energy"),
            about=gettext(
                "The initial price of wind energy, at the first year in the wind energy"
                " farm lifespan. Required if Run Valuation is selected and Use Price"
                " Table is not selected."
            ),
            required="valuation_container and (not price_table)",
            allowed="valuation_container and not price_table",
            units=u.currency / u.kilowatt_hour
        ),
        spec.RatioInput(
            id="rate_change",
            name=gettext("rate of price change"),
            about=gettext(
                "The annual rate of change in the price of wind energy. Required if Run"
                " Valuation is selected and Use Price Table is not selected."
            ),
            required="valuation_container and not price_table",
            allowed="valuation_container and not price_table",
            units=None
        )
    ],
    outputs=[
        spec.VectorOutput(
            id="final_wind_point_vector_path",
            path="output/wind_energy_points.shp",
            about=gettext("Map of summarized data at each point."),
            geometry_types={"POINT"},
            fields=WIND_DATA_FIELDS_FROM_INPUT + VALUATION_WIND_DATA_FIELDS
        ),
        spec.SingleBandRasterOutput(
            id="bathymetry_path",
            path="intermediate/bathymetry_resampled.tif",
            about=gettext("Clipped and reprojected bathymetry map"),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="bathymetry_proj_raster_path",
            path="intermediate/bathymetry_projected.tif",
            about=gettext("Clipped and reprojected bathymetry map"),
            data_type=float,
            units=u.meter
        ),
        spec.SingleBandRasterOutput(
            id="depth_mask_path",
            path="intermediate/depth_mask.tif",
            about=gettext(
                "Bathymetry map masked to show only the pixels that fall within"
                " the allowed depth range"
            ),
            data_type=float,
            units=u.meter
        ),
        spec.VectorOutput(
            id="land_poly_proj_vector_path",
            path="intermediate/projected_clipped_land_poly.shp",
            about=gettext("Clipped and reprojected land polygon vector"),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[],
        ),
        spec.SingleBandRasterOutput(
            id="dist_trans_path",
            path="intermediate/distance_trans.tif",
            about=gettext("Distance to shore from each pixel"),
            data_type=float,
            units=u.meter,
        ),
        spec.SingleBandRasterOutput(
            id="dist_mask_path",
            path="intermediate/distance_mask.tif",
            about=gettext(
                "Distance to shore, masked to show only the pixels that fall"
                " within the allowed distance range"
            ),
            data_type=float,
            units=u.meter,
        ),
        spec.SingleBandRasterOutput(
            id="final_dist_raster_path",
            path="intermediate/val_distance_trans.tif",
            about=gettext(
                "Distance to shore, masked to show only the pixels that fall"
                " within the allowed distance range"
            ),
            data_type=float,
            units=u.meter,
        ),
        spec.SingleBandRasterOutput(
            id="initial_harvested_rater_path",
            path="intermediate/harvested_unmasked.tif",
            about=gettext(
                "Rasterized harvested values from the wind data point vector,"
                " not masked by depth or distance constraints"
            ),
            data_type=float,
            units=u.megawatt_hour / u.year,
        ),
        spec.SingleBandRasterOutput(
            id="harvested_masked_path",
            path="intermediate/harvested_energy_MWhr_per_yr.tif",
            about=gettext(
                "Rasterized harvested values from the wind data point vector,"
                " masked by depth and distance constraints"
            ),
            data_type=float,
            units=u.megawatt_hour / u.year,
        ),
        spec.SingleBandRasterOutput(
            id="carbon_raster_path",
            path="intermediate/carbon_emissions_tons.tif",
            about=gettext(
                "Map of offset carbon emissions for a farm centered on each pixel"
            ),
            data_type=float,
            units=u.metric_ton / u.year,
        ),
        spec.SingleBandRasterOutput(
            id="levelized_raster_path",
            path="intermediate/levelized_cost_price_per_kWh.tif",
            about=gettext(
                "Map of the energy price that would be required to set the"
                " present value of a farm centered on each pixel equal to zero."
            ),
            data_type=float,
            units=u.currency / u.kilowatt_hour,
        ),
        spec.SingleBandRasterOutput(
            id="npv_raster_path",
            path="intermediate/npv.tif",
            about=gettext(
                "Map of the net present value of a farm centered on each pixel."
            ),
            data_type=float,
            units=u.currency,
        ),
        spec.VectorOutput(
            id="grid_point_vector_path",
            path="intermediate/val_grid_points.shp",
            about=gettext("Point vector created from the grid point data provided"
                " in the Grid Connection Points CSV. Contains all connection points"
                " of type 'GRID.'"),
            geometry_types={"POINT"},
            fields=[],
        ),
        spec.VectorOutput(
            id="grid_projected_vector_path",
            path="intermediate/grid_point_projected_clipped.shp",
            about=gettext("Point vector created from the grid point data provided"
                " in the Grid Connection Points CSV, clipped to the AOI. Contains"
                " all connection points of type 'GRID' that fall within the AOI"),
            geometry_types={"POINT"},
            fields=[],
        ),
        spec.VectorOutput(
            id="land_point_vector_path",
            path="intermediate/val_land_points.shp",
            about=gettext("Point vector created from the grid point data provided"
                " in the Grid Connection Points CSV. Contains all connection points"
                " of type 'LAND.'"),
            geometry_types={"POINT"},
            fields=[],
        ),
        spec.VectorOutput(
            id="land_projected_vector_path",
            path="intermediate/land_point_projected_clipped.shp",
            about=gettext("Point vector created from the grid point data provided"
                " in the Grid Connection Points CSV, clipped to the AOI. Contains"
                " all connection points of type 'LAND' that fall within the AOI"),
            geometry_types={"POINT"},
            fields=[],
        ),
        spec.VectorOutput(
            id="land_to_grid_vector_path",
            path="intermediate/land_point_to_grid.shp",
            about=gettext("Point vector containing shortest distances from each"
                " land point within the AOI to the grid points. Only created if"
                " there are both LAND and GRID connection points within the AOI."),
            geometry_types={"POINT"},
            fields=[
                spec.StringOutput(
                    id="type",
                    about=gettext("Value: 'land'"),
                ),
                spec.NumberOutput(
                    id="lati",
                    about=gettext("Latitude of the connection point."),
                    units=u.degree
                ),
                spec.NumberOutput(
                    id="long",
                    about=gettext("Longitude of the connection point."),
                    units=u.degree
                ),
                spec.NumberOutput(
                    id="L2G",
                    about=gettext("Distance to grid from this connection point."),
                    units=u.kilometer
                )
            ],
        ),
        spec.SingleBandRasterOutput(
            id="land_poly_dist_raster_path",
            path="intermediate/land_poly_dist.tif",
            about=gettext("Map of distance to shore, calculated using the land"
                " polygon when no grid points are provided."),
            data_type=float,
            units=u.meter,
        ),
        spec.FileOutput(
            id="wind_data_pickle_path",
            path="intermediate/wind_data.pickle",
            about=gettext("Pickled wind data dictionary")
        ),
        spec.VectorOutput(
            id="wind_point_vector_path",
            path="intermediate/wind_energy_points_from_data.shp",
            about=gettext("Point vector created from input wind point data"),
            geometry_types={"POINT"},
            fields=WIND_DATA_FIELDS_FROM_INPUT
        ),
        spec.VectorOutput(
            id="unmasked_wind_point_vector_path",
            path="intermediate/unmasked_wind_energy_points.shp",
            about=gettext("Input wind point data, clipped to the AOI"),
            geometry_types={"POINT"},
            fields=WIND_DATA_FIELDS_FROM_INPUT + [
                spec.NumberOutput(
                    id="Masked",
                    about=gettext(
                        "Indicates whether or not a point was masked out due to"
                        " depth or distance constraints in the final output"
                    )
                )
            ]
        ),
        spec.TASKGRAPH_CACHE
    ]
)


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

# The names for the computed output fields to be added to the point shapefile:
_DENSITY_FIELD_NAME = 'Dens_W/m2'
_HARVESTED_FIELD_NAME = 'Harv_MWhr'
_DEPTH_FIELD_NAME = 'DepthVal'
_DIST_FIELD_NAME = 'DistVal'
_NPV_FIELD_NAME = 'NPV'
_LEVELIZED_COST_FIELD_NAME = 'Level_Cost'
_CARBON_FIELD_NAME = 'CO2_Tons'

# The field names used to mask out points in the output shapefile:
_MASK_KEYS = [_DEPTH_FIELD_NAME, _DIST_FIELD_NAME]

# The field name that will be added to the intermediate point shapefile,
# indicating whether or not a point is masked out by depth / distance.
_MASK_FIELD_NAME = 'Masked'

# Resample method for target rasters
_TARGET_RESAMPLE_METHOD = 'near'
# Target pixel size, in meters. Given the increased availability of
# high-resolution wind data (e.g. 2km horizontal resolution), we want a fine
# resolution for rasterizing wind point vector values.
_TARGET_PIXEL_SIZE = (1500, -1500)


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
            of the land polygon that is of interest (required)
        bathymetry_path (str): a path to a GDAL raster that has the depth
            values of the area of interest (required)
        land_polygon_vector_path (str): a path to an OGR polygon vector that
            provides a coastline for determining distances from wind farm bins
            (required)
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
            for offshore wind farm installation (meters) (required)
        max_distance (float): a float value for the maximum distance from shore
            for offshore wind farm installation (meters) (required)
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
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    LOGGER.info('Starting the Wind Energy Model')
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    # Resample the bathymetry raster if it does not have square pixel size
    try:
        bathy_pixel_size = pygeoprocessing.get_raster_info(
            args['bathymetry_path'])['pixel_size']
        mean_pixel_size, _ = utils.mean_pixel_size_and_area(bathy_pixel_size)
        target_pixel_size = (mean_pixel_size, -mean_pixel_size)
        LOGGER.debug(f'Target pixel size: {target_pixel_size}')
        bathymetry_path = args['bathymetry_path']
        # The task list would be empty for clipping and reprojecting bathymetry
        bathy_dependent_task_list = None

    except ValueError:
        LOGGER.debug(
            f"{args['bathymetry_path']} has pixels that are not square. "
            "Resampling the raster to have square pixels.")
        bathymetry_path = file_registry['bathymetry_path']
        mean_pixel_size = numpy.min(numpy.absolute(bathy_pixel_size))
        # Use it as the target pixel size for resampling and warping rasters
        target_pixel_size = (mean_pixel_size, -mean_pixel_size)
        LOGGER.debug(f'Target pixel size: {target_pixel_size}')

        resample_bathymetry_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(args['bathymetry_path'], target_pixel_size,
                  file_registry['bathymetry_path'], _TARGET_RESAMPLE_METHOD),
            target_path_list=[file_registry['bathymetry_path']],
            task_name='resample_bathymetry')

        # Build the task list when clipping and reprojecting bathymetry later.
        bathy_dependent_task_list = [resample_bathymetry_task]

    # Read the biophysical turbine parameters into a dictionary
    turbine_dict = MODEL_SPEC.get_input(
        'turbine_parameters_path').get_validated_dataframe(
        args['turbine_parameters_path']).iloc[0].to_dict()
    # Read the biophysical global parameters into a dictionary
    global_params_dict = MODEL_SPEC.get_input(
        'global_wind_parameters_path').get_validated_dataframe(
        args['global_wind_parameters_path']).iloc[0].to_dict()

    # Combine the turbine and global parameters into one dictionary
    parameters_dict = global_params_dict.copy()
    parameters_dict.update(turbine_dict)

    LOGGER.debug(f'Biophysical Turbine Parameters: {parameters_dict}')

    if args['valuation_container']:
        LOGGER.info(
            'Valuation Selected. Checking required parameters from CSV files.')

        # If Price Table provided use that for price of energy, validate inputs
        time = parameters_dict['time_period']
        if args['price_table']:
            wind_price_df = MODEL_SPEC.get_input(
                'wind_schedule').get_validated_dataframe(
                args['wind_schedule']).sort_index()  # sort by year

            year_count = len(wind_price_df)
            if year_count != time + 1:
                raise ValueError(
                    "The 'time' argument in the Global Wind Energy Parameters "
                    "file must equal the number of years provided in the price "
                    "table.")

            # Save the price values into a list where the indices of the list
            # indicate the time steps for the lifespan of the wind farm
            price_list = wind_price_df['price'].tolist()
        else:
            # Build up a list of price values where the indices of the list
            # are the time steps for the lifespan of the farm and values
            # are adjusted based on the rate of change
            price_list = []
            for time_step in range(int(time) + 1):
                price_list.append(
                    args["wind_price"] * (1 + args["rate_change"])**(time_step))

    compute_density_harvested_task = task_graph.add_task(
        func=_compute_density_harvested_fields,
        args=(args['wind_data_path'], parameters_dict,
              args['number_of_turbines'],
              file_registry['wind_data_pickle_path']),
        target_path_list=[file_registry['wind_data_pickle_path']],
        task_name='compute_density_harvested_fields')

    aoi_vector_path = args['aoi_vector_path']
    reproject_bathy_task = task_graph.add_task(
        func=_reproject_bathymetry,
        args=(bathymetry_path, aoi_vector_path, _TARGET_PIXEL_SIZE,
              file_registry['bathymetry_proj_raster_path']),
        target_path_list=[file_registry['bathymetry_proj_raster_path']],
        task_name='reproject_bathymetry')

    LOGGER.info('Create point shapefile from wind data')
    # Use the projection from the AOI as reference to
    # create wind point vector from wind data dictionary
    target_sr_wkt = pygeoprocessing.get_vector_info(
        aoi_vector_path)['projection_wkt']

    wind_data_to_vector_task = task_graph.add_task(
        func=_wind_data_to_point_vector,
        args=(file_registry['wind_data_pickle_path'],
              'wind_data', file_registry['wind_point_vector_path']),
        kwargs={'ref_projection_wkt': target_sr_wkt},
        target_path_list=[file_registry['wind_point_vector_path']],
        task_name='wind_data_to_vector',
        dependent_task_list=[compute_density_harvested_task])

    # Clip the wind energy point shapefile to AOI
    LOGGER.info('Clip and project wind points to AOI')
    clip_wind_vector_task = task_graph.add_task(
        func=_clip_vector_by_vector,
        args=(file_registry['wind_point_vector_path'], aoi_vector_path,
              file_registry['unmasked_wind_point_vector_path'],
              args['workspace_dir']),
        target_path_list=[file_registry['unmasked_wind_point_vector_path']],
        task_name='clip_wind_point_by_aoi',
        dependent_task_list=[wind_data_to_vector_task])

    # Clip and project the land polygon shapefile to AOI
    LOGGER.info('Clip and project land polygon to AOI')
    clip_reproject_land_poly_task = task_graph.add_task(
        func=_clip_and_reproject_vector,
        args=(args['land_polygon_vector_path'], aoi_vector_path,
              file_registry['land_poly_proj_vector_path'],
              args['workspace_dir']),
        target_path_list=[file_registry['land_poly_proj_vector_path']],
        task_name='clip_and_reproject_land_poly_to_aoi')

    # Rasterize land polygon and calculate distance transform
    create_distance_raster_task = task_graph.add_task(
        func=_create_distance_raster,
        args=(file_registry['bathymetry_proj_raster_path'],
              file_registry['land_poly_proj_vector_path'],
              file_registry['dist_trans_path'], args['workspace_dir']),
        target_path_list=[file_registry['dist_trans_path']],
        task_name='create_distance_raster',
        dependent_task_list=[reproject_bathy_task,
            clip_reproject_land_poly_task])

    # Create the distance mask:
    LOGGER.info('Creating Distance Mask')
    create_dist_mask_task = task_graph.add_task(
        func=_mask_by_distance,
        args=(file_registry['dist_trans_path'], args['min_distance'],
              args['max_distance'], _TARGET_NODATA,
              file_registry['dist_mask_path']),
        target_path_list=[file_registry['dist_mask_path']],
        task_name='mask_raster_by_distance',
        dependent_task_list=[create_distance_raster_task])

    # Create a mask for values that are out of the range of the depth values:
    # Get the min and max depth values from the arguments and set to a negative
    # value indicating below sea level
    min_depth = abs(args['min_depth']) * -1
    max_depth = abs(args['max_depth']) * -1

    LOGGER.info('Creating Depth Mask')
    create_depth_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(file_registry['bathymetry_proj_raster_path'], 1),
               (min_depth, 'raw'), (max_depth, 'raw')],
              _depth_op, file_registry['depth_mask_path'],
              _TARGET_DATA_TYPE, _TARGET_NODATA),
        target_path_list=[file_registry['depth_mask_path']],
        task_name='mask_depth_on_bathymetry',
        dependent_task_list=[reproject_bathy_task])

    if not args['valuation_container']:
        # Write Depth and Distance mask values to Wind Points Shapefile
        LOGGER.info("Adding mask values to shapefile")
        raster_field_to_vector_list = [
            (file_registry['dist_mask_path'], _DIST_FIELD_NAME),
            (file_registry['depth_mask_path'], _DEPTH_FIELD_NAME)
        ]

        task_graph.add_task(
            func=_index_raster_values_to_point_vector,
            args=(file_registry['unmasked_wind_point_vector_path'],
                  raster_field_to_vector_list,
                  file_registry['final_wind_point_vector_path']),
            kwargs={'mask_keys': _MASK_KEYS,
                    'mask_field': _MASK_FIELD_NAME},
            target_path_list=[file_registry['final_wind_point_vector_path']],
            task_name='add_masked_vals_to_wind_vector',
            dependent_task_list=[clip_wind_vector_task,
                create_dist_mask_task, create_depth_mask_task])

        LOGGER.info('Wind Energy Biophysical Model completed')

        task_graph.close()
        task_graph.join()
        LOGGER.info('Valuation Not Selected. Model completed')
        return

    # Begin the valuation model run:
    LOGGER.info('Starting Wind Energy Valuation Model')

    # Rasterize harvested values:
    LOGGER.info('Creating Harvested Raster')
    create_harvested_raster_task = task_graph.add_task(
        func=pygeoprocessing.new_raster_from_base,
        args=(file_registry['depth_mask_path'],
              file_registry['initial_harvested_rater_path'],
              _TARGET_DATA_TYPE, [_TARGET_NODATA]),
        target_path_list=[file_registry['initial_harvested_rater_path']],
        task_name='create_harvested_raster',
        dependent_task_list=[create_depth_mask_task])

    LOGGER.info('Rasterizing Harvested Points')
    rasterize_harvested_task = task_graph.add_task(
        func=pygeoprocessing.rasterize,
        args=(file_registry['unmasked_wind_point_vector_path'],
              file_registry['initial_harvested_rater_path']),
        kwargs={'option_list': [f'ATTRIBUTE={_HARVESTED_FIELD_NAME}']},
        task_name='rasterize_harvested_points',
        dependent_task_list=[clip_wind_vector_task, create_harvested_raster_task])

    # Mask out any areas where distance or depth has determined that wind farms
    # cannot be located
    LOGGER.info('Mask Harvested raster by depth and distance')
    # We always will have a distance raster if we're running valuation
    harvest_mask_list = [
        file_registry['initial_harvested_rater_path'],
        file_registry['depth_mask_path'],
        file_registry['dist_mask_path']]

    mask_harvested_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(path, 1) for path in harvest_mask_list],
              _mask_out_depth_dist, file_registry['harvested_masked_path'],
              _TARGET_DATA_TYPE, _TARGET_NODATA),
        task_name='mask_harvested_raster',
        target_path_list=[file_registry['harvested_masked_path']],
        dependent_task_list=[rasterize_harvested_task,
            create_dist_mask_task])

    if args['grid_points_path']:
        # Handle Grid Points
        LOGGER.info('Grid Points Provided. Reading in the grid points')

        # Read the grid points csv, and convert it to land and grid dictionary
        grid_land_df = MODEL_SPEC.get_input(
            'grid_points_path').get_validated_dataframe(args['grid_points_path'])

        # Convert the dataframes to dictionaries, using 'ID' (the index) as key
        grid_dict = grid_land_df[grid_land_df['type'] == 'grid'].to_dict('index')
        land_dict = grid_land_df[grid_land_df['type'] == 'land'].to_dict('index')

        # Create a point shapefile from the grid point dictionary.
        # This makes it easier for future distance calculations and provides a
        # nice intermediate output for users
        grid_dict_to_vector_task = task_graph.add_task(
            func=_dictionary_to_point_vector,
            args=(grid_dict, 'grid_points', file_registry['grid_point_vector_path']),
            target_path_list=[file_registry['grid_point_vector_path']],
            task_name='grid_dictionary_to_vector')

        # In case any of the above points lie outside the AOI, clip the
        # shapefiles and then project them to the AOI as well.
        task_graph.add_task(
            func=_clip_and_reproject_vector,
            args=(file_registry['grid_point_vector_path'], aoi_vector_path,
                  file_registry['grid_projected_vector_path'], args['workspace_dir']),
            target_path_list=[file_registry['grid_projected_vector_path']],
            task_name='clip_and_reproject_grid_vector',
            dependent_task_list=[grid_dict_to_vector_task])

        # It is possible that NO grid points lie within the AOI, so we need to
        # handle both cases
        task_graph.join()  # need to join to get grid feature count
        grid_feature_count = _get_feature_count(file_registry['grid_projected_vector_path'])
        if grid_feature_count > 0:
            LOGGER.debug(f'There are {grid_feature_count} grid point(s) within AOI.')
            # It's possible that no land points were provided, and we need to
            # handle both cases
            if land_dict:
                # A bool used to determine if the final distance raster should
                # be calculated without land points later
                calc_grid_dist_without_land = False

                # Create a point shapefile from the land point dictionary.
                # This makes it easier for future distance calculations and
                # provides a nice intermediate output for users
                land_dict_to_vector_task = task_graph.add_task(
                    func=_dictionary_to_point_vector,
                    args=(land_dict, 'land_points', file_registry['land_point_vector_path']),
                    target_path_list=[file_registry['land_point_vector_path']],
                    task_name='land_dictionary_to_vector')

                # In case any of the above points lie outside the AOI, clip the
                # shapefiles and then project them to the AOI as well.
                task_graph.add_task(
                    func=_clip_and_reproject_vector,
                    args=(file_registry['land_point_vector_path'], aoi_vector_path,
                          file_registry['land_projected_vector_path'],
                          args['workspace_dir']),
                    target_path_list=[file_registry['land_projected_vector_path']],
                    task_name='clip_and_reproject_land_vector',
                    dependent_task_list=[land_dict_to_vector_task])

                # It is possible that NO land point lie within the AOI, so we
                # need to handle both cases
                task_graph.join()  # need to join to get land feature count
                land_feature_count = _get_feature_count(
                    file_registry['land_projected_vector_path'])
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
                    land_to_grid_task = task_graph.add_task(
                        func=_calculate_land_to_grid_distance,
                        args=(file_registry['land_projected_vector_path'],
                              file_registry['grid_projected_vector_path'],
                              _LAND_TO_GRID_FIELD,
                              file_registry['land_to_grid_vector_path']),
                        target_path_list=[file_registry['land_to_grid_vector_path']],
                        task_name='calculate_grid_point_to_land_poly')

                    # Calculate distance raster
                    final_dist_task = task_graph.add_task(
                        func=_calculate_distances_land_grid,
                        args=(file_registry['land_to_grid_vector_path'],
                              file_registry['harvested_masked_path'],
                              file_registry['final_dist_raster_path'],
                              args['workspace_dir']),
                        target_path_list=[file_registry['final_dist_raster_path']],
                        task_name='calculate_distances_land_grid',
                        dependent_task_list=[land_to_grid_task,
                                             mask_harvested_task])
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
                    args=(file_registry['harvested_masked_path'],
                          file_registry['grid_projected_vector_path'],
                          file_registry['final_dist_raster_path'],
                          args['workspace_dir']),
                    target_path_list=[file_registry['final_dist_raster_path']],
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
        avg_grid_distance = args['avg_grid_distance'] * 1000

        land_poly_dist_raster_task = task_graph.add_task(
            func=_create_distance_raster,
            args=(file_registry['harvested_masked_path'],
                  file_registry['land_poly_proj_vector_path'],
                  file_registry['land_poly_dist_raster_path'],
                  args['workspace_dir']),
            target_path_list=[file_registry['land_poly_dist_raster_path']],
            dependent_task_list=[mask_harvested_task],
            task_name='create_land_poly_dist_raster')

        final_dist_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(file_registry['land_poly_dist_raster_path'], 1),
                 (avg_grid_distance, 'raw')],
                _add_avg_dist_op, file_registry['final_dist_raster_path'],
                _TARGET_DATA_TYPE, _TARGET_NODATA),
            target_path_list=[file_registry['final_dist_raster_path']],
            task_name='calculate_final_distance_in_meters',
            dependent_task_list=[land_poly_dist_raster_task])

    # Create NPV and levelized rasters
    # Include foundation_cost, discount_rate, number_of_turbines with
    # parameters_dict to pass for NPV calculation
    for key in ['foundation_cost', 'discount_rate', 'number_of_turbines']:
        parameters_dict[key] = args[key]

    npv_levelized_task = task_graph.add_task(
        func=_calculate_npv_levelized_rasters,
        args=(file_registry['harvested_masked_path'],
              file_registry['final_dist_raster_path'],
              file_registry['npv_raster_path'],
              file_registry['levelized_raster_path'],
              parameters_dict, price_list),
        target_path_list=[file_registry['npv_raster_path'],
                          file_registry['levelized_raster_path']],
        task_name='calculate_npv_levelized_rasters',
        dependent_task_list=[final_dist_task])

    # Creating carbon offset raster
    # The amount of CO2 not released into the atmosphere, with the constant
    # conversion factor provided in the users guide by Rob Griffin
    carbon_coef = parameters_dict['carbon_coefficient']

    carbon_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(file_registry['harvested_masked_path'], 1), (carbon_coef, 'raw')],
              _calculate_carbon_op, file_registry['carbon_raster_path'],
              _TARGET_DATA_TYPE, _TARGET_NODATA),
        target_path_list=[file_registry['carbon_raster_path']],
        dependent_task_list=[mask_harvested_task],
        task_name='calculate_carbon_raster')

    # Write Valuation values to Wind Points shapefile
    LOGGER.info("Adding valuation results to shapefile")
    raster_field_to_vector_list = [
        (file_registry['harvested_masked_path'], _HARVESTED_FIELD_NAME),
        (file_registry['carbon_raster_path'], _CARBON_FIELD_NAME),
        (file_registry['levelized_raster_path'], _LEVELIZED_COST_FIELD_NAME),
        (file_registry['npv_raster_path'], _NPV_FIELD_NAME),
        (file_registry['depth_mask_path'], _DEPTH_FIELD_NAME),
        (file_registry['dist_mask_path'], _DIST_FIELD_NAME)
    ]

    task_graph.add_task(
        func=_index_raster_values_to_point_vector,
        args=(file_registry['unmasked_wind_point_vector_path'],
              raster_field_to_vector_list,
              file_registry['final_wind_point_vector_path']),
        kwargs={'mask_keys': _MASK_KEYS,
                'mask_field': _MASK_FIELD_NAME},
        target_path_list=[file_registry['final_wind_point_vector_path']],
        task_name='add_harv_valuation_to_wind_vector',
        dependent_task_list=[npv_levelized_task, carbon_task])

    task_graph.close()
    task_graph.join()
    LOGGER.info('Wind Energy Valuation Model Completed')
    return file_registry.registry


def _index_raster_values_to_point_vector(
        base_point_vector_path, raster_fieldname_list,
        target_point_vector_path, mask_keys=[], mask_field=None):
    """Add raster values to vector point feature fields.

    This function does two things:
        - Creates a copy of the base point vector and updates the copy to include
          the fields in ``raster_fieldname_list``, the values of which are
          pixel-picked from their corresponding raster. If ``mask_keys`` are
          provided, points that would be NODATA based on the raster(s)
          corresponding to fields in this list will be deleted from the copy.
        - Optionally modifies the base vector to include an additional field,
          ``mask_field``, that indicates whether or not a point was removed from
          the copy based on NODATA values in the rasters corresponding to the
          fieldnames in ``mask_keys``.

    If the base vector contains points that fall beyond the extent of the rasters,
    those points will be removed from the output vector. If ``mask_field`` is
    provided, the value will remain NULL for those points.

    Args:
        base_point_vector_path (str): a path to an OGR point vector file.
        raster_fieldname_list (list): a list of (raster_path, field_name)
            tuples. The values of rasters in this list will be added to the
            vector's features under the associated field name. All rasters must
            already be aligned.
        target_point_vector_path (str): a path to a shapefile that has the
            target field name in addition to the existing fields in the base
            point vector.
        mask_keys (list): (optional) a list of string field names that appear in
            ``raster_fieldname_list``, indicating that the associated raster
            will be used to mask out vector features. Wherever a value in a mask
            raster equals NODATA, the vector feature will be deleted from the
            target point vector.
        mask_field (str): (optional) a field name to add to the base point vector,
            where the value indicates whether or not the point was masked out in
            the target point vector, based on the values of rasters associated
            with the fields in ``mask_keys``. If ``mask_field`` is provided but
            ``mask_keys`` is not, it will be ignored.

    Returns:
        None

    """
    base_raster_path_list = [tup[0] for tup in raster_fieldname_list]
    field_name_list = [tup[1] for tup in raster_fieldname_list]

    if not set(mask_keys).issubset(set(field_name_list)):
        raise ValueError(
            f"Field name(s) in `mask_keys`: {mask_keys} do not appear in "
            f"`raster_fieldname_list`: {field_name_list}")

    # Check that raster inputs are all the same dimensions
    raster_info_list = [
        pygeoprocessing.get_raster_info(raster_path)
        for raster_path in base_raster_path_list]
    geospatial_info = [
        raster_info['raster_size'] for raster_info in raster_info_list]
    if len(set(geospatial_info)) > 1:
        mismatched_rasters = [(raster_path, dimensions) for
            (raster_path, dimensions) in
            zip(base_raster_path_list, geospatial_info)]
        raise ValueError(
            "Input Rasters are not the same dimensions. The following rasters "
            f"are not identical: {mismatched_rasters}")
    # When writing values to the vector, we replace nodata with None;
    # map the fieldnames to the nodata values of their corresponding rasters
    raster_nodata = [raster_info['nodata'][0] for raster_info in raster_info_list]
    field_nodata_map = {fieldname: nodata for fieldname, nodata in
                        zip(field_name_list, raster_nodata)}

    # Since rasters must all already be aligned, use first in list as ref
    raster_info = raster_info_list[0]

    base_vector = gdal.OpenEx(base_point_vector_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName("ESRI Shapefile")
    driver.CreateCopy(target_point_vector_path, base_vector)

    # Add the mask field to the base vector
    if mask_field:
        base_vector = gdal.OpenEx(
            base_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
        base_layer = base_vector.GetLayer()
        # Since we're mutating the base vector, if the
        # supplied fieldname already exists, don't overwrite it
        if base_layer.FindFieldIndex(mask_field, True) != -1:
            i = 1
            mask_field = f'Masked_{i}'
            while base_layer.FindFieldIndex(mask_field, True) != -1:
                i += 1
            LOGGER.warning(
                "A field with the same name as the supplied `mask_field` "
                "already exists in the base vector. To avoid overwriting, "
                f"the field name `{mask_field}` is being used instead.")
        field_defn = ogr.FieldDefn(mask_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        base_layer.CreateField(field_defn)

    target_vector = gdal.OpenEx(
        target_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_layer = target_vector.GetLayer()
    raster_gt = raster_info['geotransform']
    pixel_size_x, pixel_size_y, raster_min_x, raster_max_y = \
        abs(raster_gt[1]), abs(raster_gt[5]), raster_gt[0], raster_gt[3]

    # Create new fields for the vector attributes
    for field_name in field_name_list:
        field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)

        field_index = target_layer.FindFieldIndex(field_name, True)
        # If the field doesn't already exist, create it
        if field_index == -1:
            target_layer.CreateField(field_defn)

    # Create coordinate transformation from vector to raster, to make sure the
    # vector points are in the same projection as raster
    raster_sr = osr.SpatialReference()
    raster_sr.ImportFromWkt(raster_info['projection_wkt'])
    vector_sr = osr.SpatialReference()
    vector_sr.ImportFromWkt(
        pygeoprocessing.get_vector_info(base_point_vector_path)[
            'projection_wkt'])
    vector_coord_trans = utils.create_coordinate_transformer(
        vector_sr, raster_sr)

    # We'll check encountered features against this list at the end,
    # for the purposes of removing any points that weren't encountered
    all_fids = [feat.GetFID() for feat in target_layer]
    encountered_fids = set()

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

    iterables = [pygeoprocessing.iterblocks((base_raster_path_list[index], 1))
        for index in range(len(base_raster_path_list))]
    for block_data in zip(*iterables):
        # Rasters must all be aligned; use block data from first raster
        block_info = block_data[0][0]
        block_matrices = [block_data[index][1]
            for index in range(len(base_raster_path_list))]

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

            # Occasionally there could be points that intersect multiple block
            # bounding boxes (like points that lie on the boundary of two
            # blocks) and we don't want to double-count.
            if vector_fid in encountered_fids:
                continue

            vector_trans_x, vector_trans_y = vector.bbox[0], vector.bbox[1]

            # To get proper raster value we must index into the raster matrix
            # by getting where the point is located in terms of the matrix
            i = int((vector_trans_x - block_min_x) / pixel_size_x)
            j = int((block_max_y - vector_trans_y) / pixel_size_y)

            data = {}
            for fieldname, block_matrix in zip(field_name_list, block_matrices):
                try:
                    block_value = block_matrix[j][i]
                except IndexError:
                    # It is possible for an index to be *just* on the edge of a
                    # block and exceed the block dimensions.  If this happens,
                    # pass on this point, as another block's bounding box should
                    # catch this point instead.
                    continue
                else:
                    # Use the nodata value specific to that raster
                    if block_value == field_nodata_map[fieldname]:
                        block_value = None
                    else:
                        block_value = float(block_value)
                data[fieldname] = block_value

            encountered_fids.add(vector_fid)
            invalid_feature = False # Always valid if no mask
            if mask_keys:
                invalid_feature = None in [data.get(mask_key, None)
                                           for mask_key in mask_keys]
                if mask_field:
                    base_vector_feat = base_layer.GetFeature(vector_fid)
                    base_vector_feat.SetField(mask_field, invalid_feature)
                    base_layer.SetFeature(base_vector_feat)

            if invalid_feature:
                target_layer.DeleteFeature(vector_fid)
            else:
                target_vector_feat = target_layer.GetFeature(vector_fid)
                for fieldname, value in data.items():
                    target_vector_feat.SetField(fieldname, value)
                target_layer.SetFeature(target_vector_feat)
                feat = None

    # Finally, if any points fall outside of the raster extents, remove them
    out_of_bounds_fids = set(all_fids).difference(encountered_fids)
    if out_of_bounds_fids:
        for vector_fid in out_of_bounds_fids:
            target_layer.DeleteFeature(vector_fid)

    target_vector.ExecuteSQL('REPACK ' + target_layer.GetName())
    target_layer = None
    target_vector = None

    base_layer = None
    base_vector = None


def _reproject_bathymetry(base_raster_path, aoi_vector_path,
        comparison_pixel_size, target_raster_path):
    """Reproject and clip bathymetry raster to AOI bounding box and SRS.

    The minimum of the ``comparison_pixel_size`` and the base raster's pixel
    size will be used as the target pixel size in the ``warp_raster`` call.

    Args:
        base_raster_path (str): path to the bathymetry raster, which may not
            be projected.
        aoi_vector_path (str): path to the AOI vector, which will be used to
            clip the bathymetry raster.
        comparison_pixel_size (tuple): pixel size, in meters, to compare with
            the base raster's pixel size.
        target_raster_path (str): path to the target reprojected bathymetry
            raster.

    Returns:
        None

    """
    LOGGER.info('Clipping and projecting bathymetry to AOI')

    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    aoi_vector_info = pygeoprocessing.get_vector_info(aoi_vector_path)

    base_raster_srs = osr.SpatialReference()
    base_raster_srs.ImportFromWkt(base_raster_info['projection_wkt'])

    target_sr_wkt = aoi_vector_info['projection_wkt']
    target_bounding_box = aoi_vector_info['bounding_box']

    if base_raster_srs.IsGeographic():
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)
        aoi_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
            target_bounding_box, target_sr_wkt, wgs84_sr.ExportToWkt())

        centroid_y = (
            aoi_wgs84_bounding_box[3] + aoi_wgs84_bounding_box[1]) / 2

        # Get pixel size in square meters used for resizing the base
        # raster later on
        base_pixel_size = math.sqrt(
            pygeoprocessing.geoprocessing._m2_area_of_wg84_pixel(
            base_raster_info['pixel_size'][0], centroid_y))
    else:
        base_pixel_size = base_raster_info['pixel_size'][0]

    min_pixel_size = min(base_pixel_size, comparison_pixel_size[0])
    target_pixel_size = (min_pixel_size, -min_pixel_size)

    LOGGER.debug(f'target_sr_wkt: {target_sr_wkt}\n'
                 f'target_pixel_size: {target_pixel_size}\n'
                 f'target_bounding_box: {target_bounding_box}')

    pygeoprocessing.warp_raster(base_raster_path, target_pixel_size,
        target_raster_path, _TARGET_RESAMPLE_METHOD,
        target_bb=target_bounding_box,
        target_projection_wkt=target_sr_wkt)


def _calculate_npv_levelized_rasters(
        base_harvested_raster_path, base_dist_raster_path,
        target_npv_raster_path, target_levelized_raster_path,
        parameters_dict, price_list):
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

        parameters_dict (dict): a dictionary of the turbine and biophysical
            global parameters.

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

    # Get constants from parameters_dict to make it more readable
    # The length of infield cable in km
    infield_length = parameters_dict['infield_cable_length']
    # The cost of infield cable in currency units per km
    infield_cost = parameters_dict['infield_cable_cost']
    # The cost of the foundation in currency units
    foundation_cost = parameters_dict['foundation_cost']
    # The cost of each turbine unit in currency units
    unit_cost = parameters_dict['turbine_cost']
    # The installation cost as a decimal
    install_cost = parameters_dict['installation_cost']
    # The miscellaneous costs as a decimal factor of capex_arr
    misc_capex_cost = parameters_dict['miscellaneous_capex_cost']
    # The operations and maintenance costs as a decimal factor of capex_arr
    op_maint_cost = parameters_dict['operation_maintenance_cost']
    # The discount rate as a decimal
    discount_rate = parameters_dict['discount_rate']
    # The cost to decommission the farm as a decimal factor of capex_arr
    decom = parameters_dict['decommission_cost']
    # The mega watt value for the turbines in MW
    mega_watt = parameters_dict['turbine_rated_pwr']
    # The distance at which AC switches over to DC power
    circuit_break = parameters_dict['ac_dc_distance_break']
    # The coefficients for the AC/DC megawatt and cable cost from the CAP
    # function
    mw_coef_ac = parameters_dict['mw_coef_ac']
    mw_coef_dc = parameters_dict['mw_coef_dc']
    cable_coef_ac = parameters_dict['cable_coef_ac']
    cable_coef_dc = parameters_dict['cable_coef_dc']

    # The total mega watt capacity of the wind farm where mega watt is the
    # turbines rated power
    number_of_turbines = parameters_dict['number_of_turbines']
    total_mega_watt = mega_watt * number_of_turbines

    # Total infield cable cost
    infield_cable_cost = infield_length * infield_cost * number_of_turbines
    LOGGER.debug(f'infield_cable_cost : {infield_cable_cost}')

    # Total foundation cost
    total_foundation_cost = (foundation_cost + unit_cost) * number_of_turbines
    LOGGER.debug(f'total_foundation_cost : {total_foundation_cost}')

    # Nominal Capital Cost (CAP) minus the cost of cable which needs distances
    cap_less_dist = infield_cable_cost + total_foundation_cost
    LOGGER.debug(f'cap_less_dist : {cap_less_dist}')

    # Discount rate plus one to get that constant
    disc_const = discount_rate + 1
    LOGGER.debug(f'discount_rate : {disc_const}')

    # Discount constant raised to the total time, a constant found in the NPV
    # calculation (1+i)^T
    disc_time = disc_const**parameters_dict['time_period']
    LOGGER.debug(f'disc_time : {disc_time}')

    for (harvest_block_info, harvest_block_data), (_, dist_block_data) in zip(
            pygeoprocessing.iterblocks((base_harvested_raster_path, 1)),
            pygeoprocessing.iterblocks((base_dist_raster_path, 1))):

        target_arr_shape = harvest_block_data.shape
        target_nodata_mask = pygeoprocessing.array_equals_nodata(
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
            currency_per_kwh = price_list[year]

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
        raise KeyError(f'Unknown file extension for vector file {base_vector_path}')

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
                         ~pygeoprocessing.array_equals_nodata(bath, _TARGET_NODATA))
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
    valid_pixels_mask = ~pygeoprocessing.array_equals_nodata(tmp_dist, _TARGET_NODATA)
    out_array[valid_pixels_mask] = (
        tmp_dist[valid_pixels_mask] + avg_grid_distance)
    return out_array


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
        nodata_mask = nodata_mask | pygeoprocessing.array_equals_nodata(
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
    valid_pixels_mask = ~pygeoprocessing.array_equals_nodata(
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
    gdal.VectorTranslate(
        target_land_vector_path, base_land_vector_path,
        format=driver_name)

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
            ~pygeoprocessing.array_equals_nodata(dist_arr, raster_nodata) &
            (dist_arr >= min_dist) & (dist_arr <= max_dist))
        out_array[
            valid_pixels_mask] = dist_arr[valid_pixels_mask]
        return out_array

    pygeoprocessing.raster_calculator([(base_raster_path, 1)], _dist_mask_op,
                                      target_raster_path, _TARGET_DATA_TYPE,
                                      out_nodata)


def _create_distance_raster(base_raster_path, base_vector_path,
                            target_dist_raster_path, work_dir, where_clause=None):
    """Create and rasterize vector onto a raster, and calculate dist transform.

    Create a raster where the pixel values represent the euclidean distance to
    the vector. The distance inherits units from ``base_raster_path`` pixel
    dimensions.

    Args:
        base_raster_path (str): path to raster to create a new raster from.
        base_vector_path (str): path to vector to be rasterized.
        target_dist_raster_path (str): path to raster with distance transform.
        work_dir (str): path to create a temp folder for saving files.
        where_clause (str): If not None, is an SQL query-like string to filter
            which features are rasterized. This kwarg is passed to
            ``pygeoprocessing.rasterize``.

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
        option_list=["ALL_TOUCHED=TRUE"],
        where_clause=where_clause)

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


def _compute_density_harvested_fields(
        wind_data_path, parameters_dict, number_of_turbines,
        target_pickle_path):
    """Compute the density and harvested energy based on scale and shape keys.

    Args:
        wind_data_path (str): path to wind data input.

        parameters_dict (dict): a dictionary where the 'parameter_list'
            strings are the keys that have values pulled from bio-parameters
            CSV.

        number_of_turbines (int): an integer value for the number of machines
            for the wind farm.

        target_pickle_path (str): a path to the pickle file that has
            wind_dict_copy, a modified dictionary of wind data with additional
            fields computed from the existing fields and bio-parameters.

    Returns:
        None

    """
    # Hub Height to use for setting Weibull parameters
    hub_height = parameters_dict['hub_height']
    LOGGER.debug(f'hub_height : {hub_height}')

    # Read the wind energy data into a dictionary
    LOGGER.info('Reading in Wind Data into a dictionary')
    wind_point_df = MODEL_SPEC.get_input(
        'wind_data_path').get_validated_dataframe(wind_data_path)
    wind_point_df.columns = wind_point_df.columns.str.upper()
    # Calculate scale value at new hub height given reference values.
    # See equation 3 in users guide
    wind_point_df.rename(columns={'LAM': 'REF_LAM'}, inplace=True)
    wind_point_df['LAM'] = wind_point_df.apply(
        lambda row: row.REF_LAM * (hub_height / row.REF)**_ALPHA, axis=1)
    wind_point_df.drop(['REF'], axis=1)  # REF is not needed after calculation
    wind_dict = wind_point_df.to_dict('index')  # so keys will be 0, 1, 2, ...

    wind_dict_copy = wind_dict.copy()

    # The rated power is expressed in units of MW but the harvested energy
    # equation calls for it in terms of Wh. Thus we multiply by a million to
    # get to Wh.
    rated_power = parameters_dict['turbine_rated_pwr'] * 1000000

    # Get the rest of the inputs needed to compute harvested wind energy
    # from the dictionary so that it is in a more readable format
    exp_pwr_curve = parameters_dict['exponent_power_curve']
    air_density_standard = parameters_dict['air_density']
    v_rate = parameters_dict['rated_wspd']
    v_out = parameters_dict['cut_out_wspd']
    v_in = parameters_dict['cut_in_wspd']
    air_density_coef = parameters_dict['air_density_coefficient']
    losses = parameters_dict['loss_parameter']

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
        latitude = point_dict['lati']
        longitude = point_dict['long']

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

    LOGGER.debug(f'field_list : {field_list}')

    LOGGER.info('Creating fields for the target vector')
    for field in field_list:
        target_field = ogr.FieldDefn(field, ogr.OFTReal)
        target_layer.CreateField(target_field)

    LOGGER.info('Entering iteration to create and set the features')
    # For each inner dictionary (for each point) create a point
    for point_dict in wind_data.values():
        geom = ogr.Geometry(ogr.wkbPoint)
        latitude = point_dict['LATI']
        longitude = point_dict['LONG']
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
        # The "clip_vector_path" is always the AOI.
        raise ValueError(
            f"Clipping {base_vector_path} by {clip_vector_path} returned 0"
            f" features. This means the AOI and {base_vector_path} do not"
            " intersect spatially. Please check that the AOI has spatial"
            " overlap with all input data.")

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
    # A list to hold the individual distance transform paths in order
    land_point_dist_raster_path_list = []

    fid_field = base_point_layer.GetFIDColumn()
    if not fid_field:
        fid_field = 'FID'
    # Create a new shapefile with only one feature to burn onto a raster
    # in order to get the distance transform based on that one feature
    for feature_index, point_feature in enumerate(base_point_layer):
        # Get the point features land to grid value and add it to the list
        l2g_dist.append(float(point_feature.GetField('L2G')))

        dist_raster_path = os.path.join(temp_dir, f'dist_{feature_index}.tif')
        _create_distance_raster(
            base_raster_path, base_point_vector_path, dist_raster_path,
            work_dir, where_clause=f'{fid_field}={point_feature.GetFID()}')
        # Add each features distance transform result to list
        land_point_dist_raster_path_list.append(dist_raster_path)

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
    validation_warnings = validation.validate(args, MODEL_SPEC)
    invalid_keys = validation.get_invalid_keys(validation_warnings)
    sufficient_keys = validation.get_sufficient_keys(args)
    valid_sufficient_keys = sufficient_keys - invalid_keys

    if ('wind_schedule' in valid_sufficient_keys and
            'global_wind_parameters_path' in valid_sufficient_keys):
        year_count = utils.read_csv_to_dataframe(
            args['wind_schedule']).shape[0]
        time = MODEL_SPEC.get_input(
            'global_wind_parameters_path').get_validated_dataframe(
            args['global_wind_parameters_path']).iloc[0]['time_period']
        if year_count != time + 1:
            validation_warnings.append((
                ['wind_schedule'],
                "The 'time' argument in the Global Wind Energy Parameters "
                "file must equal the number of years provided in the price "
                "table."))
    return validation_warnings
