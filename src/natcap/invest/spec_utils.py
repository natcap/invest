import pint


# the same unit registry instance should be shared across everything
u = pint.UnitRegistry()

# Custom unit definitions #####################################################

# see https://pint.readthedocs.io/en/stable/defining.html
# pint doesn't allow multiple base units for the same dimension
# but you can define mutliple dimensionless units with []
# https://github.com/hgrecco/pint/issues/1278
u.define('currency = [value]')    # non-specific unit of value
u.define('pixel = []')            # non-specific unit of area
# used in coastal vulnerability, the DEM pixel values measure
# elevation (length) but the specific units don't matter
# and in the rec model for cell size
u.define('linear_unit = []')  # non-specific unit of length
# add "us_survey_foot" on to the aliases because it's used in some rasters
u.define('survey_foot = 1200 / 3937 * meter = sft = us_survey_foot')
# Vitamin A in the crop production nutrient table is measured in IUs
# A special unit in pharmacology that measures biologically active substances
# May be converted to weight or volume, but conversion factors are specific
# to the substance. I couldn't find a definition of its dimensionality.
u.define('international_unit = [biologic_amount] = iu = IU')
# Use u.none for unitless measurements
u.define('none = []')

# Specs for common arg types ##################################################

WORKSPACE = {
    "name": "Workspace",
    "about": (
        "The folder where all intermediate and output files of the model "
        "will be written.  If this folder does not exist, it will be "
        "created."),
    "type": "directory",
    "contents": {},
    "must_exist": False,
    "permissions": "rwx",
}

SUFFIX = {
    "name": "File suffix",
    "about": (
        'A string that will be added to the end of all files '
        'written to the workspace.'),
    "type": "freestyle_string",
    "required": False,
    "regexp": "[a-zA-Z0-9_-]*"
}

N_WORKERS = {
    "name": "Taskgraph n_workers parameter",
    "about": (
        "The n_workers parameter to provide to taskgraph. "
        "-1 will cause all jobs to run synchronously. "
        "0 will run all jobs in the same process, but scheduling will take "
        "place asynchronously. Any other positive integer will cause that "
        "many processes to be spawned to execute tasks."),
    "type": "number",
    "units": u.none,
    "required": False,
    "expression": "value >= -1"
}

AREA = {
    "type": "vector",
    "fields": {},
    "geometries": {"POLYGON", "MULTIPOLYGON"}
}

METER_RASTER = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.meter
        }
    }
}
AOI = {
    **AREA,
    "name": "area of interest",
    "about": (
        "A polygon vector containing features over which to aggregate and "
        "summarize the final results."),
}
LULC = {
    "type": "raster",
    "bands": {1: {"type": "code"}},
    "about": "Map of land use/land cover codes.",
    "name": "land use/land cover"
}
DEM = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.meter
        }
    },
    "about": "Map of elevation above sea level.",
    "name": "digital elevation model"
}
PRECIP = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.millimeter/u.year
        }
    },
    "about": "Map of average annual precipitation.",
    "name": "Precipitation"
}
ETO = {
    "name": "Evapotranspiration",
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.millimeter
        }
    },
    "about": "Map of evapotranspiration values."
}
SOIL_GROUP = {
    "type": "raster",
    "bands": {1: {"type": "code"}},
    "about": (
        "Map of soil hydrologic groups. Pixels may have values 1, 2, 3, or 4, "
        "corresponding to soil hydrologic groups A, B, C, or D, respectively."
    ),
    "name": "soil hydrologic group"
}
THRESHOLD_FLOW_ACCUMULATION = {
    "expression": "value >= 0",
    "type": "number",
    "units": u.pixel,
    "about": (
        "The number of upstream cells that must flow into a cell "
        "before it is classified as a stream."),
    "name": "Threshold Flow Accumulation Limit"
}


# geometry types ##############################################################
# the full list of ogr geometry types is in an enum in
# https://github.com/OSGeo/gdal/blob/master/gdal/ogr/ogr_core.h

POINT = {'POINT'}
LINESTRING = {'LINESTRING'}
POLYGON = {'POLYGON'}
MULTIPOINT = {'MULTIPOINT'}
MULTILINESTRING = {'MULTILINESTRING'}
MULTIPOLYGON = {'MULTIPOLYGON'}

LINES = LINESTRING | MULTILINESTRING
POLYGONS = POLYGON | MULTIPOLYGON
POINTS = POINT | MULTIPOINT
ALL_GEOMS = LINES | POLYGONS | POINTS
