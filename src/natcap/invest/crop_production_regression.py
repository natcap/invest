"""InVEST Crop Production Regression Model."""
from collections import defaultdict, namedtuple
import logging
import os
import typing

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr
from pandas import NA

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

CROP_OPTIONS = [
    spec.Option(key="barley", about=gettext("Barley")),
    spec.Option(key="maize", about=gettext("Maize")),
    spec.Option(key="oilpalm", about=gettext("Oil palm fruit")),
    spec.Option(key="potato", about=gettext("Potatoes")),
    spec.Option(key="rice", about=gettext("Rice")),
    spec.Option(key="soybean", about=gettext("Soybeans")),
    spec.Option(key="sugarbeet", about=gettext("Sugar beets")),
    spec.Option(key="sugarcane", about=gettext("Sugar cane")),
    spec.Option(key="sunflower", about=gettext("Sunflower seed")),
    spec.Option(key="wheat", about=gettext("Wheat"))
]

NUTRIENTS = [
    ("protein", "protein", u.gram/u.hectogram),
    ("lipid", "total lipid", u.gram/u.hectogram),
    ("energy", "energy", u.kilojoule/u.hectogram),
    ("ca", "calcium", u.milligram/u.hectogram),
    ("fe", "iron", u.milligram/u.hectogram),
    ("mg", "magnesium", u.milligram/u.hectogram),
    ("ph", "phosphorus", u.milligram/u.hectogram),
    ("k", "potassium", u.milligram/u.hectogram),
    ("na", "sodium", u.milligram/u.hectogram),
    ("zn", "zinc", u.milligram/u.hectogram),
    ("cu", "copper", u.milligram/u.hectogram),
    ("fl", "fluoride", u.microgram/u.hectogram),
    ("mn", "manganese", u.milligram/u.hectogram),
    ("se", "selenium", u.microgram/u.hectogram),
    ("vita", "vitamin A", u.IU/u.hectogram),
    ("betac", "beta carotene", u.microgram/u.hectogram),
    ("alphac", "alpha carotene", u.microgram/u.hectogram),
    ("vite", "vitamin E", u.milligram/u.hectogram),
    ("crypto", "cryptoxanthin", u.microgram/u.hectogram),
    ("lycopene", "lycopene", u.microgram/u.hectogram),
    ("lutein", "lutein and zeaxanthin", u.microgram/u.hectogram),
    ("betaT", "beta tocopherol", u.milligram/u.hectogram),
    ("gammaT", "gamma tocopherol", u.milligram/u.hectogram),
    ("deltaT", "delta tocopherol", u.milligram/u.hectogram),
    ("vitc", "vitamin C", u.milligram/u.hectogram),
    ("thiamin", "thiamin", u.milligram/u.hectogram),
    ("riboflavin", "riboflavin", u.milligram/u.hectogram),
    ("niacin", "niacin", u.milligram/u.hectogram),
    ("pantothenic", "pantothenic acid", u.milligram/u.hectogram),
    ("vitb6", "vitamin B6", u.milligram/u.hectogram),
    ("folate", "folate", u.microgram/u.hectogram),
    ("vitb12", "vitamin B12", u.microgram/u.hectogram),
    ("vitk", "vitamin K", u.microgram/u.hectogram)
]

NUTRIENT_UNITS = {
    "protein":     u.gram/u.hectogram,
    "lipid":       u.gram/u.hectogram,       # total lipid
    "energy":      u.kilojoule/u.hectogram,
    "ca":          u.milligram/u.hectogram,  # calcium
    "fe":          u.milligram/u.hectogram,  # iron
    "mg":          u.milligram/u.hectogram,  # magnesium
    "ph":          u.milligram/u.hectogram,  # phosphorus
    "k":           u.milligram/u.hectogram,  # potassium
    "na":          u.milligram/u.hectogram,  # sodium
    "zn":          u.milligram/u.hectogram,  # zinc
    "cu":          u.milligram/u.hectogram,  # copper
    "fl":          u.microgram/u.hectogram,  # fluoride
    "mn":          u.milligram/u.hectogram,  # manganese
    "se":          u.microgram/u.hectogram,  # selenium
    "vita":        u.IU/u.hectogram,         # vitamin A
    "betac":       u.microgram/u.hectogram,  # beta carotene
    "alphac":      u.microgram/u.hectogram,  # alpha carotene
    "vite":        u.milligram/u.hectogram,  # vitamin e
    "crypto":      u.microgram/u.hectogram,  # cryptoxanthin
    "lycopene":    u.microgram/u.hectogram,  # lycopene
    "lutein":      u.microgram/u.hectogram,  # lutein + zeaxanthin
    "betat":       u.milligram/u.hectogram,  # beta tocopherol
    "gammat":      u.milligram/u.hectogram,  # gamma tocopherol
    "deltat":      u.milligram/u.hectogram,  # delta tocopherol
    "vitc":        u.milligram/u.hectogram,  # vitamin C
    "thiamin":     u.milligram/u.hectogram,
    "riboflavin":  u.milligram/u.hectogram,
    "niacin":      u.milligram/u.hectogram,
    "pantothenic": u.milligram/u.hectogram,  # pantothenic acid
    "vitb6":       u.milligram/u.hectogram,  # vitamin B6
    "folate":      u.microgram/u.hectogram,
    "vitb12":      u.microgram/u.hectogram,  # vitamin B12
    "vitk":        u.microgram/u.hectogram,  # vitamin K
}

CropToPathTables = namedtuple(
    'CropToPathTables', ['climate_bin', 'observed_yield',
                         'percentile_yield', 'regression_yield'])
CROP_TO_PATH_TABLES = CropToPathTables(
    climate_bin='climate_bin_raster_table',
    observed_yield='observed_yield_raster_table',
    percentile_yield='percentile_yield_csv_table',
    regression_yield='regression_yield_csv_table',
)

LULC_RASTER_INPUT = spec.SingleBandRasterInput(
    id="landcover_raster_path",
    name=gettext("land use/land cover"),
    about=gettext(
        "Map of land use/land cover codes. Each land use/land cover type must"
        " be assigned a unique integer code."
    ),
    data_type=int,
    units=None,
    projected=True,
    projection_units=u.meter
)

MODEL_SPEC = spec.ModelSpec(
    model_id="crop_production_regression",
    model_title=gettext("Crop Production: Regression"),
    userguide="crop_production.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("cpr",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        [CROP_TO_PATH_TABLES.regression_yield,
         CROP_TO_PATH_TABLES.observed_yield,
         CROP_TO_PATH_TABLES.climate_bin, "crop_nutrient_table"],
        ["landcover_raster_path", "landcover_to_crop_table_path",
         "fertilization_rate_table_path", "aggregate_polygon_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        LULC_RASTER_INPUT,
        spec.CSVInput(
            id="landcover_to_crop_table_path",
            name=gettext("LULC to crop table"),
            about=gettext(
                "A table that maps each LULC code from the LULC map to one of the 10"
                " canonical crop names representing the crop grown in that LULC class."
            ),
            columns=[
                spec.IntegerInput(id="lucode", about=None),
                spec.OptionStringInput(
                    id="crop_name",
                    about=None,
                    options=CROP_OPTIONS
                )
            ],
            index_col="crop_name"
        ),
        spec.CSVInput(
            id="fertilization_rate_table_path",
            name=gettext("fertilization rate table"),
            about=gettext("A table that maps crops to fertilizer application rates."),
            columns=[
                spec.OptionStringInput(
                    id="crop_name",
                    about=gettext("One of the supported crop types."),
                    options=CROP_OPTIONS
                ),
                spec.NumberInput(
                    id="nitrogen_rate",
                    about=gettext("Rate of nitrogen application for the crop."),
                    units=u.kilogram / u.hectare
                ),
                spec.NumberInput(
                    id="phosphorus_rate",
                    about=gettext("Rate of phosphorus application for the crop."),
                    units=u.kilogram / u.hectare
                ),
                spec.NumberInput(
                    id="potassium_rate",
                    about=gettext("Rate of potassium application for the crop."),
                    units=u.kilogram / u.hectare
                )
            ],
            index_col="crop_name"
        ),
        spec.AOI.model_copy(update=dict(
            id="aggregate_polygon_path",
            required=False
        )),
        spec.CSVInput(
            id=CROP_TO_PATH_TABLES.climate_bin,
            name=gettext("Climate Bin Raster Table"),
            about=gettext(
                "A table that maps each crop name to the corresponding"
                " climate bin raster."
                " Each path may be either a relative path pointing to a local"
                " file, or a URL pointing to a remote file."
                " You do not need to create this table; it is provided for you"
                " in the sample data."
            ),
            columns=[
                spec.OptionStringInput(
                    id="crop_name",
                    about=None,
                    options=CROP_OPTIONS
                ),
                spec.SingleBandRasterInput(
                    id="path",
                    about=None,
                    data_type=int,
                    units=None,
                    projected=None
                )
            ],
            index_col="crop_name"
        ),
        spec.CSVInput(
            id=CROP_TO_PATH_TABLES.observed_yield,
            name=gettext("Observed Yield Raster Table"),
            about=gettext(
                "A table that maps each crop name to the corresponding"
                " observed yield raster."
                " Each path may be either a relative path pointing to a local"
                " file, or a URL pointing to a remote file."
                " You do not need to create this table; it is provided for you"
                " in the sample data."
            ),
            columns=[
                spec.OptionStringInput(
                    id="crop_name",
                    about=None,
                    options=CROP_OPTIONS
                ),
                spec.SingleBandRasterInput(
                    id="path",
                    about=None,
                    data_type=float,
                    units=u.metric_ton / u.hectare,
                    projected=None
                )
            ],
            index_col="crop_name"
        ),
        spec.CSVInput(
            id=CROP_TO_PATH_TABLES.regression_yield,
            name=gettext("Regression Yield CSV Table"),
            about=gettext(
                "A table that maps each crop name to the corresponding"
                " regression yield table."
                " Each path may be either a relative path pointing to a local"
                " file, or a URL pointing to a remote file."
                " You do not need to create this table; it is provided for you"
                " in the sample data."
            ),
            columns=[
                spec.OptionStringInput(
                    id="crop_name",
                    about=None,
                    options=CROP_OPTIONS
                ),
                spec.CSVInput(
                    id="path",
                    about=None,
                    columns=[
                        spec.IntegerInput(id="climate_bin", about=None),
                        spec.NumberInput(
                            id="yield_ceiling",
                            about=None,
                            units=u.metric_ton / u.hectare
                        ),
                        spec.NumberInput(id="b_nut", about=None, units=u.none),
                        spec.NumberInput(id="b_k2o", about=None, units=u.none),
                        spec.NumberInput(id="c_n", about=None, units=u.none),
                        spec.NumberInput(id="c_p2o5", about=None, units=u.none),
                        spec.NumberInput(id="c_k2o", about=None, units=u.none)
                    ],
                    index_col="climate_bin"
                )
            ],
            index_col="crop_name"
        ),
        spec.CSVInput(
            id="crop_nutrient_table",
            name=gettext("Crop Nutrient Table"),
            about=gettext(
                "A table that lists amounts of nutrients in each crop."
                " You do not need to create this table; it is provided for you"
                " in the sample data."
            ),
            columns=[
                spec.OptionStringInput(
                    id="crop_name",
                    about=None,
                    options=CROP_OPTIONS
                ),
                spec.PercentInput(
                    id="percentrefuse",
                    about=None,
                    expression="0 <= value <= 100"),
                *[spec.NumberInput(id=nutrient, units=units)
                    for nutrient, units in NUTRIENT_UNITS.items()]
            ],
            index_col="crop_name"
        ),
    ],
    outputs=[
        spec.CSVOutput(
            id="aggregate_results",
            path="aggregate_results.csv",
            about=gettext("Table of results aggregated by "),
            created_if="aggregate_polygon_path",
            columns=[
                spec.IntegerOutput(id="FID", about=gettext("FID of the AOI polygon")),
                spec.NumberOutput(
                    id="[CROP]_modeled",
                    about=gettext(
                        "Modeled production of the given crop within the polygon"
                    ),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="[CROP]_observed",
                    about=gettext(
                        "Observed production of the given crop within the polygon"
                    ),
                    units=u.metric_ton
                ),
                *[
                    spec.NumberOutput(
                        id=f"{nutrient}_{x}",
                        about=f"{x} {name} production within the polygon",
                        units=units)
                    for (nutrient, name, units) in NUTRIENTS for x in ["modeled", "observed"]
                ]
            ],
            index_col="FID"
        ),
        spec.CSVOutput(
            id="result_table",
            path="result_table.csv",
            about=gettext("Table of results aggregated by crop"),
            columns=[
                spec.StringOutput(id="crop_name", about=gettext("Name of the crop")),
                spec.NumberOutput(
                    id="area (ha)",
                    about=gettext("Area covered by the crop"),
                    units=u.hectare
                ),
                spec.NumberOutput(
                    id="production_modeled",
                    about=gettext("Modeled crop production"),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="production_observed",
                    about=gettext("Observed crop production"),
                    units=u.metric_ton
                ),
                *[
                    spec.NumberOutput(
                        id=f"{nutrient}_{x}",
                        about=f"{x} {name} production from the crop",
                        units=units
                    ) for (nutrient, name, units) in NUTRIENTS
                    for x in ["modeled", "observed"]
                ]
            ],
            index_col="crop_name"
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_observed_production",
            path="[CROP]_observed_production.tif",
            about=gettext("Observed yield for the given crop"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_regression_production",
            path="[CROP]_regression_production.tif",
            about=gettext("Modeled yield for the given crop"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.VectorOutput(
            id="aggregate_vector",
            path="intermediate_output/aggregate_vector.shp",
            about=gettext("Copy of input AOI vector"),
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="clipped_[CROP]_climate_bin_map",
            path="intermediate_output/clipped_[CROP]_climate_bin_map.tif",
            about=gettext(
                "Climate bin map for the given crop, clipped to the LULC extent"
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_[PARAMETER]_coarse_regression_parameter",
            path="intermediate_output/[CROP]_[PARAMETER]_coarse_regression_parameter.tif",
            about=gettext(
                "Regression parameter for the given crop at the coarse resolution"
                " of the climate bin map"
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_[PARAMETER]_interpolated_regression_parameter",
            path="intermediate_output/[CROP]_[PARAMETER]_interpolated_regression_parameter.tif",
            about=gettext(
                "Regression parameter for the given crop, interpolated to the"
                " resolution of the landcover map"
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_clipped_observed_yield",
            path="intermediate_output/[CROP]_clipped_observed_yield.tif",
            about=gettext(
                "Observed yield for the given crop, clipped to the extent of the"
                " landcover map"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_interpolated_observed_yield",
            path="intermediate_output/[CROP]_interpolated_observed_yield.tif",
            about=gettext(
                "Observed yield for the given crop, interpolated to the"
                " resolution of the landcover map"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_nitrogen_yield",
            path="intermediate_output/[CROP]_nitrogen_yield.tif",
            about=gettext("Nitrogen-dependent crop yield"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_phosphorus_yield",
            path="intermediate_output/[CROP]_phosphorus_yield.tif",
            about=gettext("Phosphorus-dependent crop yield"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_potassium_yield",
            path="intermediate_output/[CROP]_potassium_yield.tif",
            about=gettext("Potassium-dependent crop yield"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_zeroed_observed_yield",
            path="intermediate_output/[CROP]_zeroed_observed_yield.tif",
            about=gettext(
                "Observed yield for the given crop, with nodata converted to 0"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.TASKGRAPH_CACHE
    ]
)

_EXPECTED_REGRESSION_TABLE_HEADERS = [
    'yield_ceiling', 'b_nut', 'b_k2o', 'c_n', 'c_p2o5', 'c_k2o']

_EXPECTED_NUTRIENT_TABLE_HEADERS = [
    'protein', 'lipid', 'energy', 'ca', 'fe', 'mg', 'ph', 'k', 'na', 'zn',
    'cu', 'fl', 'mn', 'se', 'vita', 'betac', 'alphac', 'vite', 'crypto',
    'lycopene', 'lutein', 'betat', 'gammat', 'deltat', 'vitc', 'thiamin',
    'riboflavin', 'niacin', 'pantothenic', 'vitb6', 'folate', 'vitb12',
    'vitk']
_EXPECTED_LUCODE_TABLE_HEADER = 'lucode'
_NODATA_YIELD = -1


def execute(args):
    """Crop Production Regression.

    This model will take a landcover (crop cover?), N, P, and K map and
    produce modeled yields, and a nutrient table.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['landcover_raster_path'] (string): path to landcover raster
        args['landcover_to_crop_table_path'] (string): path to a table that
            converts landcover types to crop names that has two headers:

            * lucode: integer value corresponding to a landcover code in
                `args['landcover_raster_path']`.
            * crop_name: a string that must match one of the crops in
                CROP_OPTIONS. A ValueError is raised if no corresponding
                climate bin raster path is found in the Climate Bin Raster
                Table.

        args['fertilization_rate_table_path'] (string): path to CSV table
            that contains fertilization rates for the crops in the simulation,
            though it can contain additional crops not used in the simulation.
            The headers must be 'crop_name', 'nitrogen_rate',
            'phosphorus_rate', and 'potassium_rate', where 'crop_name' is the
            name string used to identify crops in the
            'landcover_to_crop_table_path', and rates are in units kg/Ha.
        args['aggregate_polygon_path'] (string): path to polygon vector
            that will be used to aggregate crop yields and total nutrient
            value. (optional, if value is None, then skipped)
        args['regression_yield_csv_table'] (string): path to a table that maps
            each crop name to a path to its corresponding regression yield
            table.
        args['climate_bin_raster_table'] (string): path to a table that maps
            each crop name to a path to its corresponding climate bin raster.
        args['observed_yield_raster_table'] (string): path to a table that maps
            each crop name to a path to its corresponding observed yield
            raster.
        args['crop_nutrient_table'] (string): path to a table that lists
            amounts of nutrients in each crop.
    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    dependent_task_list = []

    LOGGER.info(
        "Checking if the landcover raster is missing lucodes")

    # It might seem backwards to read the landcover_to_crop_table into a
    # DataFrame called crop_to_landcover_df, but since the table is indexed
    # by crop_name, it makes sense for the code to treat it as a mapping from
    # crop name to LULC code.
    crop_to_landcover_df = MODEL_SPEC.get_input(
        'landcover_to_crop_table_path').get_validated_dataframe(
        args['landcover_to_crop_table_path'])

    crop_to_fertilization_rate_df = MODEL_SPEC.get_input(
        'fertilization_rate_table_path').get_validated_dataframe(
        args['fertilization_rate_table_path'])

    lucodes_in_table = set(list(
        crop_to_landcover_df[_EXPECTED_LUCODE_TABLE_HEADER]))

    def update_unique_lucodes_in_raster(unique_codes, block):
        unique_codes.update(numpy.unique(block))
        return unique_codes

    unique_lucodes_in_raster = pygeoprocessing.raster_reduce(
        update_unique_lucodes_in_raster,
        (args['landcover_raster_path'], 1),
        set())

    lucodes_missing_from_raster = lucodes_in_table.difference(
        unique_lucodes_in_raster)
    if lucodes_missing_from_raster:
        LOGGER.warning(
            "The following lucodes are in the landcover to crop table but "
            f"aren't in the landcover raster: {lucodes_missing_from_raster}")

    lucodes_missing_from_table = unique_lucodes_in_raster.difference(
        lucodes_in_table)
    if lucodes_missing_from_table:
        LOGGER.warning(
            "The following lucodes are in the landcover raster but aren't "
            f"in the landcover to crop table: {lucodes_missing_from_table}")

    LOGGER.info("Checking that crops are supported by the model.")
    user_provided_crop_names = set(list(crop_to_landcover_df.index))
    valid_crop_names = set([crop.key for crop in CROP_OPTIONS])
    invalid_crop_names = user_provided_crop_names.difference(valid_crop_names)
    if invalid_crop_names:
        raise ValueError(
            "The following crop names were provided in "
            f"{args['landcover_to_crop_table_path']} but are not supported "
            f"by the model: {invalid_crop_names}")

    landcover_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    pixel_area_ha = numpy.prod([
        abs(x) for x in landcover_raster_info['pixel_size']]) / 10000
    landcover_nodata = landcover_raster_info['nodata'][0]
    if landcover_nodata is None:
        LOGGER.warning(
            "%s does not have nodata value defined; "
            "assuming all pixel values are valid"
            % args['landcover_raster_path'])

    # Calculate lat/lng bounding box for landcover map
    wgs84srs = osr.SpatialReference()
    wgs84srs.ImportFromEPSG(4326)  # EPSG4326 is WGS84 lat/lng
    landcover_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
        landcover_raster_info['bounding_box'],
        landcover_raster_info['projection_wkt'], wgs84srs.ExportToWkt(),
        edge_samples=11)

    crop_lucode = None
    observed_yield_nodata = None

    for crop_name, row in crop_to_landcover_df.iterrows():
        crop_lucode = row[_EXPECTED_LUCODE_TABLE_HEADER]
        LOGGER.info(f'Processing crop {crop_name}')
        crop_climate_bin_raster_path = get_full_path_from_crop_table(
            MODEL_SPEC,
            CROP_TO_PATH_TABLES.climate_bin,
            args[CROP_TO_PATH_TABLES.climate_bin],
            crop_name)

        if not crop_climate_bin_raster_path:
            raise ValueError(
                f'No climate bin raster path could be found for {crop_name}')

        # Use file_registry for clipped climate bin raster path
        crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
            crop_climate_bin_raster_path)
        crop_climate_bin_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(crop_climate_bin_raster_path,
                  crop_climate_bin_raster_info['pixel_size'],
                  file_registry['clipped_[CROP]_climate_bin_map', crop_name],
                  'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[file_registry['clipped_[CROP]_climate_bin_map', crop_name]],
            task_name='crop_climate_bin')
        dependent_task_list.append(crop_climate_bin_task)

        climate_regression_yield_table_path = get_full_path_from_crop_table(
            MODEL_SPEC,
            CROP_TO_PATH_TABLES.regression_yield,
            args[CROP_TO_PATH_TABLES.regression_yield],
            crop_name)

        crop_regression_df = MODEL_SPEC.get_input(
            CROP_TO_PATH_TABLES.regression_yield).get_column(
                'path').get_validated_dataframe(
                    climate_regression_yield_table_path)
        for _, row in crop_regression_df.iterrows():
            for header in _EXPECTED_REGRESSION_TABLE_HEADERS:
                if numpy.isnan(row[header]):
                    row[header] = 0

        yield_regression_headers = [
            x for x in crop_regression_df.columns if x != 'climate_bin']

        reclassify_error_details = {
            'raster_name': f'{crop_name} Climate Bin',
            'column_name': 'climate_bin',
            'table_name': f'Climate {crop_name} Regression Yield'}
        for yield_regression_id in yield_regression_headers:
            # there are extra headers in that table
            if yield_regression_id not in _EXPECTED_REGRESSION_TABLE_HEADERS:
                continue
            LOGGER.info("Map %s to climate bins.", yield_regression_id)
            bin_to_regression_value = crop_regression_df[yield_regression_id].to_dict()
            # reclassify nodata to a valid value of 0
            # we're assuming that the crop doesn't exist where there is no data
            # this is more likely than assuming the crop does exist, esp.
            # in the context of the provided climate bins map
            bin_to_regression_value[
                crop_climate_bin_raster_info['nodata'][0]] = 0
            create_coarse_regression_parameter_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=((file_registry['clipped_[CROP]_climate_bin_map', crop_name], 1),
                      bin_to_regression_value,
                      file_registry['[CROP]_[PARAMETER]_coarse_regression_parameter',
                        crop_name, yield_regression_id],
                      gdal.GDT_Float32, _NODATA_YIELD,
                      reclassify_error_details),
                target_path_list=[file_registry['[CROP]_[PARAMETER]_coarse_regression_parameter',
                    crop_name, yield_regression_id]],
                dependent_task_list=[crop_climate_bin_task],
                task_name='create_coarse_regression_parameter_%s_%s' % (
                    crop_name, yield_regression_id))
            dependent_task_list.append(create_coarse_regression_parameter_task)

            LOGGER.info(
                "Interpolate %s %s parameter to landcover resolution.",
                crop_name, yield_regression_id)
            create_interpolated_parameter_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(file_registry['[CROP]_[PARAMETER]_coarse_regression_parameter',
                        crop_name, yield_regression_id],
                      landcover_raster_info['pixel_size'],
                      file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, yield_regression_id],
                      'cubicspline'),
                kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                        'target_bb': landcover_raster_info['bounding_box']},
                target_path_list=[file_registry[
                    '[CROP]_[PARAMETER]_interpolated_regression_parameter',
                    crop_name, yield_regression_id]],
                dependent_task_list=[
                    create_coarse_regression_parameter_task],
                task_name='create_interpolated_parameter_%s_%s' % (
                    crop_name, yield_regression_id))
            dependent_task_list.append(create_interpolated_parameter_task)

        LOGGER.info('Calc nitrogen yield')
        calc_nitrogen_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'yield_ceiling'], 1),
                   (file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'b_nut'], 1),
                   (file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'c_n'], 1),
                   (args['landcover_raster_path'], 1),
                   (crop_to_fertilization_rate_df['nitrogen_rate'][crop_name],
                    'raw'), (crop_lucode, 'raw')],
                  _x_yield_op,
                  file_registry['[CROP]_nitrogen_yield', crop_name],
                  gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[file_registry['[CROP]_nitrogen_yield', crop_name]],
            dependent_task_list=dependent_task_list,
            task_name='calculate_nitrogen_yield_%s' % crop_name)

        LOGGER.info('Calc phosphorus yield')
        calc_phosphorus_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'yield_ceiling'], 1),
                   (file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'b_nut'], 1),
                   (file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'c_p2o5'], 1),
                   (args['landcover_raster_path'], 1),
                   (crop_to_fertilization_rate_df['phosphorus_rate'][crop_name],
                    'raw'), (crop_lucode, 'raw')],
                  _x_yield_op,
                  file_registry['[CROP]_phosphorus_yield', crop_name],
                  gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[file_registry['[CROP]_phosphorus_yield', crop_name]],
            dependent_task_list=dependent_task_list,
            task_name='calculate_phosphorus_yield_%s' % crop_name)

        LOGGER.info('Calc potassium yield')
        calc_potassium_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'yield_ceiling'], 1),
                   (file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'b_k2o'], 1),
                   (file_registry['[CROP]_[PARAMETER]_interpolated_regression_parameter',
                        crop_name, 'c_k2o'], 1),
                   (args['landcover_raster_path'], 1),
                   (crop_to_fertilization_rate_df['potassium_rate'][crop_name],
                    'raw'), (crop_lucode, 'raw')],
                  _x_yield_op,
                  file_registry['[CROP]_potassium_yield', crop_name],
                  gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[file_registry['[CROP]_potassium_yield', crop_name]],
            dependent_task_list=dependent_task_list,
            task_name='calculate_potassium_yield_%s' % crop_name)

        dependent_task_list.extend((
            calc_nitrogen_yield_task,
            calc_phosphorus_yield_task,
            calc_potassium_yield_task))

        LOGGER.info('Calc the min of N, K, and P')
        calc_min_NKP_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_min_op,
                rasters=[file_registry['[CROP]_nitrogen_yield', crop_name],
                         file_registry['[CROP]_phosphorus_yield', crop_name],
                         file_registry['[CROP]_potassium_yield', crop_name]],
                target_path=file_registry['[CROP]_regression_production', crop_name],
                target_nodata=_NODATA_YIELD),
            target_path_list=[file_registry['[CROP]_regression_production', crop_name]],
            dependent_task_list=dependent_task_list,
            task_name='calc_min_of_NKP')
        dependent_task_list.append(calc_min_NKP_task)

        LOGGER.info(f'Calculate observed yield for {crop_name}')
        global_observed_yield_raster_path = get_full_path_from_crop_table(
            MODEL_SPEC,
            CROP_TO_PATH_TABLES.observed_yield,
            args[CROP_TO_PATH_TABLES.observed_yield],
            crop_name)
        global_observed_yield_raster_info = (
            pygeoprocessing.get_raster_info(
                global_observed_yield_raster_path))
        clip_global_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(global_observed_yield_raster_path,
                  global_observed_yield_raster_info['pixel_size'],
                  file_registry['[CROP]_clipped_observed_yield', crop_name],
                  'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[file_registry['[CROP]_clipped_observed_yield', crop_name]],
            task_name='clip_global_observed_yield_%s_' % crop_name)
        dependent_task_list.append(clip_global_observed_yield_task)

        observed_yield_nodata = (
            global_observed_yield_raster_info['nodata'][0])

        nodata_to_zero_for_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(file_registry['[CROP]_clipped_observed_yield', crop_name], 1),
                   (observed_yield_nodata, 'raw')],
                  _zero_observed_yield_op,
                  file_registry['[CROP]_zeroed_observed_yield', crop_name],
                  gdal.GDT_Float32, observed_yield_nodata),
            target_path_list=[file_registry['[CROP]_zeroed_observed_yield', crop_name]],
            dependent_task_list=[clip_global_observed_yield_task],
            task_name='nodata_to_zero_for_observed_yield_%s_' % crop_name)
        dependent_task_list.append(nodata_to_zero_for_observed_yield_task)

        LOGGER.info(
            "Interpolating observed %s raster to landcover.", crop_name)
        interpolate_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(file_registry['[CROP]_zeroed_observed_yield', crop_name],
                  landcover_raster_info['pixel_size'],
                  file_registry['[CROP]_interpolated_observed_yield', crop_name],
                  'cubicspline'),
            kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                    'target_bb': landcover_raster_info['bounding_box']},
            target_path_list=[file_registry['[CROP]_interpolated_observed_yield', crop_name]],
            dependent_task_list=[nodata_to_zero_for_observed_yield_task],
            task_name='interpolate_observed_yield_to_lulc_%s' % crop_name)
        dependent_task_list.append(interpolate_observed_yield_task)

        calculate_observed_production_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(args['landcover_raster_path'], 1),
                   (file_registry['[CROP]_interpolated_observed_yield', crop_name], 1),
                   (observed_yield_nodata, 'raw'), (landcover_nodata, 'raw'),
                   (crop_lucode, 'raw')],
                  _mask_observed_yield_op,
                  file_registry['[CROP]_observed_production', crop_name],
                  gdal.GDT_Float32, observed_yield_nodata),
            target_path_list=[file_registry['[CROP]_observed_production', crop_name]],
            dependent_task_list=[interpolate_observed_yield_task],
            task_name='calculate_observed_production_%s' % crop_name)
        dependent_task_list.append(calculate_observed_production_task)

    nutrient_gdal_path = utils._GDALPath.from_uri(args['crop_nutrient_table'])
    if nutrient_gdal_path.is_local:
        nutrient_table_path = os.path.join(args['crop_nutrient_table'])
    else:
        nutrient_table_path = nutrient_gdal_path.to_normalized_path()

    nutrient_df = MODEL_SPEC.get_input(
        'crop_nutrient_table').get_validated_dataframe(nutrient_table_path)

    LOGGER.info("Generating report table")
    crop_names = list(crop_to_landcover_df.index)
    _ = task_graph.add_task(
        func=tabulate_regression_results,
        args=(nutrient_df,
              crop_names, pixel_area_ha,
              args['landcover_raster_path'], landcover_nodata,
              file_registry, file_registry['result_table']),
        target_path_list=[file_registry['result_table']],
        dependent_task_list=dependent_task_list,
        task_name='tabulate_results')

    if args['aggregate_polygon_path']:
        LOGGER.info("aggregating result over query polygon")
        _ = task_graph.add_task(
            func=aggregate_regression_results_to_polygons,
            args=(args['aggregate_polygon_path'],
                  file_registry['aggregate_vector'],
                  file_registry['aggregate_results'],
                  landcover_raster_info['projection_wkt'],
                  crop_names, nutrient_df, pixel_area_ha,
                  file_registry),
            target_path_list=[file_registry['aggregate_vector'],
                              file_registry['aggregate_results']],
            dependent_task_list=dependent_task_list,
            task_name='aggregate_results_to_polygons')

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def _x_yield_op(
        y_max, b_x, c_x, lulc_array, fert_rate, crop_lucode):
    """Calc generalized yield op, Ymax*(1-b_NP*exp(-cN * N_GC)).

    The regression model has identical mathematical equations for
    the nitrogen, phosphorus, and potassium. The only difference is
    the scalar in the equation (fertilization rate).
    """
    result = numpy.empty(b_x.shape, dtype=numpy.float32)
    result[:] = _NODATA_YIELD
    valid_mask = (
        ~pygeoprocessing.array_equals_nodata(y_max, _NODATA_YIELD) &
        ~pygeoprocessing.array_equals_nodata(b_x, _NODATA_YIELD) &
        ~pygeoprocessing.array_equals_nodata(c_x, _NODATA_YIELD) &
        (lulc_array == crop_lucode))
    result[valid_mask] = y_max[valid_mask] * (
        1 - b_x[valid_mask] * numpy.exp(
            -c_x[valid_mask] * fert_rate))

    return result


"""equation for raster_map: calculate min of inputs and multiply by Ymax."""
def _min_op(y_n, y_p, y_k): return numpy.min([y_n, y_k, y_p], axis=0)


def _zero_observed_yield_op(observed_yield_array, observed_yield_nodata):
    """Reclassify observed_yield nodata to zero.

    Args:
        observed_yield_array (numpy.ndarray): raster values
        observed_yield_nodata (float): raster nodata value

    Returns:
        numpy.ndarray with observed yield values

    """
    result = numpy.empty(
        observed_yield_array.shape, dtype=numpy.float32)
    result[:] = 0
    valid_mask = slice(None)
    if observed_yield_nodata is not None:
        valid_mask = ~pygeoprocessing.array_equals_nodata(
            observed_yield_array, observed_yield_nodata)
    result[valid_mask] = observed_yield_array[valid_mask]
    return result


def _mask_observed_yield_op(
        lulc_array, observed_yield_array, observed_yield_nodata,
        landcover_nodata, crop_lucode):
    """Mask total observed yield to crop lulc type.

    Args:
        lulc_array (numpy.ndarray): landcover raster values
        observed_yield_array (numpy.ndarray): yield raster values
        observed_yield_nodata (float): yield raster nodata value
        landcover_nodata (float): landcover raster nodata value
        crop_lucode (int): code used to mask in the current crop

    Returns:
        numpy.ndarray with float values of yields masked to crop_lucode

    """
    result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
    if landcover_nodata is not None:
        result[:] = observed_yield_nodata
        valid_mask = ~pygeoprocessing.array_equals_nodata(
            lulc_array, landcover_nodata)
        result[valid_mask] = 0
    else:
        result[:] = 0
    lulc_mask = lulc_array == crop_lucode
    result[lulc_mask] = observed_yield_array[lulc_mask]
    return result


def tabulate_regression_results(
        nutrient_df, crop_names, pixel_area_ha, landcover_raster_path,
        landcover_nodata, file_registry, target_table_path):
    """Write table with total yield and nutrient results by crop.

    This function includes all the operations that write to results_table.csv.

    Args:
        nutrient_df (pandas.DataFrame): a table of nutrient values by crop
        crop_names (list): list of crop names
        pixel_area_ha (float): area of lulc raster cells (hectares)
        landcover_raster_path (string): path to landcover raster
        landcover_nodata (float): landcover raster nodata value
        file_registry (FileRegistry): used to look up target file paths
        target_table_path (string): path to 'result_table.csv' in the output
            workspace

    Returns:
        None

    """
    nutrient_headers = [
        nutrient_id + '_' + mode
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
        for mode in ['modeled', 'observed']]

    # Since pixel values in observed and percentile rasters are Mg/(ha•yr),
    # raster sums are (Mg•px)/(ha•yr). Before recording sums in
    # production_lookup dictionary, convert to Mg/yr by multiplying by ha/px.

    with open(target_table_path, 'w') as result_table:
        result_table.write(
            'crop_name,area (ha),' + 'production_observed,production_modeled,' +
            ','.join(nutrient_headers) + '\n')
        for crop_name in sorted(crop_names):
            result_table.write(crop_name)
            production_lookup = {}
            production_pixel_count = 0
            yield_sum = 0

            LOGGER.info(
                "Calculating production area and summing observed yield.")
            observed_yield_nodata = pygeoprocessing.get_raster_info(
                file_registry['[CROP]_observed_production', crop_name])['nodata'][0]
            for _, yield_block in pygeoprocessing.iterblocks(
                    (file_registry['[CROP]_observed_production', crop_name], 1)):

                # make a valid mask showing which pixels are not nodata
                # if nodata value undefined, assume all pixels are valid
                valid_mask = numpy.full(yield_block.shape, True)
                if observed_yield_nodata is not None:
                    valid_mask = ~pygeoprocessing.array_equals_nodata(
                        yield_block, observed_yield_nodata)
                production_pixel_count += numpy.count_nonzero(
                    valid_mask & (yield_block > 0.0))
                yield_sum += numpy.sum(yield_block[valid_mask])
            yield_sum *= pixel_area_ha
            production_area = production_pixel_count * pixel_area_ha
            production_lookup['observed'] = yield_sum
            result_table.write(',%f' % production_area)
            result_table.write(",%f" % yield_sum)

            yield_sum = 0
            for _, yield_block in pygeoprocessing.iterblocks(
                    (file_registry['[CROP]_regression_production', crop_name], 1)):
                yield_sum += numpy.sum(
                    # _NODATA_YIELD will always have a value (defined above)
                    yield_block[~pygeoprocessing.array_equals_nodata(
                        yield_block, _NODATA_YIELD)])
            yield_sum *= pixel_area_ha
            production_lookup['modeled'] = yield_sum
            result_table.write(",%f" % yield_sum)

            # convert 100g to Mg and fraction left over from refuse
            nutrient_factor = 1e4 * (
                1 - nutrient_df['percentrefuse'][crop_name] / 100)
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                total_nutrient = (
                    nutrient_factor *
                    production_lookup['modeled'] *
                    nutrient_df[nutrient_id][crop_name])
                result_table.write(",%f" % (total_nutrient))
                result_table.write(
                    ",%f" % (
                        nutrient_factor *
                        production_lookup['observed'] *
                        nutrient_df[nutrient_id][crop_name]))
            result_table.write('\n')

        total_area = 0
        for _, band_values in pygeoprocessing.iterblocks(
                (landcover_raster_path, 1)):
            if landcover_nodata is not None:
                total_area += numpy.count_nonzero(
                    ~pygeoprocessing.array_equals_nodata(band_values, landcover_nodata))
            else:
                total_area += band_values.size
        result_table.write(
            '\n,total area (both crop and non-crop)\n,%f\n' % (
                total_area * pixel_area_ha))


def aggregate_regression_results_to_polygons(
        base_aggregate_vector_path, target_aggregate_vector_path,
        aggregate_results_table_path, landcover_raster_projection, crop_names,
        nutrient_df, pixel_area_ha, file_registry):
    """Write table with aggregate results of yield and nutrient values.

    Use zonal statistics to summarize total observed and interpolated
    production and nutrient information for each polygon in
    base_aggregate_vector_path.

    Args:
        base_aggregate_vector_path (string): path to polygon vector
        target_aggregate_vector_path (string): path to re-projected copy of
            polygon vector
        aggregate_results_table_path (string): path to CSV file where aggregate
            results will be reported.
        landcover_raster_projection (string): a WKT projection string
        crop_names (list): list of crop names
        nutrient_df (pandas.DataFrame): a table of nutrient values by crop
        pixel_area_ha (float): area of lulc raster cells (hectares)
        file_registry (FileRegistry): used to look up target file paths

    Returns:
        None

    """
    pygeoprocessing.reproject_vector(
        base_aggregate_vector_path,
        landcover_raster_projection,
        target_aggregate_vector_path,
        driver_name='ESRI Shapefile')

    # Since pixel values are Mg/(ha•yr), zonal stats sum is (Mg•px)/(ha•yr).
    # Before writing sum to results tables or when using sum to calculate
    # nutrient yields, convert to Mg/yr by multiplying by ha/px.

    # loop over every crop and query with pgp function
    total_yield_lookup = {}
    total_nutrient_table = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    for crop_name in crop_names:
        # convert 100g to Mg and fraction left over from refuse
        nutrient_factor = 1e4 * (
            1 - nutrient_df['percentrefuse'][crop_name] / 100)
        LOGGER.info(
            "Calculating zonal stats for %s", crop_name)
        total_yield_lookup['%s_modeled' % crop_name] = (
            pygeoprocessing.zonal_statistics(
                (file_registry['[CROP]_regression_production', crop_name], 1),
                target_aggregate_vector_path))

        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
            for fid_index in total_yield_lookup['%s_modeled' % crop_name]:
                total_nutrient_table[nutrient_id][
                    'modeled'][fid_index] += (
                        nutrient_factor
                        * total_yield_lookup['%s_modeled' % crop_name][
                            fid_index]['sum']
                        * pixel_area_ha
                        * nutrient_df[nutrient_id][crop_name])

        # process observed
        total_yield_lookup['%s_observed' % crop_name] = (
            pygeoprocessing.zonal_statistics(
                (file_registry['[CROP]_observed_production', crop_name], 1),
                target_aggregate_vector_path))
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
            for fid_index in total_yield_lookup[
                    '%s_observed' % crop_name]:
                total_nutrient_table[
                    nutrient_id]['observed'][fid_index] += (
                        nutrient_factor  # percent crop used * 1000 [100g per Mg]
                        * total_yield_lookup[
                            '%s_observed' % crop_name][fid_index]['sum']
                        * pixel_area_ha
                        * nutrient_df[nutrient_id][crop_name])  # nutrient unit per 100g crop

    # report everything to a table
    with open(aggregate_results_table_path, 'w') as aggregate_table:
        # write header
        aggregate_table.write('FID,')
        aggregate_table.write(','.join(sorted(total_yield_lookup)) + ',')
        aggregate_table.write(
            ','.join([
                '%s_%s' % (nutrient_id, model_type)
                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
                for model_type in sorted(
                    list(total_nutrient_table.values())[0])]))
        aggregate_table.write('\n')

        # iterate by polygon index
        for id_index in list(total_yield_lookup.values())[0]:
            aggregate_table.write('%s,' % id_index)
            aggregate_table.write(','.join([
                str(total_yield_lookup[yield_header][id_index]['sum']
                    * pixel_area_ha)
                for yield_header in sorted(total_yield_lookup)]))

            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for model_type in sorted(
                        list(total_nutrient_table.values())[0]):
                    aggregate_table.write(
                        ',%s' % total_nutrient_table[
                            nutrient_id][model_type][id_index])
            aggregate_table.write('\n')


def get_full_path_from_crop_table(
        model_spec: spec.ModelSpec, table_id: str, table_path: str,
        crop_name: str) -> typing.Union[str, None]:
    """Given a crop-to-path table, look up a path and expand it if appropriate.

    Args:
        table_id (str): the id of the table as defined in the model spec.
            One of ``CROP_TO_PATH_TABLES``.
        table_path (str): the path to the table as defined in the model args.
        crop_name (str): the name of the crop to look up in the table.
            One of ``CROP_OPTIONS``.

    Returns:
        One of the following:
            The full path (str), as an absolute path if it's local, or
                normalized if it's remote.
            ``None`` if ``crop_name`` is not in the table, or if the path
                found in the table is empty or not a string.

    Raises:
        ``KeyError`` if ``table_id`` is not one of ``CROP_TO_PATH_TABLES``.
    """
    if table_id not in CROP_TO_PATH_TABLES:
        raise KeyError(f'table_id {table_id} is not valid')
    df = model_spec.get_input(table_id).get_validated_dataframe(table_path)
    try:
        path_str = df.at[crop_name, 'path']
    except KeyError:
        return None
    if (path_str is NA) or (not path_str) or (type(path_str) is not str):
        return None
    gdal_path = utils._GDALPath.from_uri(path_str)
    if gdal_path.is_local:
        return utils.expand_path(path_str, table_path)
    else:
        return gdal_path.to_normalized_path()


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    return validation.validate(args, MODEL_SPEC)
