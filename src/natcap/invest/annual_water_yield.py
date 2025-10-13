"""InVEST Annual Water Yield model."""
import logging
import math
import os
import pickle

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import ogr

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

BASE_OUTPUT_FIELDS = [
    spec.NumberOutput(
        id="precip_mn",
        about=gettext(
            "Mean precipitation per pixel in the"
            " subwatershed."
        ),
        units=u.millimeter
    ),
    spec.NumberOutput(
        id="PET_mn",
        about=gettext(
            "Mean potential evapotranspiration per pixel in"
            " the subwatershed."
        ),
        units=u.millimeter
    ),
    spec.NumberOutput(
        id="AET_mn",
        about=gettext(
            "Mean actual evapotranspiration per pixel in"
            " the subwatershed."
        ),
        units=u.millimeter
    ),
    spec.NumberOutput(
        id="wyield_mn",
        about=gettext(
            "Mean water yield per pixel in the"
            " subwatershed."
        ),
        units=u.millimeter
    ),
    spec.NumberOutput(
        id="wyield_vol",
        about=gettext(
            "Total volume of water yield in the"
            " subwatershed."
        ),
        units=u.meter ** 3
    )
]

SCARCITY_OUTPUT_FIELDS = [
    spec.NumberOutput(
        id="consum_vol",
        about=gettext(
            "Total water consumption for each watershed."
        ),
        created_if="demand_table_path",
        units=u.meter ** 3
    ),
    spec.NumberOutput(
        id="consum_mn",
        about=gettext(
            "Mean water consumptive volume per pixel per"
            " watershed."
        ),
        created_if="demand_table_path",
        units=u.meter ** 3 / u.hectare
    ),
    spec.NumberOutput(
        id="rsupply_vl",
        about=gettext(
            "Total realized water supply (water yield –"
            " consumption) volume for each watershed."
        ),
        created_if="demand_table_path",
        units=u.meter ** 3
    ),
    spec.NumberOutput(
        id="rsupply_mn",
        about=gettext(
            "Mean realized water supply (water yield –"
            " consumption) volume per pixel per watershed."
        ),
        created_if="demand_table_path",
        units=u.meter ** 3 / u.hectare
    )
]

SUBWATERSHED_OUTPUT_FIELDS = [
    spec.IntegerOutput(
        id="subws_id",
        about=gettext("Unique identifier for each subwatershed.")
    ),
    *BASE_OUTPUT_FIELDS,
    *SCARCITY_OUTPUT_FIELDS
]

WATERSHED_OUTPUT_FIELDS = [
    spec.IntegerOutput(
        id="ws_id",
        about=gettext("Unique identifier for each watershed.")
    ),
    *BASE_OUTPUT_FIELDS,
    *SCARCITY_OUTPUT_FIELDS,
    spec.NumberOutput(
        id="hp_energy",
        about=gettext(
            "The amount of ecosystem service in energy"
            " production terms. This is the amount of"
            " energy produced annually by the hydropower"
            " station that can be attributed to each"
            " watershed based on the watershed’s water"
            " yield contribution."
        ),
        created_if="valuation_table_path",
        units=u.kilowatt_hour
    ),
    spec.NumberOutput(
        id="hp_val",
        about=gettext(
            "The amount of ecosystem service in economic"
            " terms. This shows the value of the landscape"
            " per watershed according to its ability to"
            " yield water for hydropower production over"
            " the specified timespan, and with respect to"
            " the discount rate."
        ),
        created_if="valuation_table_path",
        units=u.currency
    )
]

MODEL_SPEC = spec.ModelSpec(
    model_id="annual_water_yield",
    model_title=gettext("Annual Water Yield"),
    userguide="annual_water_yield.html",
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["precipitation_path", "eto_path", "depth_to_root_rest_layer_path", "pawc_path"],
        ["lulc_path", "biophysical_table_path", "seasonality_constant"],
        ["watersheds_path", "sub_watersheds_path"],
        ["demand_table_path", "valuation_table_path"]
    ],
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=("hwy", "awy"),
    module_name=__name__,
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="lulc_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover"
                " type must be assigned a unique integer code. All values in"
                " this raster must have corresponding entries in the"
                " Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="depth_to_root_rest_layer_path",
            name=gettext("root restricting layer depth"),
            about=gettext(
                "Map of root restricting layer depth, the soil depth at which"
                " root penetration is strongly inhibited because of physical or"
                " chemical characteristics."
            ),
            data_type=float,
            units=u.millimeter,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="precipitation_path",
            name=gettext("precipitation"),
            about=gettext("Map of average annual precipitation."),
            data_type=float,
            units=u.millimeter / u.year,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="pawc_path",
            name=gettext("plant available water content"),
            about=gettext(
                "Map of plant available water content, the fraction of water"
                " that can be stored in the soil profile that is available to"
                " plants."
            ),
            data_type=float,
            units=None,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="eto_path",
            projected=True,
            name=gettext("reference evapotranspiration"),
            about=gettext("Map of reference evapotranspiration values."),
            data_type=float,
            units=u.millimeter
        ),
        spec.VectorInput(
            id="watersheds_path",
            name=gettext("watersheds"),
            about=gettext(
                "Map of watershed boundaries, such that each watershed drains"
                " to a point of interest where hydropower production will be"
                " analyzed."
            ),
            geometry_types={"POLYGON"},
            fields=[
                spec.IntegerInput(
                    id="ws_id",
                    about=gettext("Unique identifier for each watershed.")
                )
            ],
            projected=True
        ),
        spec.VectorInput(
            id="sub_watersheds_path",
            name=gettext("sub-watersheds"),
            about=gettext(
                "Map of subwatershed boundaries within each watershed in the"
                " Watersheds map."
            ),
            required=False,
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.IntegerInput(
                    id="subws_id",
                    about=gettext("Unique identifier for each subwatershed.")
                )
            ],
            projected=True
        ),
        spec.CSVInput(
            id="biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "Table of biophysical parameters for each LULC class. All"
                " values in the LULC raster must have corresponding entries in"
                " this table."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.IntegerInput(
                    id="lulc_veg",
                    about=gettext(
                        "Code indicating whether the the LULC class is"
                        " vegetated for the purpose of AET. Enter 1 for all"
                        " vegetated classes except wetlands, and 0 for all"
                        " other classes, including wetlands, urban areas, water"
                        " bodies, etc."
                    )
                ),
                spec.NumberInput(
                    id="root_depth",
                    about=gettext(
                        "Maximum root depth for plants in this LULC class. Only"
                        " used for classes with a 'lulc_veg' value of 1."
                    ),
                    units=u.millimeter
                ),
                spec.NumberInput(
                    id="kc",
                    about=gettext("Crop coefficient for this LULC class."),
                    units=u.none
                )
            ],
            index_col="lucode"
        ),
        spec.NumberInput(
            id="seasonality_constant",
            name=gettext("z parameter"),
            about=gettext(
                "The seasonality factor, representing hydrogeological"
                " characterisitics and the seasonal distribution of"
                " precipitation. Values typically range from 1 - 30."
            ),
            units=u.none,
            expression="value > 0"
        ),
        spec.CSVInput(
            id="demand_table_path",
            name=gettext("water demand table"),
            about=gettext(
                "A table of water demand for each LULC class. Each LULC code in"
                " the LULC raster must have a corresponding row in this table. "
                " Required if 'valuation_table_path' is provided."
            ),
            required="valuation_table_path",
            columns=[
                spec.IntegerInput(
                    id="lucode",
                    about=gettext("LULC code corresponding to the LULC raster")
                ),
                spec.NumberInput(
                    id="demand",
                    about=gettext(
                        "Average consumptive water use in this LULC class."
                    ),
                    units=u.meter ** 3 / u.year / u.pixel
                )
            ],
            index_col="lucode"
        ),
        spec.CSVInput(
            id="valuation_table_path",
            name=gettext("hydropower valuation table"),
            about=gettext(
                "A table mapping each watershed to the associated valuation"
                " parameters for its hydropower station."
            ),
            required=False,
            columns=[
                spec.IntegerInput(
                    id="ws_id",
                    about=gettext(
                        "Unique identifier for the hydropower station. This"
                        " must match the 'ws_id' value for the corresponding"
                        " watershed in the Watersheds vector. Each watershed in"
                        " the Watersheds vector must have its 'ws_id' entered"
                        " in this column."
                    )
                ),
                spec.RatioInput(
                    id="efficiency",
                    about=gettext(
                        "Turbine efficiency, the proportion of potential energy"
                        " captured and converted to electricity by the turbine."
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="fraction",
                    about=gettext(
                        "The proportion of inflow water volume that is used to"
                        " generate energy."
                    ),
                    units=None
                ),
                spec.NumberInput(
                    id="height",
                    about=gettext(
                        "The head, measured as the average annual effective"
                        " height of water behind each dam at the turbine"
                        " intake."
                    ),
                    units=u.meter
                ),
                spec.NumberInput(
                    id="kw_price",
                    about=gettext(
                        "The price of power produced by the station. Must be in"
                        " the same currency used in the 'cost' column."
                    ),
                    units=u.currency / u.kilowatt_hour
                ),
                spec.NumberInput(
                    id="cost",
                    about=gettext(
                        "Annual maintenance and operations cost of running the"
                        " hydropower station. Must be in the same currency used"
                        " in the 'kw_price' column."
                    ),
                    units=u.currency / u.year
                ),
                spec.NumberInput(
                    id="time_span",
                    about=gettext(
                        "Number of years over which to value the hydropower"
                        " station. This is either the station's expected"
                        " lifespan or the duration of the land use scenario of"
                        " interest."
                    ),
                    units=u.year
                ),
                spec.PercentInput(
                    id="discount",
                    about=gettext(
                        "The annual discount rate, applied for each year in the"
                        " time span."
                    ),
                    units=None
                )
            ],
            index_col="ws_id"
        )
    ],
    outputs=[
        spec.VectorOutput(
            id="watershed_results_wyield",
            path="output/watershed_results_wyield.shp",
            about=gettext(
                "Shapefile containing biophysical output values per"
                " watershed."
            ),
            geometry_types={"POLYGON"},
            fields=WATERSHED_OUTPUT_FIELDS
        ),
        spec.CSVOutput(
            id="watershed_results_wyield_csv",
            path="output/watershed_results_wyield.csv",
            about=gettext(
                "Table containing biophysical output values per"
                " watershed."
            ),
            columns=WATERSHED_OUTPUT_FIELDS,
            index_col="ws_id"
        ),
        spec.VectorOutput(
            id="subwatershed_results_wyield",
            path="output/subwatershed_results_wyield.shp",
            about=gettext(
                "Shapefile containing biophysical output values per"
                " subwatershed."
            ),
            geometry_types={"POLYGON"},
            fields=SUBWATERSHED_OUTPUT_FIELDS
        ),
        spec.CSVOutput(
            id="subwatershed_results_wyield_csv",
            path="output/subwatershed_results_wyield.csv",
            about=gettext(
                "Table containing biophysical output values per"
                " subwatershed."
            ),
            columns=SUBWATERSHED_OUTPUT_FIELDS,
            index_col="subws_id"
        ),
        spec.SingleBandRasterOutput(
            id="fractp",
            path="output/per_pixel/fractp.tif",
            about=gettext(
                "The fraction of precipitation that actually"
                " evapotranspires at the pixel level."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="aet",
            path="output/per_pixel/aet.tif",
            about=gettext(
                "Estimated actual evapotranspiration per pixel."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="demand",
            path="intermediate/demand.tif",
            about=gettext("Water demand per pixel."),
            data_type=float,
            units=u.meter ** 3 / u.year
        ),
        spec.SingleBandRasterOutput(
            id="wyield",
            path="output/per_pixel/wyield.tif",
            about=gettext("Estimated water yield per pixel."),
            data_type=float,
            units=u.millimeter,
            created_if="demand_table_path"
        ),
        spec.SingleBandRasterOutput(
            id="clipped_lulc",
            path="intermediate/clipped_lulc.tif",
            about=gettext("Aligned and clipped copy of LULC input."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="depth_to_root_rest_layer",
            path="intermediate/depth_to_root_rest_layer.tif",
            about=gettext(
                "Aligned and clipped copy of root restricting layer"
                " depth input."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="eto",
            path="intermediate/eto.tif",
            about=gettext("Aligned and clipped copy of ET0 input."),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="kc_raster",
            path="intermediate/kc_raster.tif",
            about=gettext("Map of KC values."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="pawc",
            path="intermediate/pawc.tif",
            about=gettext("Aligned and clipped copy of PAWC input."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="pet",
            path="intermediate/pet.tif",
            about=gettext("Map of potential evapotranspiration."),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="precip",
            path="intermediate/precip.tif",
            about=gettext(
                "Aligned and clipped copy of precipitation input."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="root_depth",
            path="intermediate/root_depth.tif",
            about=gettext("Map of root depth."),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="veg",
            path="intermediate/veg.tif",
            about=gettext("Map of vegetated state."),
            data_type=int,
            units=None
        ),
        spec.FileOutput(
            id="ws_id_wyield_mn",
            path="intermediate/_tmp_zonal_stats/ws_id_wyield_mn.pickle",
            about=gettext("Pickled watershed zonal statistics dictionary for wyield"),
        ),
        spec.FileOutput(
            id="ws_id_aet_mn",
            path="intermediate/_tmp_zonal_stats/ws_id_AET_mn.pickle",
            about=gettext("Pickled watershed zonal statistics dictionary for AET"),
        ),
        spec.FileOutput(
            id="ws_id_pet_mn",
            path="intermediate/_tmp_zonal_stats/ws_id_PET_mn.pickle",
            about=gettext("Pickled watershed zonal statistics dictionary for PET"),
        ),
        spec.FileOutput(
            id="ws_id_precip_mn",
            path="intermediate/_tmp_zonal_stats/ws_id_precip_mn.pickle",
            about=gettext("Pickled watershed zonal statistics dictionary for precip"),
        ),
        spec.FileOutput(
            id="ws_id_demand",
            path="intermediate/_tmp_zonal_stats/ws_id_demand.pickle",
            about=gettext("Pickled watershed zonal statistics dictionary for demand"),
            created_if="demand_table_path"
        ),
        spec.FileOutput(
            id="subws_id_wyield_mn",
            path="intermediate/_tmp_zonal_stats/subws_id_wyield_mn.pickle",
            about=gettext("Pickled subwatershed zonal statistics dictionary for wyield"),
            created_if="sub_watersheds_path"
        ),
        spec.FileOutput(
            id="subws_id_aet_mn",
            path="intermediate/_tmp_zonal_stats/subws_id_AET_mn.pickle",
            about=gettext("Pickled subwatershed zonal statistics dictionary for AET"),
            created_if="sub_watersheds_path"
        ),
        spec.FileOutput(
            id="subws_id_pet_mn",
            path="intermediate/_tmp_zonal_stats/subws_id_PET_mn.pickle",
            about=gettext("Pickled subwatershed zonal statistics dictionary for PET"),
            created_if="sub_watersheds_path"
        ),
        spec.FileOutput(
            id="subws_id_precip_mn",
            path="intermediate/_tmp_zonal_stats/subws_id_precip_mn.pickle",
            about=gettext("Pickled subwatershed zonal statistics dictionary for precip"),
            created_if="sub_watersheds_path"
        ),
        spec.FileOutput(
            id="subws_id_demand",
            path="intermediate/_tmp_zonal_stats/subws_id_demand.pickle",
            about=gettext("Pickled subwatershed zonal statistics dictionary for demand"),
            created_if="sub_watersheds_path and demand_table_path"
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Annual Water Yield: Reservoir Hydropower Production.

    Executes the hydropower/annual water yield model

    Args:
        args['workspace_dir'] (string): a path to the directory that will write
            output and other temporary files during calculation. (required)

        args['lulc_path'] (string): a path to a land use/land cover raster
            whose LULC indexes correspond to indexes in the biophysical table
            input. Used for determining soil retention and other biophysical
            properties of the landscape. (required)

        args['depth_to_root_rest_layer_path'] (string): a path to an input
            raster describing the depth of "good" soil before reaching this
            restrictive layer (required)

        args['precipitation_path'] (string): a path to an input raster
            describing the average annual precipitation value for each cell
            (mm) (required)

        args['pawc_path'] (string): a path to an input raster describing the
            plant available water content value for each cell. Plant Available
            Water Content fraction (PAWC) is the fraction of water that can be
            stored in the soil profile that is available for plants' use.
            PAWC is a fraction from 0 to 1 (required)

        args['eto_path'] (string): a path to an input raster describing the
            annual average evapotranspiration value for each cell. Potential
            evapotranspiration is the potential loss of water from soil by
            both evaporation from the soil and transpiration by healthy
            Alfalfa (or grass) if sufficient water is available (mm)
            (required)

        args['watersheds_path'] (string): a path to an input shapefile of the
            watersheds of interest as polygons. (required)

        args['sub_watersheds_path'] (string): a path to an input shapefile of
            the subwatersheds of interest that are contained in the
            ``args['watersheds_path']`` shape provided as input. (optional)

        args['biophysical_table_path'] (string): a path to an input CSV table
            of land use/land cover classes, containing data on biophysical
            coefficients such as root_depth (mm) and Kc, which are required.
            A column with header LULC_veg is also required which should
            have values of 1 or 0, 1 indicating a land cover type of
            vegetation, a 0 indicating non vegetation or wetland, water.
            NOTE: these data are attributes of each LULC class rather than
            attributes of individual cells in the raster map (required)

        args['seasonality_constant'] (float): floating point value between
            1 and 30 corresponding to the seasonal distribution of
            precipitation (required)

        args['results_suffix'] (string): a string that will be concatenated
            onto the end of file names (optional)

        args['demand_table_path'] (string): (optional) if a non-empty string,
            a path to an input CSV
            table of LULC classes, showing consumptive water use for each
            landuse / land-cover type (cubic meters per year) to calculate
            water scarcity.  Required if ``valuation_table_path`` is provided.

        args['valuation_table_path'] (string): (optional) if a non-empty
            string, a path to an input CSV table of
            hydropower stations with the following fields to calculate
            valuation: 'ws_id', 'time_span', 'discount', 'efficiency',
            'fraction', 'cost', 'height', 'kw_price'

        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, graph = MODEL_SPEC.setup(args)

    # valuation_df is passed to create_vector_output()
    # which computes valuation if valuation_df is not None.
    valuation_df = None
    if args['valuation_table_path']:
        LOGGER.info(
            'Checking that watersheds have entries for every `ws_id` in the '
            'valuation table.')
        # Open/read in valuation parameters from CSV file
        valuation_df = MODEL_SPEC.get_input(
            'valuation_table_path').get_validated_dataframe(args['valuation_table_path'])
        watershed_vector = gdal.OpenEx(
            args['watersheds_path'], gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()
        missing_ws_ids = []
        for watershed_feature in watershed_layer:
            watershed_ws_id = watershed_feature.GetField('ws_id')
            if watershed_ws_id not in valuation_df.index:
                missing_ws_ids.append(watershed_ws_id)
        watershed_feature = None
        watershed_layer = None
        watershed_vector = None
        if missing_ws_ids:
            raise ValueError(
                'The following `ws_id`s exist in the watershed vector file '
                'but are not found in the valuation table. Check your '
                'valuation table to see if they are missing: '
                f'"{", ".join(str(x) for x in sorted(missing_ws_ids))}"')

    watershed_paths_list = [(
        args['watersheds_path'], 'ws_id',
        file_registry['watershed_results_wyield'],
        file_registry['watershed_results_wyield_csv'])]

    if args['sub_watersheds_path']:
        watershed_paths_list.append((
            args['sub_watersheds_path'], 'subws_id',
            file_registry['subwatershed_results_wyield'],
            file_registry['subwatershed_results_wyield_csv']))

    base_raster_path_list = [
        args['eto_path'],
        args['precipitation_path'],
        args['depth_to_root_rest_layer_path'],
        args['pawc_path'],
        args['lulc_path']]

    aligned_raster_path_list = [
        file_registry['eto'],
        file_registry['precip'],
        file_registry['depth_to_root_rest_layer'],
        file_registry['pawc'],
        file_registry['clipped_lulc']]

    target_pixel_size = pygeoprocessing.get_raster_info(
        args['lulc_path'])['pixel_size']
    align_raster_stack_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        args=(base_raster_path_list, aligned_raster_path_list,
              ['near'] * len(base_raster_path_list),
              target_pixel_size, 'intersection'),
        kwargs={'raster_align_index': 4,
                'base_vector_path_list': [args['watersheds_path']]},
        target_path_list=aligned_raster_path_list,
        task_name='align_raster_stack')
    # Joining now since this task will always be the root node
    # and it's useful to have the raster info available.
    align_raster_stack_task.join()

    nodata_dict = {
        'out_nodata': -1,
        'precip': pygeoprocessing.get_raster_info(file_registry['precip'])['nodata'][0],
        'eto': pygeoprocessing.get_raster_info(file_registry['eto'])['nodata'][0],
        'depth_root': pygeoprocessing.get_raster_info(
            file_registry['depth_to_root_rest_layer'])['nodata'][0],
        'pawc': pygeoprocessing.get_raster_info(file_registry['pawc'])['nodata'][0],
        'lulc': pygeoprocessing.get_raster_info(file_registry['clipped_lulc'])['nodata'][0]}

    # Open/read in the csv file into a dictionary and add to arguments
    bio_df = MODEL_SPEC.get_input('biophysical_table_path').get_validated_dataframe(
        args['biophysical_table_path'])

    bio_lucodes = set(bio_df.index.values)
    bio_lucodes.add(nodata_dict['lulc'])
    LOGGER.debug(f'bio_lucodes: {bio_lucodes}')

    if args['demand_table_path']:
        demand_df = MODEL_SPEC.get_input('demand_table_path').get_validated_dataframe(
            args['demand_table_path'])
        demand_reclassify_dict = dict(
            [(lucode, row['demand']) for lucode, row in demand_df.iterrows()])
        demand_lucodes = set(demand_df.index.values)
        demand_lucodes.add(nodata_dict['lulc'])
        LOGGER.debug(f'demand_lucodes: {demand_lucodes}', )
    else:
        demand_lucodes = None

    # Break the bio_df into three separate dictionaries based on
    # Kc, root_depth, and LULC_veg fields to use for reclassifying
    Kc_dict = {}
    root_dict = {}
    vegetated_dict = {}

    for lulc_code, row in bio_df.iterrows():
        Kc_dict[lulc_code] = row['kc']

        # Catch invalid LULC_veg values with an informative error.
        if row['lulc_veg'] not in set([0, 1]):
            # If the user provided an invalid LULC_veg value, raise an
            # informative error.
            raise ValueError(
                f'LULC_veg value must be either 1 or 0, not {row["lulc_veg"]}')
        vegetated_dict[lulc_code] = row['lulc_veg']

        # If LULC_veg value is 1 get root depth value
        if vegetated_dict[lulc_code] == 1:
            root_dict[lulc_code] = row['root_depth']
        # If LULC_veg value is 0 then we do not care about root
        # depth value so will just substitute in a 1. This
        # value will not end up being used.
        else:
            root_dict[lulc_code] = 1

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}
    # Create Kc raster from table values to use in future calculations
    LOGGER.info("Reclassifying temp_Kc raster")
    create_Kc_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((file_registry['clipped_lulc'], 1), Kc_dict, file_registry['kc_raster'],
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[file_registry['kc_raster']],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_Kc_raster')

    # Create root raster from table values to use in future calculations
    LOGGER.info("Reclassifying tmp_root raster")
    create_root_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((file_registry['clipped_lulc'], 1), root_dict, file_registry['root_depth'],
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[file_registry['root_depth']],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_root_raster')

    # Create veg raster from table values to use in future calculations
    # of determining which AET equation to use
    LOGGER.info("Reclassifying tmp_veg raster")
    create_veg_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((file_registry['clipped_lulc'], 1), vegetated_dict, file_registry['veg'],
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[file_registry['veg']],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_veg_raster')

    dependent_tasks_for_watersheds_list = []

    LOGGER.info('Calculate PET from Ref Evap times Kc')
    calculate_pet_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.multiply,  # PET = ET0 * KC
            rasters=[file_registry['eto'], file_registry['kc_raster']],
            target_path=file_registry['pet'],
            target_nodata=nodata_dict['out_nodata']),
        target_path_list=[file_registry['pet']],
        dependent_task_list=[create_Kc_raster_task],
        task_name='calculate_pet')
    dependent_tasks_for_watersheds_list.append(calculate_pet_task)

    # List of rasters to pass into the vectorized fractp operation
    raster_list = [
        file_registry['kc_raster'], file_registry['eto'], file_registry['precip'], file_registry['root_depth'],
        file_registry['depth_to_root_rest_layer'], file_registry['pawc'], file_registry['veg']]

    LOGGER.debug('Performing fractp operation')
    calculate_fractp_task = graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(x, 1) for x in raster_list]
              + [(nodata_dict, 'raw'), (args['seasonality_constant'], 'raw')],
              fractp_op, file_registry['fractp'], gdal.GDT_Float32,
              nodata_dict['out_nodata']),
        target_path_list=[file_registry['fractp']],
        dependent_task_list=[
            create_Kc_raster_task, create_veg_raster_task,
            create_root_raster_task, align_raster_stack_task],
        task_name='calculate_fractp')

    LOGGER.info('Performing wyield operation')
    calculate_wyield_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=wyield_op,
            rasters=[file_registry['fractp'], file_registry['precip']],
            target_path=file_registry['wyield'],
            target_nodata=nodata_dict['out_nodata']),
        target_path_list=[file_registry['wyield']],
        dependent_task_list=[calculate_fractp_task, align_raster_stack_task],
        task_name='calculate_wyield')
    dependent_tasks_for_watersheds_list.append(calculate_wyield_task)

    LOGGER.debug('Performing aet operation')
    calculate_aet_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.multiply,  # AET = fractp * precip
            rasters=[file_registry['fractp'], file_registry['precip']],
            target_path=file_registry['aet'],
            target_nodata=nodata_dict['out_nodata']),
        target_path_list=[file_registry['aet']],
        dependent_task_list=[
            calculate_fractp_task, create_veg_raster_task,
            align_raster_stack_task],
        task_name='calculate_aet')
    dependent_tasks_for_watersheds_list.append(calculate_aet_task)

    # list of rasters that will always be summarized with zonal stats
    raster_names_paths_list = [
        ('precip_mn', file_registry['precip']),
        ('PET_mn', file_registry['pet']),
        ('AET_mn', file_registry['aet']),
        ('wyield_mn', file_registry['wyield'])]

    if args['demand_table_path']:
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Demand'}
        # Create demand raster from table values to use in future calculations
        create_demand_raster_task = graph.add_task(
            func=utils.reclassify_raster,
            args=((file_registry['clipped_lulc'], 1), demand_reclassify_dict, file_registry['demand'],
                  gdal.GDT_Float32, nodata_dict['out_nodata'],
                  reclass_error_details),
            target_path_list=[file_registry['demand']],
            dependent_task_list=[align_raster_stack_task],
            task_name='create_demand_raster')
        dependent_tasks_for_watersheds_list.append(create_demand_raster_task)
        raster_names_paths_list.append(('demand', file_registry['demand']))

    # Aggregate results to watershed polygons, and do the optional
    # scarcity and valuation calculations.
    for base_ws_path, ws_id_name, target_ws_path, target_csv_path in watershed_paths_list:
        # make a copy so we don't modify the original
        # do zonal stats with the copy so that FIDS are correct
        copy_watersheds_vector_task = graph.add_task(
            func=copy_vector,
            args=[base_ws_path, target_ws_path],
            target_path_list=[target_ws_path],
            task_name='create copy of watersheds vector')

        zonal_stats_task_list = []
        zonal_stats_pickle_list = []

        # Do zonal stats with the input shapefiles provided by the user
        # and store results dictionaries in pickles
        for key_name, rast_path in raster_names_paths_list:
            target_stats_pickle = file_registry[f'{ws_id_name}_{key_name.lower()}']
            zonal_stats_pickle_list.append((target_stats_pickle, key_name))
            zonal_stats_task_list.append(graph.add_task(
                func=zonal_stats_tofile,
                args=(target_ws_path, rast_path, target_stats_pickle),
                target_path_list=[target_stats_pickle],
                dependent_task_list=[
                    *dependent_tasks_for_watersheds_list,
                    copy_watersheds_vector_task],
                task_name=f'{ws_id_name}_{key_name}_zonalstats'))

        # Add the zonal stats data to the output vector's attribute table
        # Compute optional scarcity and valuation
        write_output_vector_attributes_task = graph.add_task(
            func=write_output_vector_attributes,
            args=(target_ws_path, ws_id_name, zonal_stats_pickle_list,
                  valuation_df),
            target_path_list=[target_ws_path],
            dependent_task_list=[
                *zonal_stats_task_list, copy_watersheds_vector_task],
            task_name=f'create_{ws_id_name}_vector_output')

        # Export a CSV with all the fields present in the output vector
        create_output_table_task = graph.add_task(
            func=convert_vector_to_csv,
            args=(target_ws_path, target_csv_path),
            target_path_list=[target_csv_path],
            dependent_task_list=[write_output_vector_attributes_task],
            task_name=f'create_{ws_id_name}_table_output')

    graph.join()
    return file_registry.registry


# wyield equation to pass to raster_map
def wyield_op(fractp, precip): return (1 - fractp) * precip


def copy_vector(base_vector_path, target_vector_path):
    """Wrapper around CreateCopy that handles opening & closing the dataset.

    Args:
        base_vector_path: path to the vector to copy
        target_vector_path: path to copy the vector to

    Returns:
        None
    """
    esri_shapefile_driver = gdal.GetDriverByName('ESRI Shapefile')
    base_dataset = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    esri_shapefile_driver.CreateCopy(target_vector_path, base_dataset)
    base_dataset = None


def write_output_vector_attributes(target_vector_path, ws_id_name,
                                   stats_path_list, valuation_df):
    """Add data attributes to the vector outputs of this model.

    Join results of zonal stats to copies of the watershed shapefiles.
    Also do optional scarcity and valuation calculations.

    Args:
        target_vector_path (string): Path to the watersheds vector to modify
        ws_id_name (string): Either 'ws_id' or 'subws_id', which are required
            names of a unique ID field in the watershed and subwatershed
            shapefiles, respectively. Used to determine if the polygons
            represent watersheds or subwatersheds.
        stats_path_list (list): List of file paths to pickles storing the zonal
            stats results.
        valuation_df (pandas.DataFrame): dataframe built from
            args['valuation_table_path']. Or None if valuation table was not
            provided.

    Returns:
        None

    """
    for pickle_path, key_name in stats_path_list:
        with open(pickle_path, 'rb') as picklefile:
            ws_stats_dict = pickle.load(picklefile)

            if key_name == 'wyield_mn':
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, key_name, 'mean')
                # Also create and populate 'wyield_vol' field, which
                # relies on 'wyield_mn' already present in attribute table
                compute_water_yield_volume(target_vector_path)

            # consum_* variables rely on 'wyield_*' fields present,
            # so this would fail if somehow 'demand' comes before 'wyield_mn'
            # in key_names. The order is hardcoded in raster_names_paths_list.
            elif key_name == 'demand':
                # Add aggregated consumption to sheds shapefiles
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, 'consum_vol', 'sum')

                # Add aggregated consumption means to sheds shapefiles
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, 'consum_mn', 'mean')
                compute_rsupply_volume(target_vector_path)

            else:
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, key_name, 'mean')

    if valuation_df is not None:
        # only do valuation for watersheds, not subwatersheds
        if ws_id_name == 'ws_id':
            compute_watershed_valuation(target_vector_path, valuation_df)


def convert_vector_to_csv(base_vector_path, target_csv_path):
    """Create a CSV with all the fields present in vector attribute table.

    Args:
        base_vector_path (string):
            Path to the watershed shapefile in the output workspace.
        target_csv_path (string):
            Path to a CSV to create in the output workspace.

    Returns:
        None

    """
    watershed_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    csv_driver = gdal.GetDriverByName('CSV')
    _ = csv_driver.CreateCopy(target_csv_path, watershed_vector)


def zonal_stats_tofile(base_vector_path, raster_path, target_stats_pickle):
    """Calculate zonal statistics for watersheds and write results to a file.

    Args:
        base_vector_path (string): Path to the watershed shapefile in the
            output workspace.
        raster_path (string): Path to raster to aggregate.
        target_stats_pickle (string): Path to pickle file to store dictionary
            returned by zonal stats.

    Returns:
        None

    """
    ws_stats_dict = pygeoprocessing.zonal_statistics(
        (raster_path, 1), base_vector_path, ignore_nodata=True)
    with open(target_stats_pickle, 'wb') as picklefile:
        picklefile.write(pickle.dumps(ws_stats_dict))


def fractp_op(
        Kc, eto, precip, root, soil, pawc, veg,
        nodata_dict, seasonality_constant):
    """Calculate actual evapotranspiration fraction of precipitation.

    Args:
        Kc (numpy.ndarray): Kc (plant evapotranspiration
          coefficient) raster values
        eto (numpy.ndarray): potential evapotranspiration raster
          values (mm)
        precip (numpy.ndarray): precipitation raster values (mm)
        root (numpy.ndarray): root depth (maximum root depth for
           vegetated land use classes) raster values (mm)
        soil (numpy.ndarray): depth to root restricted layer raster
            values (mm)
        pawc (numpy.ndarray): plant available water content raster
           values
        veg (numpy.ndarray): 1 or 0 where 1 depicts the land type as
            vegetation and 0 depicts the land type as non
            vegetation (wetlands, urban, water, etc...). If 1 use
            regular AET equation if 0 use: AET = Kc * ETo
        nodata_dict (dict): stores nodata values keyed by raster names
        seasonality_constant (float): floating point value between
            1 and 30 corresponding to the seasonal distribution of
            precipitation.

    Returns:
        numpy.ndarray (float) of actual evapotranspiration as fraction
            of precipitation.

    """
    # Kc, root, & veg were created by reclassify_raster, which set nodata
    # to out_nodata. All others are products of align_and_resize_raster_stack
    # and retain their original nodata values.
    # out_nodata is defined above and should never be None.
    valid_mask = (
        ~pygeoprocessing.array_equals_nodata(Kc, nodata_dict['out_nodata']) &
        ~pygeoprocessing.array_equals_nodata(root, nodata_dict['out_nodata']) &
        ~pygeoprocessing.array_equals_nodata(veg, nodata_dict['out_nodata']) &
        ~pygeoprocessing.array_equals_nodata(precip, 0))
    if nodata_dict['eto'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(eto, nodata_dict['eto'])
    if nodata_dict['precip'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(precip, nodata_dict['precip'])
    if nodata_dict['depth_root'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(
            soil, nodata_dict['depth_root'])
    if nodata_dict['pawc'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(pawc, nodata_dict['pawc'])

    # Compute Budyko Dryness index
    # Use the original AET equation if the land cover type is vegetation
    # If not vegetation (wetlands, urban, water, etc...) use
    # Alternative equation Kc * Eto
    phi = (Kc[valid_mask] * eto[valid_mask]) / precip[valid_mask]
    pet = Kc[valid_mask] * eto[valid_mask]

    # Calculate plant available water content (mm) using the minimum
    # of soil depth and root depth
    awc = numpy.where(
        root[valid_mask] < soil[valid_mask], root[valid_mask],
        soil[valid_mask]) * pawc[valid_mask]
    climate_w = (
        (awc / precip[valid_mask]) * seasonality_constant) + 1.25
    # Capping to 5 to set to upper limit if exceeded
    climate_w[climate_w > 5] = 5

    # Compute evapotranspiration partition of the water balance
    aet_p = (
        1 + (pet / precip[valid_mask])) - (
            (1 + (pet / precip[valid_mask]) ** climate_w) ** (
                1 / climate_w))

    # We take the minimum of the following values (phi, aet_p)
    # to determine the evapotranspiration partition of the
    # water balance (see users guide)
    veg_result = numpy.where(phi < aet_p, phi, aet_p)
    # Take the minimum of precip and Kc * ETo to avoid x / p > 1
    nonveg_result = Kc[valid_mask] * eto[valid_mask]
    nonveg_mask = precip[valid_mask] < Kc[valid_mask] * eto[valid_mask]
    nonveg_result[nonveg_mask] = precip[valid_mask][nonveg_mask]
    nonveg_result_fract = nonveg_result / precip[valid_mask]

    # If veg is 1 use the result for vegetated areas else use result
    # for non veg areas
    result = numpy.where(
        veg[valid_mask] == 1,
        veg_result, nonveg_result_fract)

    fractp = numpy.empty(valid_mask.shape, dtype=numpy.float32)
    fractp[:] = nodata_dict['out_nodata']
    fractp[valid_mask] = result
    return fractp


def compute_watershed_valuation(watershed_results_vector_path, val_df):
    """Compute net present value and energy for the watersheds.

    Args:
        watershed_results_vector_path (string):
            Path to an OGR shapefile for the watershed results.
            Where the results will be added.
        val_df (pandas.DataFrame): a dataframe that has all the valuation
            parameters for each watershed.

    Returns:
        None.

    """
    ws_ds = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    ws_layer = ws_ds.GetLayer()

    # The field names for the new attributes
    energy_field = 'hp_energy'
    npv_field = 'hp_val'

    # Add the new fields to the shapefile
    for new_field in [energy_field, npv_field]:
        field_defn = ogr.FieldDefn(new_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        ws_layer.CreateField(field_defn)

    ws_layer.ResetReading()
    # Iterate over the number of features (polygons)
    for ws_feat in ws_layer:
        # Get the watershed ID to index into the valuation parameter dictionary
        # Since we only allow valuation on watersheds (not subwatersheds)
        # it's okay to hardcode 'ws_id' here.
        ws_id = ws_feat.GetField('ws_id')
        # Get the rsupply volume for the watershed
        rsupply_vl = ws_feat.GetField('rsupply_vl')

        # there won't be a rsupply_vl value if the polygon feature only
        # covers nodata raster values, so check before doing math.
        if rsupply_vl is not None:
            # Compute hydropower energy production (KWH)
            # This is from the equation given in the Users' Guide
            energy = (
                val_df['efficiency'][ws_id] * val_df['fraction'][ws_id] *
                val_df['height'][ws_id] * rsupply_vl * 0.00272)

            dsum = 0
            # Divide by 100 because it is input at a percent and we need
            # decimal value
            disc = val_df['discount'][ws_id] / 100
            # To calculate the summation of the discount rate term over the life
            # span of the dam we can use a geometric series
            ratio = 1 / (1 + disc)
            if ratio != 1:
                dsum = (1 - math.pow(ratio, val_df['time_span'][ws_id])) / (1 - ratio)

            npv = ((val_df['kw_price'][ws_id] * energy) - val_df['cost'][ws_id]) * dsum

            # Get the volume field index and add value
            ws_feat.SetField(energy_field, energy)
            ws_feat.SetField(npv_field, npv)

            ws_layer.SetFeature(ws_feat)


def compute_rsupply_volume(watershed_results_vector_path):
    """Calculate the total realized water supply volume.

    And the mean realized water supply volume per pixel for the given sheds.
    Output units in cubic meters and cubic meters per pixel respectively.

    Args:
        watershed_results_vector_path (string): a path to a vector that
            contains fields 'wyield_vol' and 'wyield_mn'.

    Returns:
        None.

    """
    ws_ds = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    ws_layer = ws_ds.GetLayer()

    # The field names for the new attributes
    rsupply_vol_name = 'rsupply_vl'
    rsupply_mn_name = 'rsupply_mn'

    # Add the new fields to the shapefile
    for new_field in [rsupply_vol_name, rsupply_mn_name]:
        field_defn = ogr.FieldDefn(new_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        ws_layer.CreateField(field_defn)

    ws_layer.ResetReading()
    # Iterate over the number of features (polygons)
    for ws_feat in ws_layer:
        # Get mean and volume water yield values
        wyield_mn = ws_feat.GetField('wyield_mn')
        wyield = ws_feat.GetField('wyield_vol')

        # Get water demand/consumption values
        consump_vol = ws_feat.GetField('consum_vol')
        consump_mn = ws_feat.GetField('consum_mn')

        # Calculate realized supply
        # these values won't exist if the polygon feature only
        # covers nodata raster values, so check before doing math.
        if wyield_mn is not None and consump_mn is not None:
            rsupply_vol = wyield - consump_vol
            rsupply_mn = wyield_mn - consump_mn

            # Set values for the new rsupply fields
            ws_feat.SetField(rsupply_vol_name, rsupply_vol)
            ws_feat.SetField(rsupply_mn_name, rsupply_mn)

            ws_layer.SetFeature(ws_feat)


def compute_water_yield_volume(watershed_results_vector_path):
    """Calculate the water yield volume per sub-watershed or watershed.

    Results are added to a 'wyield_vol' field in
    `watershed_results_vector_path`. Units are cubic meters.

    Args:
        watershed_results_vector_path (str): Path to a sub-watershed
            or watershed vector. This vector's features should have a
            'wyield_mn' attribute.

    Returns:
        None.

    """
    shape = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    layer = shape.GetLayer()

    # The field names for the new attributes
    vol_name = 'wyield_vol'

    # Add the new field to the shapefile
    field_defn = ogr.FieldDefn(vol_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)

    layer.ResetReading()
    # Iterate over the number of features (polygons) and compute volume
    for feat in layer:
        wyield_mn = feat.GetField('wyield_mn')
        # there won't be a wyield_mn value if the polygon feature only
        # covers nodata raster values, so check before doing math.
        if wyield_mn is not None:
            geom = feat.GetGeometryRef()
            # Calculate water yield volume,
            # 1000 is for converting the mm of wyield to meters
            vol = wyield_mn * geom.Area() / 1000
            # Get the volume field index and add value
            feat.SetField(vol_name, vol)

            layer.SetFeature(feat)


def _add_zonal_stats_dict_to_shape(
        watershed_results_vector_path,
        stats_map, field_name, aggregate_field_id):
    """Add a new field to a shapefile with values from a dictionary.

    Args:
        watershed_results_vector_path (string): a path to a vector whose FIDs
            correspond with the keys in `stats_map`.
        stats_map (dict): a dictionary in the format generated by
            pygeoprocessing.zonal_statistics that contains at least the key
            value of `aggregate_field_id` per feature id.
        field_name (str): a string for the name of the new field to add to
            the target vector.
        aggregate_field_id (string): one of 'min' 'max' 'sum' 'mean' 'count'
            or 'nodata_count' as defined by pygeoprocessing.zonal_statistics.

    Returns:
        None

    """
    vector = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    layer = vector.GetLayer()

    # Create the new field
    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)

    # Get the number of features (polygons) and iterate through each
    layer.ResetReading()
    for feature in layer:
        feature_fid = feature.GetFID()

        # Using the unique feature ID, index into the
        # dictionary to get the corresponding value
        # only write a value if zonal stats found valid pixels in the polygon:
        if stats_map[feature_fid]['count'] > 0:
            if aggregate_field_id == 'mean':
                field_val = float(
                    stats_map[feature_fid]['sum']) / stats_map[feature_fid]['count']
            else:
                field_val = float(stats_map[feature_fid][aggregate_field_id])

            # Set the value for the new field
            feature.SetField(field_name, field_val)

            layer.SetFeature(feature)


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
