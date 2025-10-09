"""InVEST Crop Production Percentile Model."""
from collections import defaultdict
import logging
import os
import re

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr

from . import gettext
from . import spec
from . import utils
from . import validation
from .crop_production_regression import (
    NUTRIENTS, NUTRIENT_UNITS, CROP_TO_PATH_TABLES, LULC_RASTER_INPUT,
    get_full_path_from_crop_table)
from .file_registry import FileRegistry
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

CROP_OPTIONS = [
    # Human-readable/translatable crop names come from three sources:
    # (1) Monfreda et. al. Table 1
    # (2) "EarthStat and FAO crop names and crop groups" table
    # (3) FAO's _World Programme for the Census of Agriculture 2020_
    # Where (1) and (2) differ, default to (1), except where (2) is
    # more descriptive (i.e., include additional list items, alternate
    # names, qualifiers, and other disambiguations).
    # Where discrepancies remain, consult (3) for additional context.
    # See #614 for more details and links to sources.
    spec.Option(key="abaca", about=gettext("Abaca (manila hemp)")),
    spec.Option(key="agave", about=gettext("Agave fibers, other")),
    spec.Option(key="alfalfa", about=gettext("Alfalfa")),
    spec.Option(key="almond", about=gettext("Almonds, with shell")),
    spec.Option(key="aniseetc", about=gettext("Anise, badian, fennel, coriander")),
    spec.Option(key="apple", about=gettext("Apples")),
    spec.Option(key="apricot", about=gettext("Apricots")),
    spec.Option(key="areca", about=gettext("Areca nuts (betel)")),
    spec.Option(key="artichoke", about=gettext("Artichokes")),
    spec.Option(key="asparagus", about=gettext("Asparagus")),
    spec.Option(key="avocado", about=gettext("Avocados")),
    spec.Option(key="bambara", about=gettext("Bambara beans")),
    spec.Option(key="banana", about=gettext("Bananas")),
    spec.Option(key="barley", about=gettext("Barley")),
    spec.Option(key="bean", about=gettext("Beans, dry")),
    spec.Option(key="beetfor", about=gettext("Beets for fodder")),
    spec.Option(key="berrynes", about=gettext("Berries, other")),
    spec.Option(key="blueberry", about=gettext("Blueberries")),
    spec.Option(key="brazil", about=gettext("Brazil nuts, with shell")),
    spec.Option(key="broadbean", about=gettext("Broad beans, horse beans, dry")),
    spec.Option(key="buckwheat", about=gettext("Buckwheat")),
    spec.Option(key="cabbage", about=gettext("Cabbages and other brassicas")),
    spec.Option(key="cabbagefor", about=gettext("Cabbage for fodder")),
    spec.Option(key="canaryseed", about=gettext("Canary seed")),
    spec.Option(key="carob", about=gettext("Carobs")),
    spec.Option(key="carrot", about=gettext("Carrots and turnips")),
    spec.Option(key="carrotfor", about=gettext("Carrots for fodder")),
    spec.Option(key="cashew", about=gettext("Cashew nuts, with shell")),
    spec.Option(key="cashewapple", about=gettext("Cashew apple")),
    spec.Option(key="cassava", about=gettext("Cassava")),
    spec.Option(key="castor", about=gettext("Castor beans")),
    spec.Option(key="cauliflower", about=gettext("Cauliflower and broccoli")),
    spec.Option(key="cerealnes", about=gettext("Cereals, other")),
    spec.Option(key="cherry", about=gettext("Cherries")),
    spec.Option(key="chestnut", about=gettext("Chestnuts")),
    spec.Option(key="chickpea", about=gettext("Chick peas")),
    spec.Option(key="chicory", about=gettext("Chicory roots")),
    spec.Option(key="chilleetc", about=gettext("Chilies and peppers, green")),
    spec.Option(key="cinnamon", about=gettext("Cinnamon (canella)")),
    spec.Option(key="citrusnes", about=gettext("Citrus fruit, other")),
    spec.Option(key="clove", about=gettext("Cloves")),
    spec.Option(key="clover", about=gettext("Clover")),
    spec.Option(key="cocoa", about=gettext("Cocoa beans")),
    spec.Option(key="coconut", about=gettext("Coconuts")),
    spec.Option(key="coffee", about=gettext("Coffee, green")),
    spec.Option(key="cotton", about=gettext("Cotton")),
    spec.Option(key="cowpea", about=gettext("Cow peas, dry")),
    spec.Option(key="cranberry", about=gettext("Cranberries")),
    spec.Option(key="cucumberetc", about=gettext("Cucumbers and gherkins")),
    spec.Option(key="currant", about=gettext("Currants")),
    spec.Option(key="date", about=gettext("Dates")),
    spec.Option(key="eggplant", about=gettext("Eggplants (aubergines)")),
    spec.Option(key="fibrenes", about=gettext("Fiber crops, other")),
    spec.Option(key="fig", about=gettext("Figs")),
    spec.Option(key="flax", about=gettext("Flax fiber and tow")),
    spec.Option(key="fonio", about=gettext("Fonio")),
    spec.Option(key="fornes", about=gettext("Forage products, other")),
    spec.Option(key="fruitnes", about=gettext("Fresh fruit, other")),
    spec.Option(key="garlic", about=gettext("Garlic")),
    spec.Option(key="ginger", about=gettext("Ginger")),
    spec.Option(key="gooseberry", about=gettext("Gooseberries")),
    spec.Option(key="grape", about=gettext("Grapes")),
    spec.Option(key="grapefruitetc", about=gettext("Grapefruit and pomelos")),
    spec.Option(key="grassnes", about=gettext("Grasses, other")),
    spec.Option(key="greenbean", about=gettext("Beans, green")),
    spec.Option(key="greenbroadbean", about=gettext("Broad beans, green")),
    spec.Option(key="greencorn", about=gettext("Green corn (maize)")),
    spec.Option(key="greenonion", about=gettext("Onions and shallots, green")),
    spec.Option(key="greenpea", about=gettext("Peas, green")),
    spec.Option(key="groundnut", about=gettext("Groundnuts, with shell")),
    spec.Option(key="hazelnut", about=gettext("Hazelnuts (filberts), with shell")),
    spec.Option(key="hemp", about=gettext("Hemp fiber and tow")),
    spec.Option(key="hempseed", about=gettext("Hempseed")),
    spec.Option(key="hop", about=gettext("Hops")),
    spec.Option(key="jute", about=gettext("Jute")),
    spec.Option(key="jutelikefiber", about=gettext("Jute-like fibers")),
    spec.Option(key="kapokfiber", about=gettext("Kapok fiber")),
    spec.Option(key="kapokseed", about=gettext("Kapok seed in shell")),
    spec.Option(key="karite", about=gettext("Karite nuts (shea nuts)")),
    spec.Option(key="kiwi", about=gettext("Kiwi fruit")),
    spec.Option(key="kolanut", about=gettext("Kola nuts")),
    spec.Option(key="legumenes", about=gettext("Legumes, other")),
    spec.Option(key="lemonlime", about=gettext("Lemons and limes")),
    spec.Option(key="lentil", about=gettext("Lentils")),
    spec.Option(key="lettuce", about=gettext("Lettuce and chicory")),
    spec.Option(key="linseed", about=gettext("Linseed")),
    spec.Option(key="lupin", about=gettext("Lupins")),
    spec.Option(key="maize", about=gettext("Maize")),
    spec.Option(key="maizefor", about=gettext("Maize for forage and silage")),
    spec.Option(key="mango", about=gettext("Mangoes, mangosteens, guavas")),
    spec.Option(key="mate", about=gettext("Mate")),
    spec.Option(key="melonetc", about=gettext("Cantaloupes and other melons")),
    spec.Option(key="melonseed", about=gettext("Melon seed")),
    spec.Option(key="millet", about=gettext("Millet")),
    spec.Option(key="mixedgrain", about=gettext("Mixed grain")),
    spec.Option(key="mixedgrass", about=gettext("Mixed grasses and legumes")),
    spec.Option(key="mushroom", about=gettext("Mushrooms and truffles")),
    spec.Option(key="mustard", about=gettext("Mustard seed")),
    spec.Option(key="nutmeg", about=gettext("Nutmeg, mace, and cardamoms")),
    spec.Option(key="nutnes", about=gettext("Nuts, other")),
    spec.Option(key="oats", about=gettext("Oats")),
    spec.Option(key="oilpalm", about=gettext("Oil palm fruit")),
    spec.Option(key="oilseedfor", about=gettext("Green oilseeds for fodder")),
    spec.Option(key="oilseednes", about=gettext("Oilseeds, other")),
    spec.Option(key="okra", about=gettext("Okra")),
    spec.Option(key="olive", about=gettext("Olives")),
    spec.Option(key="onion", about=gettext("Onions, dry")),
    spec.Option(key="orange", about=gettext("Oranges")),
    spec.Option(key="papaya", about=gettext("Papayas")),
    spec.Option(key="pea", about=gettext("Peas, dry")),
    spec.Option(key="peachetc", about=gettext("Peaches and nectarines")),
    spec.Option(key="pear", about=gettext("Pears")),
    spec.Option(key="pepper", about=gettext("Pepper (Piper spp.)")),
    spec.Option(key="peppermint", about=gettext("Peppermint")),
    spec.Option(key="persimmon", about=gettext("Persimmons")),
    spec.Option(key="pigeonpea", about=gettext("Pigeon peas")),
    spec.Option(key="pimento", about=gettext("Chilies and peppers, dry")),
    spec.Option(key="pineapple", about=gettext("Pineapples")),
    spec.Option(key="pistachio", about=gettext("Pistachios")),
    spec.Option(key="plantain", about=gettext("Plantains")),
    spec.Option(key="plum", about=gettext("Plums and sloes")),
    spec.Option(key="poppy", about=gettext("Poppy seed")),
    spec.Option(key="potato", about=gettext("Potatoes")),
    spec.Option(key="pulsenes", about=gettext("Pulses, other")),
    spec.Option(key="pumpkinetc", about=gettext("Pumpkins, squash, gourds")),
    spec.Option(key="pyrethrum", about=gettext("Pyrethrum, dried flowers")),
    spec.Option(key="quince", about=gettext("Quinces")),
    spec.Option(key="quinoa", about=gettext("Quinoa")),
    spec.Option(key="ramie", about=gettext("Ramie")),
    spec.Option(key="rapeseed", about=gettext("Rapeseed")),
    spec.Option(key="rasberry", about=gettext("Raspberries")),
    spec.Option(key="rice", about=gettext("Rice")),
    spec.Option(key="rootnes", about=gettext("Roots and tubers, other")),
    spec.Option(key="rubber", about=gettext("Natural rubber")),
    spec.Option(key="rye", about=gettext("Rye")),
    spec.Option(key="ryefor", about=gettext("Rye grass for forage and silage")),
    spec.Option(key="safflower", about=gettext("Safflower seed")),
    spec.Option(key="sesame", about=gettext("Sesame seed")),
    spec.Option(key="sisal", about=gettext("Sisal")),
    spec.Option(key="sorghum", about=gettext("Sorghum")),
    spec.Option(key="sorghumfor", about=gettext("Sorghum for forage and silage")),
    spec.Option(key="sourcherry", about=gettext("Sour cherries")),
    spec.Option(key="soybean", about=gettext("Soybeans")),
    spec.Option(key="spicenes", about=gettext("Spices, other")),
    spec.Option(key="spinach", about=gettext("Spinach")),
    spec.Option(key="stonefruitnes", about=gettext("Stone fruit, other")),
    spec.Option(key="strawberry", about=gettext("Strawberries")),
    spec.Option(key="stringbean", about=gettext("String beans")),
    spec.Option(key="sugarbeet", about=gettext("Sugar beets")),
    spec.Option(key="sugarcane", about=gettext("Sugar cane")),
    spec.Option(key="sugarnes", about=gettext("Sugar crops, other")),
    spec.Option(key="sunflower", about=gettext("Sunflower seed")),
    spec.Option(key="swedefor", about=gettext("Swedes for fodder")),
    spec.Option(key="sweetpotato", about=gettext("Sweet potatoes")),
    spec.Option(key="tangetc", about=gettext("Tangerines, mandarins, clementines")),
    spec.Option(key="taro", about=gettext("Taro")),
    spec.Option(key="tea", about=gettext("Tea")),
    spec.Option(key="tobacco", about=gettext("Tobacco leaves")),
    spec.Option(key="tomato", about=gettext("Tomatoes")),
    spec.Option(key="triticale", about=gettext("Triticale")),
    spec.Option(key="tropicalnes", about=gettext("Fresh tropical fruit, other")),
    spec.Option(key="tung", about=gettext("Tung nuts")),
    spec.Option(key="turnipfor", about=gettext("Turnips for fodder")),
    spec.Option(key="vanilla", about=gettext("Vanilla")),
    spec.Option(key="vegetablenes", about=gettext("Fresh vegetables, other")),
    spec.Option(key="vegfor", about=gettext("Vegetables and roots for fodder")),
    spec.Option(key="vetch", about=gettext("Vetches")),
    spec.Option(key="walnut", about=gettext("Walnuts, with shell")),
    spec.Option(key="watermelon", about=gettext("Watermelons")),
    spec.Option(key="wheat", about=gettext("Wheat")),
    spec.Option(key="yam", about=gettext("Yams")),
    spec.Option(key="yautia", about=gettext("Yautia"))
]

MODEL_SPEC = spec.ModelSpec(
    model_id="crop_production_percentile",
    model_title=gettext("Crop Production: Percentile"),
    userguide="crop_production.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("cpp",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        [CROP_TO_PATH_TABLES.percentile_yield,
         CROP_TO_PATH_TABLES.observed_yield,
         CROP_TO_PATH_TABLES.climate_bin, "crop_nutrient_table"],
        ["landcover_raster_path", "landcover_to_crop_table_path",
         "aggregate_polygon_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        LULC_RASTER_INPUT,
        spec.CSVInput(
            id="landcover_to_crop_table_path",
            name=gettext("LULC to Crop Table"),
            about=gettext(
                "A table that maps each LULC code from the LULC map to one of"
                " the 172 canonical crop names representing the crop grown in"
                " that LULC class."
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
        spec.AOI.model_copy(update=dict(
            id="aggregate_polygon_path",
            required=False,
            projected=True
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
            id=CROP_TO_PATH_TABLES.percentile_yield,
            name=gettext("Percentile Yield CSV Table"),
            about=gettext(
                "A table that maps each crop name to the corresponding"
                " percentile yield table."
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
                            id="yield_25th",
                            about=None,
                            units=u.metric_ton / u.hectare
                        ),
                        spec.NumberInput(
                            id="yield_50th",
                            about=None,
                            units=u.metric_ton / u.hectare
                        ),
                        spec.NumberInput(
                            id="yield_75th",
                            about=None,
                            units=u.metric_ton / u.hectare
                        ),
                        spec.NumberInput(
                            id="yield_95th",
                            about=None,
                            units=u.metric_ton / u.hectare
                        ),
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
            about=gettext("Model results aggregated to AOI polygons"),
            created_if="aggregate_polygon_path",
            columns=[
                spec.IntegerOutput(id="FID", about=gettext("FID of the AOI polygon")),
                spec.NumberOutput(
                    id="[CROP]_observed",
                    about=gettext(
                        "Observed production of the given crop within the polygon"
                    ),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="[CROP]_yield_[PERCENTILE]",
                    about=gettext(
                        "Modeled production of the given crop within the polygon at the"
                        " given percentile"
                    ),
                    units=u.metric_ton
                ),
                *[
                    spec.NumberOutput(
                        id=f"{nutrient_code}_observed",
                        about=f"Observed {nutrient} production within the polygon",
                        units=units
                    ) for nutrient_code, nutrient, units in NUTRIENTS
                ],
                *[
                    spec.NumberOutput(
                        id=f"{nutrient_code}_[PERCENTILE]",
                        about=(
                            f"Modeled {nutrient} production within the polygon at"
                            "the given percentile"),
                        units=units
                    ) for nutrient_code, nutrient, units in NUTRIENTS
                ]
            ],
            index_col="FID"
        ),
        spec.CSVOutput(
            id="result_table",
            path="result_table.csv",
            about=gettext("Model results aggregated by crop"),
            columns=[
                spec.StringOutput(id="crop_name", about=gettext("Name of the crop")),
                spec.NumberOutput(
                    id="area (ha)",
                    about=gettext("Area covered by the crop"),
                    units=u.hectare
                ),
                spec.NumberOutput(
                    id="production_observed",
                    about=gettext("Observed crop production"),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="production_[PERCENTILE]",
                    about=gettext("Modeled crop production at the given percentile"),
                    units=u.metric_ton
                ),
                *[
                    spec.NumberOutput(
                        id=f"{nutrient_code}_observed",
                        about=f"Observed {nutrient} production from the crop",
                        units=units
                    ) for nutrient_code, nutrient, units in NUTRIENTS
                ],
                *[
                    spec.NumberOutput(
                        id=f"{nutrient_code}_[PERCENTILE]",
                        about=(
                            f"Modeled {nutrient} production from the crop at"
                            "the given percentile"),
                        units=units
                    ) for nutrient_code, nutrient, units in NUTRIENTS
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
            id="[CROP]_[PERCENTILE]_production",
            path="[CROP]_[PERCENTILE]_production.tif",
            about=gettext("Modeled yield for the given crop at the given percentile"),
            data_type=float,
            units=u.metric_ton / u.hectare
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
            id="[CROP]_clipped_observed_yield",
            path="intermediate_output/[CROP]_clipped_observed_yield.tif",
            about=gettext(
                "Observed yield for the given crop, clipped to the extent of"
                " the landcover map"
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
            id="[CROP]_[PERCENTILE]_coarse_yield",
            path="intermediate_output/[CROP]_[PERCENTILE]_coarse_yield.tif",
            about=gettext(
                "Percentile yield of the given crop, at the coarse resolution of"
                " the climate bin map"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_[PERCENTILE]_interpolated_yield",
            path="intermediate_output/[CROP]_[PERCENTILE]_interpolated_yield.tif",
            about=gettext(
                "Percentile yield of the given crop, interpolated to the"
                " resolution of the landcover map"
            ),
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
        spec.VectorOutput(
            id="aggregate_vector",
            path="intermediate_output/aggregate_vector.shp",
            about=gettext("Model results aggregated to AOI polygons"),
            created_if="aggregate_polygon_path",
            fields=[
                spec.IntegerOutput(id="FID", about=gettext("FID of the AOI polygon")),
                spec.NumberOutput(
                    id="[CROP]_observed",
                    about=gettext(
                        "Observed production of the given crop within the polygon"
                    ),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="[CROP]_yield_[PERCENTILE]",
                    about=gettext(
                        "Modeled production of the given crop within the polygon at the"
                        " given percentile"
                    ),
                    units=u.metric_ton
                ),
                *[
                    spec.NumberOutput(
                        id=f"{nutrient_code}_observed",
                        about=f"Observed {nutrient} production within the polygon",
                        units=units
                    ) for nutrient_code, nutrient, units in NUTRIENTS
                ],
                *[
                    spec.NumberOutput(
                        id=f"{nutrient_code}_[PERCENTILE]",
                        about=(
                            f"Modeled {nutrient} production within the polygon at"
                            "the given percentile"),
                        units=units
                    ) for nutrient_code, nutrient, units in NUTRIENTS
                ]
            ]
        ),
        spec.TASKGRAPH_CACHE
    ]
)

_YIELD_PERCENTILE_FIELD_PATTERN = 'yield_([^_]+)'

_EXPECTED_NUTRIENT_TABLE_HEADERS = list(NUTRIENT_UNITS.keys())
_EXPECTED_LUCODE_TABLE_HEADER = 'lucode'
_NODATA_YIELD = -1


def execute(args):
    """Crop Production Percentile.

    This model will take a landcover (crop cover?) map and produce yields,
    production, and observed crop yields, a nutrient table, and a clipped
    observed map.

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

        args['aggregate_polygon_path'] (string): path to polygon shapefile
            that will be used to aggregate crop yields and total nutrient
            value. (optional, if value is None, then skipped)
        args['percentile_yield_csv_table'] (string): path to a table that maps
            each crop name to a path to its corresponding percentile yield
            table.
        args['climate_bin_raster_table'] (string): path to a table that maps
            each crop name to a path to its corresponding climate bin raster.
        args['observed_yield_raster_table'] (string): path to a table that maps
            each crop name to a path to its corresponding observed yield
            raster.
        args['crop_nutrient_table'] (string): path to a table that lists
            amounts of nutrients in each crop.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    # It might seem backwards to read the landcover_to_crop_table into a
    # DataFrame called crop_to_landcover_df, but since the table is indexed
    # by crop_name, it makes sense for the code to treat it as a mapping from
    # crop name to LULC code.
    crop_to_landcover_df = MODEL_SPEC.get_input(
        'landcover_to_crop_table_path').get_validated_dataframe(
            args['landcover_to_crop_table_path'])

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

    dependent_task_list = []

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

        LOGGER.info(
            "Clipping global climate bin raster to landcover bounding box.")
        crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
            crop_climate_bin_raster_path)
        crop_climate_bin_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(crop_climate_bin_raster_path,
                  crop_climate_bin_raster_info['pixel_size'],
                  file_registry['clipped_[CROP]_climate_bin_map', crop_name],
                  'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[
                file_registry['clipped_[CROP]_climate_bin_map', crop_name]],
            task_name='crop_climate_bin')
        dependent_task_list.append(crop_climate_bin_task)

        climate_percentile_yield_table_path = get_full_path_from_crop_table(
            MODEL_SPEC,
            CROP_TO_PATH_TABLES.percentile_yield,
            args[CROP_TO_PATH_TABLES.percentile_yield],
            crop_name)

        crop_climate_percentile_df = MODEL_SPEC.get_input(
            CROP_TO_PATH_TABLES.percentile_yield).get_column(
                'path').get_validated_dataframe(
                    climate_percentile_yield_table_path)

        yield_percentile_headers = [
            x for x in crop_climate_percentile_df.columns if x != 'climate_bin']

        reclassify_error_details = {
            'raster_name': f'{crop_name} Climate Bin',
            'column_name': 'climate_bin',
            'table_name': f'Climate {crop_name} Percentile Yield'}
        for yield_percentile_id in yield_percentile_headers:
            LOGGER.info("Map %s to climate bins.", yield_percentile_id)
            bin_to_percentile_yield = (
                crop_climate_percentile_df[yield_percentile_id].to_dict())
            # reclassify nodata to a valid value of 0
            # we're assuming that the crop doesn't exist where there is no data
            # this is more likely than assuming the crop does exist, esp.
            # in the context of the provided climate bins map
            bin_to_percentile_yield[
                crop_climate_bin_raster_info['nodata'][0]] = 0
            create_coarse_yield_percentile_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=((file_registry['clipped_[CROP]_climate_bin_map', crop_name], 1),
                      bin_to_percentile_yield,
                      file_registry['[CROP]_[PERCENTILE]_coarse_yield',
                        crop_name, yield_percentile_id], gdal.GDT_Float32,
                      _NODATA_YIELD, reclassify_error_details),
                target_path_list=[file_registry['[CROP]_[PERCENTILE]_coarse_yield',
                    crop_name, yield_percentile_id]],
                dependent_task_list=[crop_climate_bin_task],
                task_name='create_coarse_yield_percentile_%s_%s' % (
                    crop_name, yield_percentile_id))
            dependent_task_list.append(create_coarse_yield_percentile_task)

            LOGGER.info(
                "Interpolate %s %s yield raster to landcover resolution.",
                crop_name, yield_percentile_id)
            create_interpolated_yield_percentile_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(file_registry['[CROP]_[PERCENTILE]_coarse_yield',
                        crop_name, yield_percentile_id],
                      landcover_raster_info['pixel_size'],
                      file_registry['[CROP]_[PERCENTILE]_interpolated_yield',
                        crop_name, yield_percentile_id],
                      'cubicspline'),
                kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                        'target_bb': landcover_raster_info['bounding_box']},
                target_path_list=[file_registry['[CROP]_[PERCENTILE]_interpolated_yield',
                    crop_name, yield_percentile_id]],
                dependent_task_list=[create_coarse_yield_percentile_task],
                task_name='create_interpolated_yield_percentile_%s_%s' % (
                    crop_name, yield_percentile_id))
            dependent_task_list.append(
                create_interpolated_yield_percentile_task)

            LOGGER.info(
                "Calculate yield for %s at %s", crop_name,
                yield_percentile_id)

            create_percentile_production_task = task_graph.add_task(
                func=calculate_crop_production,
                args=(
                    args['landcover_raster_path'],
                    file_registry['[CROP]_[PERCENTILE]_interpolated_yield',
                        crop_name, yield_percentile_id],
                    crop_lucode,
                    file_registry['[CROP]_[PERCENTILE]_production',
                        crop_name, yield_percentile_id]),
                target_path_list=[file_registry['[CROP]_[PERCENTILE]_production',
                    crop_name, yield_percentile_id]],
                dependent_task_list=[
                    create_interpolated_yield_percentile_task],
                task_name='create_percentile_production_%s_%s' % (
                    crop_name, yield_percentile_id))
            dependent_task_list.append(create_percentile_production_task)

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
            target_path_list=[
                file_registry['[CROP]_clipped_observed_yield', crop_name]],
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
            target_path_list=[
                file_registry['[CROP]_interpolated_observed_yield', crop_name]],
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

    crop_names = crop_to_landcover_df.index.to_list()
    _ = task_graph.add_task(
        func=tabulate_results,
        args=(nutrient_df, yield_percentile_headers,
              crop_names, pixel_area_ha,
              args['landcover_raster_path'], landcover_nodata,
              file_registry, file_registry['result_table']),
        target_path_list=[file_registry['result_table']],
        dependent_task_list=dependent_task_list,
        task_name='tabulate_results')

    if args['aggregate_polygon_path']:
        LOGGER.info("aggregating result over query polygon")
        _ = task_graph.add_task(
            func=aggregate_to_polygons,
            args=(args['aggregate_polygon_path'],
                  file_registry['aggregate_vector'],
                  landcover_raster_info['projection_wkt'],
                  crop_names, nutrient_df, yield_percentile_headers,
                  pixel_area_ha, file_registry,
                  file_registry['aggregate_results']),
            target_path_list=[file_registry['aggregate_vector'],
                              file_registry['aggregate_results']],
            dependent_task_list=dependent_task_list,
            task_name='aggregate_results_to_polygons')

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def calculate_crop_production(lulc_path, yield_path, crop_lucode,
                              target_path):
    """Calculate crop production for a particular crop.

    The resulting production value is:

    - nodata, where either the LULC or yield input has nodata
    - 0, where the LULC does not match the given LULC code
    - yield (in Mg/ha), where the given LULC code exists

    Args:
        lulc_path (str): path to a raster of LULC codes
        yield_path (str): path of a raster of yields for the crop identified
            by ``crop_lucode``, in units per hectare
        crop_lucode (int): LULC code that identifies the crop of interest in
            the ``lulc_path`` raster.
        target_path (str): Path to write the output crop production raster

    Returns:
        None
    """
    pygeoprocessing.raster_map(
        op=lambda lulc, _yield: numpy.where(
            lulc == crop_lucode, _yield, 0),
        rasters=[lulc_path, yield_path],
        target_path=target_path,
        target_nodata=_NODATA_YIELD)


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
        valid_mask = ~pygeoprocessing.array_equals_nodata(lulc_array,
                                                          landcover_nodata)
        result[valid_mask] = 0
    else:
        result[:] = 0
    lulc_mask = lulc_array == crop_lucode
    result[lulc_mask] = observed_yield_array[lulc_mask]
    return result


def tabulate_results(
        nutrient_df, yield_percentile_headers, crop_names, pixel_area_ha,
        landcover_raster_path, landcover_nodata, file_registry,
        target_table_path):
    """Write table with total yield and nutrient results by crop.

    This function includes all the operations that write to results_table.csv.

    Args:
        nutrient_df (pandas.DataFrame): a table of nutrient values by crop
        yield_percentile_headers (list): list of strings indicating percentiles
            at which yield was calculated.
        crop_names (list): list of crop names
        pixel_area_ha (float): area of lulc raster cells (hectares)
        landcover_raster_path (string): path to landcover raster
        landcover_nodata (float): landcover raster nodata value
        file_registry (FileRegistry): used to look up output file paths
        target_table_path (string): path to 'result_table.csv' in the output
            workspace

    Returns:
        None

    """
    LOGGER.info("Generating report table")
    production_percentile_headers = [
        'production_' + re.match(
            _YIELD_PERCENTILE_FIELD_PATTERN,
            yield_percentile_id).group(1) for yield_percentile_id in sorted(
                yield_percentile_headers)]
    nutrient_headers = [
        nutrient_id + '_' + re.match(
            _YIELD_PERCENTILE_FIELD_PATTERN,
            yield_percentile_id).group(1)
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
        for yield_percentile_id in sorted(yield_percentile_headers) + [
            'yield_observed']]

    # Since pixel values in observed and percentile rasters are Mg/(ha•yr),
    # raster sums are (Mg•px)/(ha•yr). Before recording sums in
    # production_lookup dictionary, convert to Mg/yr by multiplying by ha/px.

    with open(target_table_path, 'w') as result_table:
        result_table.write(
            'crop_name,area (ha),' + 'production_observed,' +
            ','.join(production_percentile_headers) + ',' + ','.join(
                nutrient_headers) + '\n')
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
                    valid_mask & (yield_block > 0))
                yield_sum += numpy.sum(yield_block[valid_mask])
            yield_sum *= pixel_area_ha
            production_area = production_pixel_count * pixel_area_ha
            production_lookup['observed'] = yield_sum
            result_table.write(',%f' % production_area)
            result_table.write(",%f" % yield_sum)

            for yield_percentile_id in sorted(yield_percentile_headers):
                yield_sum = 0
                for _, yield_block in pygeoprocessing.iterblocks(
                        (file_registry['[CROP]_[PERCENTILE]_production',
                            crop_name, yield_percentile_id], 1)):
                    # _NODATA_YIELD will always have a value (defined above)
                    yield_sum += numpy.sum(
                        yield_block[~pygeoprocessing.array_equals_nodata(
                            yield_block, _NODATA_YIELD)])
                yield_sum *= pixel_area_ha
                production_lookup[yield_percentile_id] = yield_sum
                result_table.write(",%f" % yield_sum)

            # convert 100g to Mg and fraction left over from refuse
            nutrient_factor = 1e4 * (
                1 - nutrient_df['percentrefuse'][crop_name] / 100)
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for yield_percentile_id in sorted(yield_percentile_headers):
                    total_nutrient = (
                        nutrient_factor *
                        production_lookup[yield_percentile_id] *
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
                    ~pygeoprocessing.array_equals_nodata(band_values,
                                                         landcover_nodata))
            else:
                total_area += band_values.size
        result_table.write(
            '\n,total area (both crop and non-crop)\n,%f\n' % (
                total_area * pixel_area_ha))


def aggregate_to_polygons(
        base_aggregate_vector_path, target_aggregate_vector_path,
        landcover_raster_projection, crop_names, nutrient_df,
        yield_percentile_headers, pixel_area_ha, file_registry,
        target_aggregate_table_path):
    """Write table with aggregate results of yield and nutrient values.

    Use zonal statistics to summarize total observed and interpolated
    production and nutrient information for each polygon in
    base_aggregate_vector_path.

    Args:
        base_aggregate_vector_path (string): path to polygon vector
        target_aggregate_vector_path (string):
            path to re-projected copy of polygon vector
        landcover_raster_projection (string): a WKT projection string
        crop_names (list): list of crop names
        nutrient_df (pandas.DataFrame): a table of nutrient values by crop
        yield_percentile_headers (list): list of strings indicating percentiles
            at which yield was calculated.
        pixel_area_ha (float): area of lulc raster cells (hectares)
        file_registry (FileRegistry): used to look up output file paths
        target_aggregate_table_path (string): path to 'aggregate_results.csv'
            in the output workspace

    Returns:
        None

    """
    # reproject polygon to LULC's projection
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
        # loop over percentiles
        for yield_percentile_id in yield_percentile_headers:
            LOGGER.info(
                "Calculating zonal stats for %s  %s", crop_name,
                yield_percentile_id)
            total_yield_lookup['%s_%s' % (
                crop_name, yield_percentile_id)] = (
                    pygeoprocessing.zonal_statistics(
                        (file_registry['[CROP]_[PERCENTILE]_production',
                            crop_name, yield_percentile_id], 1),
                        target_aggregate_vector_path))

            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for id_index in total_yield_lookup['%s_%s' % (
                        crop_name, yield_percentile_id)]:
                    total_nutrient_table[nutrient_id][
                        yield_percentile_id][id_index] += (
                            nutrient_factor
                            * total_yield_lookup[
                                    '%s_%s' % (crop_name, yield_percentile_id)
                                ][id_index]['sum']
                            * pixel_area_ha
                            * nutrient_df[nutrient_id][crop_name])

        # process observed
        total_yield_lookup[f'{crop_name}_observed'] = (
            pygeoprocessing.zonal_statistics(
                (file_registry['[CROP]_observed_production', crop_name], 1),
                target_aggregate_vector_path))
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
            for id_index in total_yield_lookup[f'{crop_name}_observed']:
                total_nutrient_table[
                    nutrient_id]['observed'][id_index] += (
                        nutrient_factor
                        * total_yield_lookup[
                            f'{crop_name}_observed'][id_index]['sum']
                        * pixel_area_ha
                        * nutrient_df[nutrient_id][crop_name])

    # report everything to a table
    with open(target_aggregate_table_path, 'w') as aggregate_table:
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


# This decorator ensures the input arguments are formatted for InVEST
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
