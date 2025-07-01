"""InVEST Crop Production Percentile Model."""
import collections
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
from .crop_production_regression import NUTRIENTS
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

nutrient_units = {
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

MODEL_SPEC = spec.ModelSpec(
    model_id="crop_production_percentile",
    model_title=gettext("Crop Production: Percentile"),
    userguide="crop_production.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("cpp",),
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["model_data_path", "landcover_raster_path", "landcover_to_crop_table_path",
         "aggregate_polygon_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="landcover_raster_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover type must be"
                " assigned a unique integer code."
            ),
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.CSVInput(
            id="landcover_to_crop_table_path",
            name=gettext("LULC to Crop Table"),
            about=gettext(
                "A table that maps each LULC code from the LULC map to one of the 175"
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
        spec.AOI.model_copy(update=dict(
            id="aggregate_polygon_path",
            required=False,
            projected=True
        )),
        spec.DirectoryInput(
            id="model_data_path",
            name=gettext("model data directory"),
            about=gettext("Path to the InVEST Crop Production Data directory."),
            contents=[
                spec.DirectoryInput(
                    id="climate_percentile_yield_tables",
                    about=gettext(
                        "Table mapping each climate bin to yield percentiles for each"
                        " crop."
                    ),
                    contents=[
                        spec.CSVInput(
                            id="[CROP]_percentile_yield_table.csv",
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
                                )
                            ],
                            index_col="climate_bin"
                        )
                    ]
                ),
                spec.DirectoryInput(
                    id="extended_climate_bin_maps",
                    about=gettext("Maps of climate bins for each crop."),
                    contents=[
                        spec.SingleBandRasterInput(
                            id="extendedclimatebins[CROP]",
                            about=None,
                            data_type=int,
                            units=None,
                            projected=None
                        )
                    ]
                ),
                spec.DirectoryInput(
                    id="observed_yield",
                    about=gettext("Maps of actual observed yield for each crop."),
                    contents=[
                        spec.SingleBandRasterInput(
                            id="[CROP]_observed_yield.tif",
                            about=None,
                            data_type=float,
                            units=u.metric_ton / u.hectare,
                            projected=None
                        )
                    ]
                ),
                spec.CSVInput(
                    id="crop_nutrient.csv",
                    about=None,
                    columns=[
                        spec.OptionStringInput(
                            id="crop",
                            about=None,
                            options=CROP_OPTIONS
                        ),
                        spec.PercentInput(id="percentrefuse", about=None, units=None),
                        *[spec.NumberInput(id=nutrient, units=units)
                            for nutrient, units in nutrient_units.items()]
                    ],
                    index_col="crop"
                )
            ]
        )
    ],
    outputs=[
        spec.CSVOutput(
            id="aggregate_results.csv",
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
            id="result_table.csv",
            about=gettext("Model results aggregated by crop"),
            columns=[
                spec.StringOutput(id="crop", about=gettext("Name of the crop")),
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
            index_col="crop"
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_observed_production.tif",
            about=gettext("Observed yield for the given crop"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="[CROP]_yield_[PERCENTILE]_production.tif",
            about=gettext("Modeled yield for the given crop at the given percentile"),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.DirectoryOutput(
            id="intermediate",
            about=None,
            contents=[
                spec.SingleBandRasterOutput(
                    id="clipped_[CROP]_climate_bin_map.tif",
                    about=gettext(
                        "Climate bin map for the given crop, clipped to the LULC extent"
                    ),
                    data_type=int,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="[CROP]_clipped_observed_yield.tif",
                    about=gettext(
                        "Observed yield for the given crop, clipped to the extend of the"
                        " landcover map"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare
                ),
                spec.SingleBandRasterOutput(
                    id="[CROP]_interpolated_observed_yield.tif",
                    about=gettext(
                        "Observed yield for the given crop, interpolated to the"
                        " resolution of the landcover map"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare
                ),
                spec.SingleBandRasterOutput(
                    id="[CROP]_yield_[PERCENTILE]_coarse_yield.tif",
                    about=gettext(
                        "Percentile yield of the given crop, at the coarse resolution of"
                        " the climate bin map"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare
                ),
                spec.SingleBandRasterOutput(
                    id="[CROP]_yield_[PERCENTILE]_interpolated_yield.tif",
                    about=gettext(
                        "Percentile yield of the given crop, interpolated to the"
                        " resolution of the landcover map"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare
                ),
                spec.SingleBandRasterOutput(
                    id="[CROP]_zeroed_observed_yield.tif",
                    about=gettext(
                        "Observed yield for the given crop, with nodata converted to 0"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare
                )
            ]
        ),
        spec.TASKGRAPH_DIR
    ]
)


_INTERMEDIATE_OUTPUT_DIR = 'intermediate_output'

_YIELD_PERCENTILE_FIELD_PATTERN = 'yield_([^_]+)'
_GLOBAL_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    'observed_yield', '%s_yield_map.tif')  # crop_name
_EXTENDED_CLIMATE_BIN_FILE_PATTERN = os.path.join(
    'extended_climate_bin_maps', 'extendedclimatebins%s.tif')  # crop_name
_CLIMATE_PERCENTILE_TABLE_PATTERN = os.path.join(
    'climate_percentile_yield_tables',
    '%s_percentile_yield_table.csv')  # crop_name

# crop_name, yield_percentile_id
_INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_%s_interpolated_yield%s.tif')

# crop_name, file_suffix
_CLIPPED_CLIMATE_BIN_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR,
    'clipped_%s_climate_bin_map%s.tif')

# crop_name, yield_percentile_id, file_suffix
_COARSE_YIELD_PERCENTILE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_%s_coarse_yield%s.tif')

# crop_name, yield_percentile_id, file_suffix
_PERCENTILE_CROP_PRODUCTION_FILE_PATTERN = os.path.join(
    '.', '%s_%s_production%s.tif')

# crop_name, file_suffix
_CLIPPED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_clipped_observed_yield%s.tif')

# crop_name, file_suffix
_ZEROED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_zeroed_observed_yield%s.tif')

# crop_name, file_suffix
_INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_interpolated_observed_yield%s.tif')

# crop_name, file_suffix
_OBSERVED_PRODUCTION_FILE_PATTERN = os.path.join(
    '.', '%s_observed_production%s.tif')

# file_suffix
_AGGREGATE_VECTOR_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'aggregate_vector%s.shp')

# file_suffix
_AGGREGATE_TABLE_FILE_PATTERN = os.path.join(
    '.', 'aggregate_results%s.csv')

_EXPECTED_NUTRIENT_TABLE_HEADERS = list(nutrient_units.keys())
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
              args['model_data_path']/climate_bin_maps/[cropname]_*
              A ValueError is raised if strings don't match.

        args['aggregate_polygon_path'] (string): path to polygon shapefile
            that will be used to aggregate crop yields and total nutrient
            value. (optional, if value is None, then skipped)
        args['model_data_path'] (string): path to the InVEST Crop Production
            global data directory.  This model expects that the following
            directories are subdirectories of this path:

            * climate_bin_maps (contains [cropname]_climate_bin.tif files)
            * climate_percentile_yield (contains
              [cropname]_percentile_yield_table.csv files)

            Please see the InVEST user's guide chapter on crop production for
            details about how to download these data.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None.

    """
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

    bad_crop_name_list = []
    for crop_name in crop_to_landcover_df.index:
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'],
            _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)
        if not os.path.exists(crop_climate_bin_raster_path):
            bad_crop_name_list.append(crop_name)
    if bad_crop_name_list:
        raise ValueError(
            "The following crop names were provided in %s but no such crops "
            "exist for this model: %s" % (
                args['landcover_to_crop_table_path'], bad_crop_name_list))

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories([
        output_dir, os.path.join(output_dir, _INTERMEDIATE_OUTPUT_DIR)])

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

    # Initialize a TaskGraph
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Single process mode.
    task_graph = taskgraph.TaskGraph(
        os.path.join(output_dir, 'taskgraph_cache'), n_workers)
    dependent_task_list = []

    crop_lucode = None
    observed_yield_nodata = None
    for crop_name, row in crop_to_landcover_df.iterrows():
        crop_lucode = row[_EXPECTED_LUCODE_TABLE_HEADER]
        LOGGER.info("Processing crop %s", crop_name)
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'],
            _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)

        LOGGER.info(
            "Clipping global climate bin raster to landcover bounding box.")
        clipped_climate_bin_raster_path = os.path.join(
            output_dir, _CLIPPED_CLIMATE_BIN_FILE_PATTERN % (
                crop_name, file_suffix))
        crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
            crop_climate_bin_raster_path)
        crop_climate_bin_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(crop_climate_bin_raster_path,
                  crop_climate_bin_raster_info['pixel_size'],
                  clipped_climate_bin_raster_path, 'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[clipped_climate_bin_raster_path],
            task_name='crop_climate_bin')
        dependent_task_list.append(crop_climate_bin_task)

        climate_percentile_yield_table_path = os.path.join(
            args['model_data_path'],
            _CLIMATE_PERCENTILE_TABLE_PATTERN % crop_name)
        crop_climate_percentile_df = MODEL_SPEC.get_input(
            'model_data_path').get_contents(
            'climate_percentile_yield_tables').get_contents(
            '[CROP]_percentile_yield_table.csv').get_validated_dataframe(
            climate_percentile_yield_table_path)
        yield_percentile_headers = [
            x for x in crop_climate_percentile_df.columns if x != 'climate_bin']

        reclassify_error_details = {
            'raster_name': f'{crop_name} Climate Bin',
            'column_name': 'climate_bin',
            'table_name': f'Climate {crop_name} Percentile Yield'}
        for yield_percentile_id in yield_percentile_headers:
            LOGGER.info("Map %s to climate bins.", yield_percentile_id)
            interpolated_yield_percentile_raster_path = os.path.join(
                output_dir,
                _INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))
            bin_to_percentile_yield = (
                crop_climate_percentile_df[yield_percentile_id].to_dict())
            # reclassify nodata to a valid value of 0
            # we're assuming that the crop doesn't exist where there is no data
            # this is more likely than assuming the crop does exist, esp.
            # in the context of the provided climate bins map
            bin_to_percentile_yield[
                crop_climate_bin_raster_info['nodata'][0]] = 0
            coarse_yield_percentile_raster_path = os.path.join(
                output_dir,
                _COARSE_YIELD_PERCENTILE_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))
            create_coarse_yield_percentile_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=((clipped_climate_bin_raster_path, 1),
                      bin_to_percentile_yield,
                      coarse_yield_percentile_raster_path, gdal.GDT_Float32,
                      _NODATA_YIELD, reclassify_error_details),
                target_path_list=[coarse_yield_percentile_raster_path],
                dependent_task_list=[crop_climate_bin_task],
                task_name='create_coarse_yield_percentile_%s_%s' % (
                    crop_name, yield_percentile_id))
            dependent_task_list.append(create_coarse_yield_percentile_task)

            LOGGER.info(
                "Interpolate %s %s yield raster to landcover resolution.",
                crop_name, yield_percentile_id)
            create_interpolated_yield_percentile_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(coarse_yield_percentile_raster_path,
                      landcover_raster_info['pixel_size'],
                      interpolated_yield_percentile_raster_path, 'cubicspline'),
                kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                        'target_bb': landcover_raster_info['bounding_box']},
                target_path_list=[interpolated_yield_percentile_raster_path],
                dependent_task_list=[create_coarse_yield_percentile_task],
                task_name='create_interpolated_yield_percentile_%s_%s' % (
                    crop_name, yield_percentile_id))
            dependent_task_list.append(
                create_interpolated_yield_percentile_task)

            LOGGER.info(
                "Calculate yield for %s at %s", crop_name,
                yield_percentile_id)
            percentile_crop_production_raster_path = os.path.join(
                output_dir,
                _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))

            create_percentile_production_task = task_graph.add_task(
                func=calculate_crop_production,
                args=(
                    args['landcover_raster_path'],
                    interpolated_yield_percentile_raster_path,
                    crop_lucode,
                    percentile_crop_production_raster_path),
                target_path_list=[percentile_crop_production_raster_path],
                dependent_task_list=[
                    create_interpolated_yield_percentile_task],
                task_name='create_percentile_production_%s_%s' % (
                    crop_name, yield_percentile_id))
            dependent_task_list.append(create_percentile_production_task)

        LOGGER.info("Calculate observed yield for %s", crop_name)
        global_observed_yield_raster_path = os.path.join(
            args['model_data_path'],
            _GLOBAL_OBSERVED_YIELD_FILE_PATTERN % crop_name)
        global_observed_yield_raster_info = (
            pygeoprocessing.get_raster_info(
                global_observed_yield_raster_path))

        clipped_observed_yield_raster_path = os.path.join(
            output_dir, _CLIPPED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))
        clip_global_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(global_observed_yield_raster_path,
                  global_observed_yield_raster_info['pixel_size'],
                  clipped_observed_yield_raster_path, 'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[clipped_observed_yield_raster_path],
            task_name='clip_global_observed_yield_%s_' % crop_name)
        dependent_task_list.append(clip_global_observed_yield_task)

        observed_yield_nodata = (
            global_observed_yield_raster_info['nodata'][0])

        zeroed_observed_yield_raster_path = os.path.join(
            output_dir, _ZEROED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))

        nodata_to_zero_for_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(clipped_observed_yield_raster_path, 1),
                   (observed_yield_nodata, 'raw')],
                  _zero_observed_yield_op, zeroed_observed_yield_raster_path,
                  gdal.GDT_Float32, observed_yield_nodata),
            target_path_list=[zeroed_observed_yield_raster_path],
            dependent_task_list=[clip_global_observed_yield_task],
            task_name='nodata_to_zero_for_observed_yield_%s_' % crop_name)
        dependent_task_list.append(nodata_to_zero_for_observed_yield_task)

        interpolated_observed_yield_raster_path = os.path.join(
            output_dir, _INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))

        LOGGER.info(
            "Interpolating observed %s raster to landcover.", crop_name)
        interpolate_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(zeroed_observed_yield_raster_path,
                  landcover_raster_info['pixel_size'],
                  interpolated_observed_yield_raster_path, 'cubicspline'),
            kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                    'target_bb': landcover_raster_info['bounding_box']},
            target_path_list=[interpolated_observed_yield_raster_path],
            dependent_task_list=[nodata_to_zero_for_observed_yield_task],
            task_name='interpolate_observed_yield_to_lulc_%s' % crop_name)
        dependent_task_list.append(interpolate_observed_yield_task)

        observed_production_raster_path = os.path.join(
            output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))

        calculate_observed_production_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(args['landcover_raster_path'], 1),
                   (interpolated_observed_yield_raster_path, 1),
                   (observed_yield_nodata, 'raw'), (landcover_nodata, 'raw'),
                   (crop_lucode, 'raw')],
                  _mask_observed_yield_op, observed_production_raster_path,
                  gdal.GDT_Float32, observed_yield_nodata),
            target_path_list=[observed_production_raster_path],
            dependent_task_list=[interpolate_observed_yield_task],
            task_name='calculate_observed_production_%s' % crop_name)
        dependent_task_list.append(calculate_observed_production_task)

    # both 'crop_nutrient.csv' and 'crop' are known data/header values for
    # this model data.
    nutrient_df = MODEL_SPEC.get_input(
        'model_data_path').get_contents(
        'crop_nutrient.csv').get_validated_dataframe(
            os.path.join(args['model_data_path'], 'crop_nutrient.csv'))

    result_table_path = os.path.join(
        output_dir, 'result_table%s.csv' % file_suffix)

    crop_names = crop_to_landcover_df.index.to_list()
    _ = task_graph.add_task(
        func=tabulate_results,
        args=(nutrient_df, yield_percentile_headers,
              crop_names, pixel_area_ha,
              args['landcover_raster_path'], landcover_nodata,
              output_dir, file_suffix, result_table_path),
        target_path_list=[result_table_path],
        dependent_task_list=dependent_task_list,
        task_name='tabulate_results')

    if ('aggregate_polygon_path' in args and
            args['aggregate_polygon_path'] not in ['', None]):
        LOGGER.info("aggregating result over query polygon")
        target_aggregate_vector_path = os.path.join(
            output_dir, _AGGREGATE_VECTOR_FILE_PATTERN % (file_suffix))
        aggregate_results_table_path = os.path.join(
            output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)
        _ = task_graph.add_task(
            func=aggregate_to_polygons,
            args=(args['aggregate_polygon_path'],
                  target_aggregate_vector_path,
                  landcover_raster_info['projection_wkt'],
                  crop_names, nutrient_df,
                  yield_percentile_headers, pixel_area_ha, output_dir,
                  file_suffix, aggregate_results_table_path),
            target_path_list=[target_aggregate_vector_path,
                              aggregate_results_table_path],
            dependent_task_list=dependent_task_list,
            task_name='aggregate_results_to_polygons')

    task_graph.close()
    task_graph.join()


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
        nutrient_df, yield_percentile_headers,
        crop_names, pixel_area_ha, landcover_raster_path,
        landcover_nodata, output_dir, file_suffix, target_table_path):
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
        output_dir (string): the file path to the output workspace.
        file_suffix (string): string to append to any output filenames.
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
            'crop,area (ha),' + 'production_observed,' +
            ','.join(production_percentile_headers) + ',' + ','.join(
                nutrient_headers) + '\n')
        for crop_name in sorted(crop_names):
            result_table.write(crop_name)
            production_lookup = {}
            production_pixel_count = 0
            yield_sum = 0
            observed_production_raster_path = os.path.join(
                output_dir,
                _OBSERVED_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))

            LOGGER.info(
                "Calculating production area and summing observed yield.")
            observed_yield_nodata = pygeoprocessing.get_raster_info(
                observed_production_raster_path)['nodata'][0]
            for _, yield_block in pygeoprocessing.iterblocks(
                    (observed_production_raster_path, 1)):

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
                yield_percentile_raster_path = os.path.join(
                    output_dir,
                    _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
                yield_sum = 0
                for _, yield_block in pygeoprocessing.iterblocks(
                        (yield_percentile_raster_path, 1)):
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
        yield_percentile_headers, pixel_area_ha, output_dir, file_suffix,
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
        output_dir (string): the file path to the output workspace.
        file_suffix (string): string to append to any output filenames.
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
    total_nutrient_table = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(
            float)))
    for crop_name in crop_names:
        # convert 100g to Mg and fraction left over from refuse
        nutrient_factor = 1e4 * (
            1 - nutrient_df['percentrefuse'][crop_name] / 100)
        # loop over percentiles
        for yield_percentile_id in yield_percentile_headers:
            percentile_crop_production_raster_path = os.path.join(
                output_dir,
                _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                    crop_name, yield_percentile_id, file_suffix))
            LOGGER.info(
                "Calculating zonal stats for %s  %s", crop_name,
                yield_percentile_id)
            total_yield_lookup['%s_%s' % (
                crop_name, yield_percentile_id)] = (
                    pygeoprocessing.zonal_statistics(
                        (percentile_crop_production_raster_path, 1),
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
        observed_yield_path = os.path.join(
            output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))
        total_yield_lookup[f'{crop_name}_observed'] = (
            pygeoprocessing.zonal_statistics(
                (observed_yield_path, 1),
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
