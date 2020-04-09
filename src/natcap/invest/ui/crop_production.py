# coding=UTF-8
from natcap.invest.ui import model, inputs

import natcap.invest.crop_production_percentile
import natcap.invest.crop_production_regression


class CropProductionPercentile(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Crop Production Percentile Model',
            target=natcap.invest.crop_production_percentile.execute,
            validator=natcap.invest.crop_production_percentile.validate,
            localdoc='crop_production.html')

        self.model_data_path = inputs.Folder(
            args_key='model_data_path',
            helptext=(
                "A path to the InVEST Crop Production Data directory. "
                "These data would have been included with the InVEST "
                "installer if selected, or can be manually downloaded "
                "from http://data.naturalcapitalproject.org/invest- "
                "data/.  If downloaded with InVEST, the default value "
                "should be used.</b>"),
            label='Directory to model data',
            validator=self.validator)
        self.add_input(self.model_data_path)
        self.landcover_raster_path = inputs.File(
            args_key='landcover_raster_path',
            helptext=(
                "A raster file, representing integer land use/land "
                "code covers for each cell. This raster should have"
                "a projected coordinate system with units of meters "
                "(e.g. UTM) because pixel areas are divided by 10000"
                "in order to report some results in hectares."),
            label='Land-Use/Land-Cover Map (raster)',
            validator=self.validator)
        self.add_input(self.landcover_raster_path)
        self.landcover_to_crop_table_path = inputs.File(
            args_key='landcover_to_crop_table_path',
            helptext=(
                "A CSV table mapping canonical crop names to land use "
                "codes contained in the landcover/use raster.   The "
                "allowed crop names are abaca, agave, alfalfa, almond, "
                "aniseetc, apple, apricot, areca, artichoke, "
                "asparagus, avocado, bambara, banana, barley, bean, "
                "beetfor, berrynes, blueberry, brazil, broadbean, "
                "buckwheat, cabbage, cabbagefor, canaryseed, carob, "
                "carrot, carrotfor, cashew, cashewapple, cassava, "
                "castor, cauliflower, cerealnes, cherry, chestnut, "
                "chickpea, chicory, chilleetc, cinnamon, citrusnes, "
                "clove, clover, cocoa, coconut, coffee, cotton, "
                "cowpea, cranberry, cucumberetc, currant, date, "
                "eggplant, fibrenes, fig, flax, fonio, fornes, "
                "fruitnes, garlic, ginger, gooseberry, grape, "
                "grapefruitetc, grassnes, greenbean, greenbroadbean, "
                "greencorn, greenonion, greenpea, groundnut, hazelnut, "
                "hemp, hempseed, hop, jute, jutelikefiber, kapokfiber, "
                "kapokseed, karite, kiwi, kolanut, legumenes, "
                "lemonlime, lentil, lettuce, linseed, lupin, maize, "
                "maizefor, mango, mate, melonetc, melonseed, millet, "
                "mixedgrain, mixedgrass, mushroom, mustard, nutmeg, "
                "nutnes, oats, oilpalm, oilseedfor, oilseednes, okra, "
                "olive, onion, orange, papaya, pea, peachetc, pear, "
                "pepper, peppermint, persimmon, pigeonpea, pimento, "
                "pineapple, pistachio, plantain, plum, poppy, potato, "
                "pulsenes, pumpkinetc, pyrethrum, quince, quinoa, "
                "ramie, rapeseed, rasberry, rice, rootnes, rubber, "
                "rye, ryefor, safflower, sesame, sisal, sorghum, "
                "sorghumfor, sourcherry, soybean, spicenes, spinach, "
                "stonefruitnes, strawberry, stringbean, sugarbeet, "
                "sugarcane, sugarnes, sunflower, swedefor, "
                "sweetpotato, tangetc, taro, tea, tobacco, tomato, "
                "triticale, tropicalnes, tung, turnipfor, vanilla, "
                "vegetablenes, vegfor, vetch, walnut, watermelon, "
                "wheat, yam, and yautia."),
            label='Landcover to Crop Table (csv)',
            validator=self.validator)
        self.add_input(self.landcover_to_crop_table_path)
        self.aggregate_polygon_path = inputs.File(
            args_key='aggregate_polygon_path',
            helptext=(
                "A polygon shapefile containing features with"
                "which to aggregate/summarize final results."
                "It is fine to have overlapping polygons."),
            label='Aggregate results polygon (vector) (optional)',
            validator=self.validator)
        self.add_input(self.aggregate_polygon_path)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.model_data_path.args_key: self.model_data_path.value(),
            self.landcover_raster_path.args_key:
                self.landcover_raster_path.value(),
            self.landcover_to_crop_table_path.args_key:
                self.landcover_to_crop_table_path.value(),
            self.aggregate_polygon_path.args_key:
                self.aggregate_polygon_path.value(),
        }
        return args


class CropProductionRegression(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Crop Production Regression Model',
            target=natcap.invest.crop_production_regression.execute,
            validator=natcap.invest.crop_production_regression.validate,
            localdoc='crop_production.html')

        self.model_data_path = inputs.Folder(
            args_key='model_data_path',
            helptext=(
                "A path to the InVEST Crop Production Data directory. "
                "These data would have been included with the InVEST "
                "installer if selected, or can be manually downloaded "
                "from http://data.naturalcapitalproject.org/invest- "
                "data/.  If downloaded with InVEST, the default value "
                "should be used.</b>"),
            label='Directory to model data',
            validator=self.validator)
        self.add_input(self.model_data_path)
        self.landcover_raster_path = inputs.File(
            args_key='landcover_raster_path',
            helptext=(
                "A raster file, representing integer land use/land "
                "code covers for each cell. This raster should have"
                "a projected coordinate system with units of meters "
                "(e.g. UTM) because pixel areas are divided by 10000"
                "in order to report some results in hectares."),
            label='Land-Use/Land-Cover Map (raster)',
            validator=self.validator)
        self.add_input(self.landcover_raster_path)
        self.landcover_to_crop_table_path = inputs.File(
            args_key='landcover_to_crop_table_path',
            helptext=(
                "A CSV table mapping canonical crop names to land use "
                "codes contained in the landcover/use raster.   The "
                "allowed crop names are barley, maize, oilpalm, "
                "potato, rice, soybean, sugarbeet, sugarcane, "
                "sunflower, and wheat."),
            label='Landcover to Crop Table (csv)',
            validator=self.validator)
        self.add_input(self.landcover_to_crop_table_path)
        self.fertilization_rate_table_path = inputs.File(
            args_key='fertilization_rate_table_path',
            helptext=(
                "A table that maps fertilization rates to crops in "
                "the simulation.  Must include the headers "
                "'crop_name', 'nitrogen_rate',  'phosphorous_rate', "
                "and 'potassium_rate'."),
            label='Fertilization Rate Table Path (csv)',
            validator=self.validator)
        self.add_input(self.fertilization_rate_table_path)
        self.aggregate_polygon_path = inputs.File(
            args_key='aggregate_polygon_path',
            helptext=(
                "A polygon shapefile containing features with"
                "which to aggregate/summarize final results."
                "It is fine to have overlapping polygons."),
            label='Aggregate results polygon (vector) (optional)',
            validator=self.validator)
        self.add_input(self.aggregate_polygon_path)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.model_data_path.args_key: self.model_data_path.value(),
            self.landcover_raster_path.args_key:
                self.landcover_raster_path.value(),
            self.landcover_to_crop_table_path.args_key:
                self.landcover_to_crop_table_path.value(),
            self.fertilization_rate_table_path.args_key:
                self.fertilization_rate_table_path.value(),
            self.aggregate_polygon_path.args_key:
                self.aggregate_polygon_path.value(),
        }
        return args
