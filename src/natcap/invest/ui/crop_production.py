# coding=UTF-8
from natcap.invest.ui import model, inputs

import natcap.invest.crop_production_percentile
import natcap.invest.crop_production_regression


class CropProductionPercentile(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Crop Production Percentile Model',
            target=natcap.invest.crop_production_percentile.execute,
            validator=natcap.invest.crop_production_percentile.validate,
            localdoc=u'crop_production.html')

        self.model_data_path = inputs.Folder(
            args_key=u'model_data_path',
            helptext=(
                u"A path to the InVEST Crop Production Data directory. "
                u"These data would have been included with the InVEST "
                u"installer if selected, or can be manually downloaded "
                u"from http://data.naturalcapitalproject.org/invest- "
                u"data/.  If downloaded with InVEST, the default value "
                u"should be used.</b>"),
            label=u'Directory to model data',
            validator=self.validator)
        self.add_input(self.model_data_path)
        self.landcover_raster_path = inputs.File(
            args_key=u'landcover_raster_path',
            helptext=(
                u"A raster file, representing integer land use/land "
                u"code covers for each cell. This raster should have"
                u"a projected coordinate system with units of meters "
                u"(e.g. UTM) because pixel areas are divided by 10000"
                u"in order to report some results in hectares."),
            label=u'Land-Use/Land-Cover Map (raster)',
            validator=self.validator)
        self.add_input(self.landcover_raster_path)
        self.landcover_to_crop_table_path = inputs.File(
            args_key=u'landcover_to_crop_table_path',
            helptext=(
                u"A CSV table mapping canonical crop names to land use "
                u"codes contained in the landcover/use raster.   The "
                u"allowed crop names are abaca, agave, alfalfa, almond, "
                u"aniseetc, apple, apricot, areca, artichoke, "
                u"asparagus, avocado, bambara, banana, barley, bean, "
                u"beetfor, berrynes, blueberry, brazil, broadbean, "
                u"buckwheat, cabbage, cabbagefor, canaryseed, carob, "
                u"carrot, carrotfor, cashew, cashewapple, cassava, "
                u"castor, cauliflower, cerealnes, cherry, chestnut, "
                u"chickpea, chicory, chilleetc, cinnamon, citrusnes, "
                u"clove, clover, cocoa, coconut, coffee, cotton, "
                u"cowpea, cranberry, cucumberetc, currant, date, "
                u"eggplant, fibrenes, fig, flax, fonio, fornes, "
                u"fruitnes, garlic, ginger, gooseberry, grape, "
                u"grapefruitetc, grassnes, greenbean, greenbroadbean, "
                u"greencorn, greenonion, greenpea, groundnut, hazelnut, "
                u"hemp, hempseed, hop, jute, jutelikefiber, kapokfiber, "
                u"kapokseed, karite, kiwi, kolanut, legumenes, "
                u"lemonlime, lentil, lettuce, linseed, lupin, maize, "
                u"maizefor, mango, mate, melonetc, melonseed, millet, "
                u"mixedgrain, mixedgrass, mushroom, mustard, nutmeg, "
                u"nutnes, oats, oilpalm, oilseedfor, oilseednes, okra, "
                u"olive, onion, orange, papaya, pea, peachetc, pear, "
                u"pepper, peppermint, persimmon, pigeonpea, pimento, "
                u"pineapple, pistachio, plantain, plum, poppy, potato, "
                u"pulsenes, pumpkinetc, pyrethrum, quince, quinoa, "
                u"ramie, rapeseed, rasberry, rice, rootnes, rubber, "
                u"rye, ryefor, safflower, sesame, sisal, sorghum, "
                u"sorghumfor, sourcherry, soybean, spicenes, spinach, "
                u"stonefruitnes, strawberry, stringbean, sugarbeet, "
                u"sugarcane, sugarnes, sunflower, swedefor, "
                u"sweetpotato, tangetc, taro, tea, tobacco, tomato, "
                u"triticale, tropicalnes, tung, turnipfor, vanilla, "
                u"vegetablenes, vegfor, vetch, walnut, watermelon, "
                u"wheat, yam, and yautia."),
            label=u'Landcover to Crop Table (csv)',
            validator=self.validator)
        self.add_input(self.landcover_to_crop_table_path)
        self.aggregate_polygon_path = inputs.File(
            args_key=u'aggregate_polygon_path',
            helptext=(
                u"A polygon shapefile containing features with"
                u"which to aggregate/summarize final results."
                u"It is fine to have overlapping polygons."),
            label=u'Aggregate results polygon (vector) (optional)',
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
            label=u'Crop Production Regression Model',
            target=natcap.invest.crop_production_regression.execute,
            validator=natcap.invest.crop_production_regression.validate,
            localdoc=u'crop_production.html')

        self.model_data_path = inputs.Folder(
            args_key=u'model_data_path',
            helptext=(
                u"A path to the InVEST Crop Production Data directory. "
                u"These data would have been included with the InVEST "
                u"installer if selected, or can be manually downloaded "
                u"from http://data.naturalcapitalproject.org/invest- "
                u"data/.  If downloaded with InVEST, the default value "
                u"should be used.</b>"),
            label=u'Directory to model data',
            validator=self.validator)
        self.add_input(self.model_data_path)
        self.landcover_raster_path = inputs.File(
            args_key=u'landcover_raster_path',
            helptext=(
                u"A raster file, representing integer land use/land "
                u"code covers for each cell. This raster should have"
                u"a projected coordinate system with units of meters "
                u"(e.g. UTM) because pixel areas are divided by 10000"
                u"in order to report some results in hectares."),
            label=u'Land-Use/Land-Cover Map (raster)',
            validator=self.validator)
        self.add_input(self.landcover_raster_path)
        self.landcover_to_crop_table_path = inputs.File(
            args_key=u'landcover_to_crop_table_path',
            helptext=(
                u"A CSV table mapping canonical crop names to land use "
                u"codes contained in the landcover/use raster.   The "
                u"allowed crop names are barley, maize, oilpalm, "
                u"potato, rice, soybean, sugarbeet, sugarcane, "
                u"sunflower, and wheat."),
            label=u'Landcover to Crop Table (csv)',
            validator=self.validator)
        self.add_input(self.landcover_to_crop_table_path)
        self.fertilization_rate_table_path = inputs.File(
            args_key=u'fertilization_rate_table_path',
            helptext=(
                u"A table that maps fertilization rates to crops in "
                u"the simulation.  Must include the headers "
                u"'crop_name', 'nitrogen_rate',  'phosphorous_rate', "
                u"and 'potassium_rate'."),
            label=u'Fertilization Rate Table Path (csv)',
            validator=self.validator)
        self.add_input(self.fertilization_rate_table_path)
        self.aggregate_polygon_path = inputs.File(
            args_key=u'aggregate_polygon_path',
            helptext=(
                u"A polygon shapefile containing features with"
                u"which to aggregate/summarize final results."
                u"It is fine to have overlapping polygons."),
            label=u'Aggregate results polygon (vector) (optional)',
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
