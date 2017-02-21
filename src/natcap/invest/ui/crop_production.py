# coding=UTF-8

from natcap.invest.ui import model
from natcap.ui import inputs
from natcap.invest.crop_production import crop_production


class CropProduction(model.Model):
    label = u'Crop Production'
    target = staticmethod(crop_production.execute)
    validator = staticmethod(crop_production.validate)
    localdoc = u'../documentation/crop_production.html'

    def __init__(self):
        model.Model.__init__(self)

        self.lookup_table = inputs.File(
            args_key=u'lookup_table',
            helptext=(
                u"The table should contain three columns: a 'name' "
                u"column, a 'code' column, and an 'is_crop' column."),
            label=u'Lookup Table (CSV)',
            required=True,
            validator=self.validator)
        self.add_input(self.lookup_table)
        self.aoi_raster = inputs.File(
            args_key=u'aoi_raster',
            helptext=(
                u'A GDAL-supported raster representing a crop management '
                u'scenario.'),
            label=u'Crop Management Scenario Map (Raster)',
            required=True,
            validator=self.validator)
        self.add_input(self.aoi_raster)
        self.dataset_dir = inputs.Folder(
            args_key=u'dataset_dir',
            helptext=(
                u"The provided folder should contain a set of folders "
                u"and data specified in the 'Running the Model' section "
                u"of the model's User Guide."),
            label=u'Global Dataset Folder',
            required=True,
            validator=self.validator)
        self.add_input(self.dataset_dir)
        self.yield_function = inputs.Dropdown(
            args_key=u'yield_function',
            helptext=u'Determines how yield is estimated in the model.',
            label=u'Yield Function',
            options=[u'observed', u'percentile', u'regression'])
        self.add_input(self.yield_function)
        self.percentile_column = inputs.Text(
            args_key=u'percentile_column',
            helptext=(
                u"Required for Percentile Yield Function.  This input "
                u"is used to select the column of yield values from the "
                u"tables in the climate_percentile_yield folder of the "
                u"global dataset."),
            interactive=False,
            label=u'If Percentile Yield: Percentile Column',
            required=False,
            validator=self.validator)
        self.add_input(self.percentile_column)
        self.fertilizer_dir = inputs.Folder(
            args_key=u'fertilizer_dir',
            helptext=(
                u"Required for Regression Yield Function.  The folder "
                u"should contain three rasters: 'nitrogen.tif', "
                u"'potash.tif', and 'phosphorus.tif' representing the "
                u"kilograms of fertilizer applied per hectare to each "
                u"cell.  Please see the model's User Guide for more "
                u"details."),
            interactive=False,
            label=u'If Regression Yield: Fertilizer Raster Folder',
            required=False,
            validator=self.validator)
        self.add_input(self.fertilizer_dir)
        self.irrigation_raster = inputs.File(
            args_key=u'irrigation_raster',
            helptext=(
                u"Required for Regression Yield Function.  The raster "
                u"should contain 1s for cells that are irrigated and 0s "
                u"for cells that are rainfed.  Please see the model's "
                u"User Guide for more details."),
            interactive=False,
            label=u'If Regression Yield: Irrigation Map (Raster)',
            required=False,
            validator=self.validator)
        self.add_input(self.irrigation_raster)
        self.compute_nutritional_contents = inputs.Checkbox(
            args_key=u'compute_nutritional_contents',
            helptext=(
                u"If yes, a table of nutrient contents is generated "
                u"based on total yield of each crop and data provided "
                u"in table below."),
            label=u'Compute Nutrient Contents')
        self.add_input(self.compute_nutritional_contents)
        self.nutrient_table = inputs.File(
            args_key=u'nutrient_table',
            helptext=(
                u"A table containing data related to the nutrient "
                u"contents of each crop by weight.  Please see the "
                u"model's User Guide for more details."),
            interactive=False,
            label=u'Crop Nutrient Information (CSV)',
            required=True,
            validator=self.validator)
        self.add_input(self.nutrient_table)
        self.compute_financial_analysis = inputs.Checkbox(
            args_key=u'compute_financial_analysis',
            helptext=(
                u"If yes, a financial analysis table is generated "
                u"based on total yield of each crop, fertilizer "
                u"application rates, and data provided in table below."),
            label=u'Compute Financial Analysis')
        self.add_input(self.compute_financial_analysis)
        self.economics_table = inputs.File(
            args_key=u'economics_table',
            helptext=(
                u"A table containing data related to price and costs "
                u"associated with each crop.  Please see the model's "
                u"User Guide for more details."),
            interactive=False,
            label=u'Crop Economic Information (CSV)',
            required=True,
            validator=self.validator)
        self.add_input(self.economics_table)

        # Set interactivity, requirement as input sufficiency changes
        self.yield_function.sufficiency_changed.connect(
            self.percentile_column.set_interactive)
        self.yield_function.sufficiency_changed.connect(
            self.fertilizer_dir.set_interactive)
        self.yield_function.sufficiency_changed.connect(
            self.irrigation_raster.set_interactive)
        self.compute_nutritional_contents.sufficiency_changed.connect(
            self.nutrient_table.set_interactive)
        self.compute_financial_analysis.sufficiency_changed.connect(
            self.economics_table.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.lookup_table.args_key: self.lookup_table.value(),
            self.aoi_raster.args_key: self.aoi_raster.value(),
            self.dataset_dir.args_key: self.dataset_dir.value(),
            self.yield_function.args_key: self.yield_function.value(),
            self.percentile_column.args_key: self.percentile_column.value(),
            self.fertilizer_dir.args_key: self.fertilizer_dir.value(),
            self.irrigation_raster.args_key: self.irrigation_raster.value(),
            self.compute_nutritional_contents.args_key: (
                self.compute_nutritional_contents.value()),
            self.nutrient_table.args_key: self.nutrient_table.value(),
            self.compute_financial_analysis.args_key: (
                self.compute_financial_analysis.value()),
            self.economics_table.args_key: self.economics_table.value(),
        }

        return args
