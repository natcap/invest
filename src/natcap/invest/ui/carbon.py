# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import carbon


class Carbon(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['carbon'].model_title,
            target=carbon.execute,
            validator=carbon.validate,
            localdoc=MODEL_METADATA['carbon'].userguide)

        self.cur_lulc_raster = inputs.File(
            args_key='lulc_cur_path',
            helptext=(
                "A GDAL-supported raster representing the land-cover "
                "of the current scenario."),
            label='Current Land Use/Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.cur_lulc_raster)
        self.carbon_pools_path = inputs.File(
            args_key='carbon_pools_path',
            helptext=(
                "A table that maps the land-cover IDs to carbon "
                "pools.  The table must contain columns of 'LULC', "
                "'C_above', 'C_Below', 'C_Soil', 'C_Dead' as described "
                "in the User's Guide.  The values in LULC must at "
                "least include the LULC IDs in the land cover maps."),
            label='Carbon Pools',
            validator=self.validator)
        self.add_input(self.carbon_pools_path)
        self.calc_sequestration = inputs.Checkbox(
            helptext=(
                "Check to enable sequestration analysis. This "
                "requires inputs of Land Use/Land Cover maps for both "
                "current and future scenarios."),
            args_key='calc_sequestration',
            label='Calculate Sequestration')
        self.add_input(self.calc_sequestration)
        self.fut_lulc_raster = inputs.File(
            args_key='lulc_fut_path',
            helptext=(
                "A GDAL-supported raster representing the land-cover "
                "of the future scenario.  <br><br>If REDD scenario "
                "analysis is enabled, this should be the reference, or "
                "baseline, future scenario against which to compare "
                "the REDD policy scenario."),
            interactive=False,
            label='Future Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.fut_lulc_raster)
        self.redd = inputs.Checkbox(
            helptext=(
                "Check to enable REDD scenario analysis.  This "
                "requires three Land Use/Land Cover maps: one for the "
                "current scenario, one for the future baseline "
                "scenario, and one for the future REDD policy "
                "scenario."),
            interactive=False,
            args_key='do_redd',
            label='REDD Scenario Analysis')
        self.add_input(self.redd)
        self.redd_lulc_raster = inputs.File(
            args_key='lulc_redd_path',
            helptext=(
                "A GDAL-supported raster representing the land-cover "
                "of the REDD policy future scenario.  This scenario "
                "will be compared to the baseline future scenario."),
            interactive=False,
            label='REDD Policy (Raster)',
            validator=self.validator)
        self.add_input(self.redd_lulc_raster)
        self.valuation_container = inputs.Container(
            args_key='do_valuation',
            expandable=True,
            expanded=False,
            interactive=False,
            label='Run Valuation Model')
        self.add_input(self.valuation_container)
        self.cur_lulc_year = inputs.Text(
            args_key='lulc_cur_year',
            helptext='The calendar year of the current scenario.',
            interactive=False,
            label='Current Land Cover Calendar Year',
            validator=self.validator)
        self.valuation_container.add_input(self.cur_lulc_year)
        self.fut_lulc_year = inputs.Text(
            args_key='lulc_fut_year',
            helptext='The calendar year of the future scenario.',
            interactive=False,
            label='Future Land Cover Calendar Year',
            validator=self.validator)
        self.valuation_container.add_input(self.fut_lulc_year)
        self.price_per_metric_ton_of_c = inputs.Text(
            args_key='price_per_metric_ton_of_c',
            label='Price/Metric Ton of Carbon',
            validator=self.validator)
        self.valuation_container.add_input(self.price_per_metric_ton_of_c)
        self.discount_rate = inputs.Text(
            args_key='discount_rate',
            helptext='The discount rate as a floating point percent.',
            label='Market Discount in Price of Carbon (%)',
            validator=self.validator)
        self.valuation_container.add_input(self.discount_rate)
        self.rate_change = inputs.Text(
            args_key='rate_change',
            helptext=(
                "The floating point percent increase of the price of "
                "carbon per year."),
            label='Annual Rate of Change in Price of Carbon (%)',
            validator=self.validator)
        self.valuation_container.add_input(self.rate_change)

        # Set interactivity, requirement as input sufficiency changes
        self.calc_sequestration.sufficiency_changed.connect(
            self.fut_lulc_raster.set_interactive)
        self.calc_sequestration.sufficiency_changed.connect(
            self.redd.set_interactive)
        self.redd.sufficiency_changed.connect(
            self.redd_lulc_raster.set_interactive)
        self.calc_sequestration.sufficiency_changed.connect(
            self.valuation_container.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.cur_lulc_raster.args_key: self.cur_lulc_raster.value(),
            self.carbon_pools_path.args_key: self.carbon_pools_path.value(),
            self.valuation_container.args_key:
                self.valuation_container.value(),
            self.calc_sequestration.args_key:
                self.calc_sequestration.value(),
            self.redd.args_key: self.redd.value(),
        }

        if self.calc_sequestration.value():
            args[self.redd_lulc_raster.args_key] = (
                self.redd_lulc_raster.value())
            args[self.fut_lulc_raster.args_key] = self.fut_lulc_raster.value()

            for arg in (self.cur_lulc_year, self.fut_lulc_year):
                args[arg.args_key] = arg.value()

            # Attempt to cast valuation parameters to float
            if self.valuation_container.value():
                for arg in (self.price_per_metric_ton_of_c,
                            self.discount_rate, self.rate_change):
                    args[arg.args_key] = arg.value()

        return args
