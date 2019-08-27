# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.carbon


class Carbon(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(self,
                                   label=u'InVEST Carbon Model',
                                   target=natcap.invest.carbon.execute,
                                   validator=natcap.invest.carbon.validate,
                                   localdoc=u'carbonstorage.html')

        self.cur_lulc_raster = inputs.File(
            args_key=u'lulc_cur_path',
            helptext=(
                u"A GDAL-supported raster representing the land-cover "
                u"of the current scenario."),
            label=u'Current Land Use/Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.cur_lulc_raster)
        self.carbon_pools_path = inputs.File(
            args_key=u'carbon_pools_path',
            helptext=(
                u"A table that maps the land-cover IDs to carbon "
                u"pools.  The table must contain columns of 'LULC', "
                u"'C_above', 'C_Below', 'C_Soil', 'C_Dead' as described "
                u"in the User's Guide.  The values in LULC must at "
                u"least include the LULC IDs in the land cover maps."),
            label=u'Carbon Pools',
            validator=self.validator)
        self.add_input(self.carbon_pools_path)
        self.cur_lulc_year = inputs.Text(
            args_key=u'lulc_cur_year',
            helptext=u'The calendar year of the current scenario.',
            interactive=False,
            label=u'Current Landcover Calendar Year',
            validator=self.validator)
        self.add_input(self.cur_lulc_year)
        self.calc_sequestration = inputs.Checkbox(
            helptext=(
                u"Check to enable sequestration analysis.  This "
                u"requires inputs of Land Use/Land Cover maps for both "
                u"current and future scenarios."),
            args_key='calc_sequestration',
            label=u'Calculate Sequestration')
        self.add_input(self.calc_sequestration)
        self.fut_lulc_raster = inputs.File(
            args_key=u'lulc_fut_path',
            helptext=(
                u"A GDAL-supported raster representing the land-cover "
                u"of the future scenario.  <br><br>If REDD scenario "
                u"analysis is enabled, this should be the reference, or "
                u"baseline, future scenario against which to compare "
                u"the REDD policy scenario."),
            interactive=False,
            label=u'Future Landcover (Raster)',
            validator=self.validator)
        self.add_input(self.fut_lulc_raster)
        self.fut_lulc_year = inputs.Text(
            args_key=u'lulc_fut_year',
            helptext=u'The calendar year of the future scenario.',
            interactive=False,
            label=u'Future Landcover Calendar Year',
            validator=self.validator)
        self.add_input(self.fut_lulc_year)
        self.redd = inputs.Checkbox(
            helptext=(
                u"Check to enable REDD scenario analysis.  This "
                u"requires three Land Use/Land Cover maps: one for the "
                u"current scenario, one for the future baseline "
                u"scenario, and one for the future REDD policy "
                u"scenario."),
            interactive=False,
            args_key='do_redd',
            label=u'REDD Scenario Analysis')
        self.add_input(self.redd)
        self.redd_lulc_raster = inputs.File(
            args_key=u'lulc_redd_path',
            helptext=(
                u"A GDAL-supported raster representing the land-cover "
                u"of the REDD policy future scenario.  This scenario "
                u"will be compared to the baseline future scenario."),
            interactive=False,
            label=u'REDD Policy (Raster)',
            validator=self.validator)
        self.add_input(self.redd_lulc_raster)
        self.valuation_container = inputs.Container(
            args_key=u'do_valuation',
            expandable=True,
            expanded=False,
            interactive=False,
            label=u'Run Valuation Model')
        self.add_input(self.valuation_container)
        self.price_per_metric_ton_of_c = inputs.Text(
            args_key=u'price_per_metric_ton_of_c',
            label=u'Price/Metric ton of carbon',
            validator=self.validator)
        self.valuation_container.add_input(self.price_per_metric_ton_of_c)
        self.discount_rate = inputs.Text(
            args_key=u'discount_rate',
            helptext=u'The discount rate as a floating point percent.',
            label=u'Market Discount in Price of Carbon (%)',
            validator=self.validator)
        self.valuation_container.add_input(self.discount_rate)
        self.rate_change = inputs.Text(
            args_key=u'rate_change',
            helptext=(
                u"The floating point percent increase of the price of "
                u"carbon per year."),
            label=u'Annual Rate of Change in Price of Carbon (%)',
            validator=self.validator)
        self.valuation_container.add_input(self.rate_change)

        # Set interactivity, requirement as input sufficiency changes
        self.calc_sequestration.sufficiency_changed.connect(
            self.cur_lulc_year.set_interactive)
        self.calc_sequestration.sufficiency_changed.connect(
            self.fut_lulc_raster.set_interactive)
        self.calc_sequestration.sufficiency_changed.connect(
            self.fut_lulc_year.set_interactive)
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
