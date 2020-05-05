# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.finfish_aquaculture import finfish_aquaculture

from osgeo import gdal


class FinfishAquaculture(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Marine Aquaculture: Finfish',
            target=finfish_aquaculture.execute,
            validator=finfish_aquaculture.validate,
            localdoc='marine_fish.html')

        self.farm_location = inputs.File(
            args_key='ff_farm_loc',
            helptext=(
                "An OGR-supported vector file containing polygon or "
                "point, with a latitude and longitude value and a "
                "numerical identifier for each farm.  File can be "
                "named anything, but no spaces in the "
                "name.<br><br>File type: polygon shapefile or "
                ".gdb<br>Rows: each row is a specific netpen or entire "
                "aquaculture farm<br>Columns: columns contain "
                "attributes about each netpen (area, location, "
                "etc.).<br>Sample data set: "
                "\InVEST\Aquaculture\Input\Finfish_Netpens.shp"),
            label='Finfish Farm Location (Vector)',
            validator=self.validator)
        self.add_input(self.farm_location)
        self.farm_identifier = inputs.Dropdown(
            args_key='farm_ID',
            helptext=(
                "The name of a column heading used to identify each "
                "farm and link the spatial information from the "
                "shapefile to subsequent table input data (farm "
                "operation and daily water temperature at farm "
                "tables). Additionally, the numbers underneath this "
                "farm identifier name must be unique integers for all "
                "the inputs."),
            interactive=False,
            options=('UNKNOWN',),  # No options until valid OGR vector provided
            label='Farm Identifier Name')
        self.add_input(self.farm_identifier)
        self.param_a = inputs.Text(
            args_key='g_param_a',
            helptext=(
                "Default a  = (0.038 g/day). If the user chooses to "
                "adjust these parameters, we recommend using them in "
                "the simple growth model to determine if the time "
                "taken for a fish to reach a target harvest weight "
                "typical for the region of interest is accurate."),
            label='Fish Growth Parameter (a)',
            validator=self.validator)
        self.add_input(self.param_a)
        self.param_b = inputs.Text(
            args_key='g_param_b',
            helptext=(
                "Default b  = (0.6667 g/day). If the user chooses to "
                "adjust these parameters, we recommend using them in "
                "the simple growth model to determine if the time "
                "taken for a fish to reach a target harvest weight "
                "typical for the region of interest is accurate."),
            label='Fish Growth Parameter (b)',
            validator=self.validator)
        self.add_input(self.param_b)
        self.param_tau = inputs.Text(
            args_key='g_param_tau',
            helptext=(
                "Default tau = (0.08 C^-1).  Specifies how sensitive "
                "finfish growth is to temperature.  If the user "
                "chooses to adjust these parameters, we recommend "
                "using them in the simple growth model to determine if "
                "the time taken for a fish to reach a target harvest "
                "weight typical for the region of interest is "
                "accurate."),
            label='Fish Growth Parameter (tau)',
            validator=self.validator)
        self.add_input(self.param_tau)
        self.uncertainty_data_container = inputs.Container(
            args_key='use_uncertainty',
            expandable=True,
            label='Enable Uncertainty Analysis')
        self.add_input(self.uncertainty_data_container)
        self.param_a_sd = inputs.Text(
            args_key='g_param_a_sd',
            helptext=(
                "Standard deviation for fish growth parameter a. "
                "This indicates the level of uncertainty in the "
                "estimate for parameter a."),
            label='Standard Deviation for Parameter (a)',
            validator=self.validator)
        self.uncertainty_data_container.add_input(self.param_a_sd)
        self.param_b_sd = inputs.Text(
            args_key='g_param_b_sd',
            helptext=(
                "Standard deviation for fish growth parameter b. "
                "This indicates the level of uncertainty in the "
                "estimate for parameter b."),
            label='Standard Deviation for Parameter (b)',
            validator=self.validator)
        self.uncertainty_data_container.add_input(self.param_b_sd)
        self.num_monte_carlo_runs = inputs.Text(
            args_key='num_monte_carlo_runs',
            helptext=(
                "Number of runs of the model to perform as part of a "
                "Monte Carlo simulation.  A larger number will tend to "
                "produce more consistent and reliable output, but will "
                "also take longer to run."),
            label='Number of Monte Carlo Simulation Runs',
            validator=self.validator)
        self.uncertainty_data_container.add_input(self.num_monte_carlo_runs)
        self.water_temperature = inputs.File(
            args_key='water_temp_tbl',
            helptext=(
                "Users must provide a time series of daily water "
                "temperature (C) for each farm in the shapefile.  When "
                "daily temperatures are not available, users can "
                "interpolate seasonal or monthly temperatures to a "
                "daily resolution.  Water temperatures collected at "
                "existing aquaculture facilities are preferable, but "
                "if unavailable, users can consult online sources such "
                "as NOAAs 4 km AVHRR Pathfinder Data and Canadas "
                "Department of Fisheries and Oceans Oceanographic "
                "Database.  The most appropriate temperatures to use "
                "are those from the upper portion of the water column, "
                "which are the temperatures experienced by the fish in "
                "the netpens."),
            label='Table of Daily Water Temperature at Farm (CSV)',
            validator=self.validator)
        self.add_input(self.water_temperature)
        self.farm_operations = inputs.File(
            args_key='farm_op_tbl',
            helptext=(
                "A table of general and farm-specific operations "
                "parameters.  Please refer to the sample data table "
                "for reference to ensure correct incorporation of data "
                "in the model.<br><br>The values for 'farm operations' "
                "(applied to all farms) and 'add new farms' (beginning "
                "with row 32) may be modified according to the user's "
                "needs . However, the location of cells in this "
                "template must not be modified.  If for example, if "
                "the model is to run for three farms only, the farms "
                "should be listed in rows 10, 11 and 12 (farms 1, 2, "
                "and 3, respectively). Several default values that are "
                "applicable to Atlantic salmon farming in British "
                "Columbia are also included in the sample data table."),
            label='Farm Operations Table (CSV)',
            validator=self.validator)
        self.add_input(self.farm_operations)
        self.outplant_buffer = inputs.Text(
            args_key='outplant_buffer',
            helptext=(
                "This value will allow the outplant start day to "
                "start plus or minus the number of days specified "
                "here."),
            label='Outplant Date Buffer',
            validator=self.validator)
        self.add_input(self.outplant_buffer)
        self.valuation = inputs.Checkbox(
            args_key='do_valuation',
            helptext=(
                "By checking this box, a valuation analysis will be "
                "run on the model."),
            label='Run Valuation?')
        self.add_input(self.valuation)
        self.market_price = inputs.Text(
            args_key='p_per_kg',
            helptext=(
                "Default value comes from Urner-Berry monthly fresh "
                "sheet reports on price of farmed Atlantic salmon."),
            interactive=False,
            label='Market Price per Kilogram of Processed Fish',
            validator=self.validator)
        self.add_input(self.market_price)
        self.fraction_price = inputs.Text(
            args_key='frac_p',
            helptext=(
                "Fraction of market price that accounts for costs "
                "rather than profit.  Default value is 0.3 (30%)."),
            interactive=False,
            label='Fraction of Price that Accounts to Costs',
            validator=self.validator)
        self.add_input(self.fraction_price)
        self.discount_rate = inputs.Text(
            args_key='discount',
            helptext=(
                "We use a 7% annual discount rate, adjusted to a "
                "daily rate of 0.000192 for 0.0192% (7%/365 days)."),
            interactive=False,
            label='Daily Market Discount Rate',
            validator=self.validator)
        self.add_input(self.discount_rate)

        # Set interactivity, requirement as input sufficiency changes
        self.farm_location.sufficiency_changed.connect(
            self.farm_identifier.set_interactive)
        self.farm_location.sufficiency_changed.connect(
            self._load_colnames)
        self.valuation.sufficiency_changed.connect(
            self.market_price.set_interactive)
        self.valuation.sufficiency_changed.connect(
            self.fraction_price.set_interactive)
        self.valuation.sufficiency_changed.connect(
            self.discount_rate.set_interactive)

    def _load_colnames(self, new_interactivity):
        if new_interactivity:
            new_vector_path = self.farm_location.value()
            new_vector = gdal.OpenEx(new_vector_path, gdal.OF_VECTOR)
            new_layer = new_vector.GetLayer()
            colnames = [defn.GetName() for defn in new_layer.schema]
            self.farm_identifier.set_options(colnames)
            self.farm_identifier.set_interactive(True)
        elif not self.farm_location.sufficient:
            self.farm_identifier.set_options([])

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.farm_location.args_key: self.farm_location.value(),
            self.farm_identifier.args_key: self.farm_identifier.value(),
            self.param_a.args_key: self.param_a.value(),
            self.param_b.args_key: self.param_b.value(),
            self.param_tau.args_key: self.param_tau.value(),
            self.uncertainty_data_container.args_key: (
                self.uncertainty_data_container.value()),
            self.water_temperature.args_key: self.water_temperature.value(),
            self.farm_operations.args_key: self.farm_operations.value(),
            self.outplant_buffer.args_key: self.outplant_buffer.value(),
            self.valuation.args_key: self.valuation.value(),
            self.market_price.args_key: self.market_price.value(),
            self.fraction_price.args_key: self.fraction_price.value(),
            self.discount_rate.args_key: self.discount_rate.value(),
        }

        if self.uncertainty_data_container.value():
            args[self.param_a_sd.args_key] = self.param_a_sd.value()
            args[self.param_b_sd.args_key] = self.param_b_sd.value()
            args[self.num_monte_carlo_runs.args_key] = (
                self.num_monte_carlo_runs.value())

        return args
