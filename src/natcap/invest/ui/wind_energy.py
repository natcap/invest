# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest import wind_energy


class WindEnergy(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Wind Energy',
            target=wind_energy.execute,
            validator=wind_energy.validate,
            localdoc='wind_energy.html',
            suffix_args_key='suffix'
        )

        self.wind_data = inputs.File(
            args_key='wind_data_path',
            helptext=(
                "A CSV file that represents the wind input data "
                "(Weibull parameters). Please see the User's Guide for "
                "a more detailed description of the parameters."),
            label='Wind Data Points (CSV)',
            validator=self.validator)
        self.add_input(self.wind_data)
        self.aoi = inputs.File(
            args_key='aoi_vector_path',
            helptext=(
                "Optional.  An OGR-supported vector file containing a "
                "single polygon defining the area of interest.  The "
                "AOI must be projected with linear units equal to "
                "meters.  If the AOI is provided it will clip and "
                "project the outputs to that of the AOI. The Distance "
                "inputs are dependent on the AOI and will only be "
                "accessible if the AOI is selected.  If the AOI is "
                "selected and the Distance parameters are selected, "
                "then the AOI should also cover a portion of the land "
                "polygon to calculate distances correctly.  An AOI is "
                "required for valuation."),
            label='Area Of Interest (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.aoi)
        self.bathymetry = inputs.File(
            args_key='bathymetry_path',
            helptext=(
                "A GDAL-supported raster file containing elevation "
                "values represented in meters for the area of "
                "interest.  The DEM should cover at least the entire "
                "span of the area of interest and if no AOI is "
                "provided then the default global DEM should be used."),
            label='Bathymetric Digital Elevation Model (Raster)',
            validator=self.validator)
        self.add_input(self.bathymetry)
        self.land_polygon = inputs.File(
            args_key='land_polygon_vector_path',
            helptext=(
                "An OGR-supported polygon vector that represents the "
                "land and coastline that is of interest.  For this "
                "input to be selectable the AOI must be selected.  The "
                "AOI should also cover a portion of this land polygon "
                "to properly calculate distances.  This coastal "
                "polygon, and the area covered by the AOI, form the "
                "basis for distance calculations for wind farm "
                "electrical transmission.  This input is required for "
                "masking by distance values and for valuation."),
            interactive=False,
            label='Land Polygon for Distance Calculation (Vector)',
            validator=self.validator)
        self.add_input(self.land_polygon)
        self.global_wind_parameters = inputs.File(
            args_key='global_wind_parameters_path',
            helptext=(
                "A CSV file that holds wind energy model parameters "
                "for both the biophysical and valuation modules. "
                "These parameters are defaulted to values that are "
                "supported and reviewed in the User's Guide.  It is "
                "recommended that careful consideration be taken "
                "before changing these values and to make a new CSV "
                "file so that the default one always remains."),
            label='Global Wind Energy Parameters (CSV)',
            validator=self.validator)
        self.add_input(self.global_wind_parameters)
        self.turbine_group = inputs.Container(
            label='Turbine Properties')
        self.add_input(self.turbine_group)
        self.turbine_parameters = inputs.File(
            args_key='turbine_parameters_path',
            helptext=(
                "A CSV file that contains parameters corresponding to "
                "a specific turbine type.  The InVEST package comes "
                "with two turbine model options, 3.6 MW and 5.0 MW. A "
                "new turbine class may be created by using the "
                "existing file format conventions and filling in new "
                "parameters.  Likewise an existing class may be "
                "modified according to the user's needs.  It is "
                "recommended that the existing default CSV files are "
                "not overwritten."),
            label='Turbine Type Parameters File (CSV)',
            validator=self.validator)
        self.turbine_group.add_input(self.turbine_parameters)
        self.number_of_machines = inputs.Text(
            args_key='number_of_turbines',
            helptext=(
                "An integer value indicating the number of wind "
                "turbines per wind farm."),
            label='Number Of Turbines',
            validator=self.validator)
        self.turbine_group.add_input(self.number_of_machines)
        self.min_depth = inputs.Text(
            args_key='min_depth',
            helptext=(
                "A floating point value in meters for the minimum "
                "depth of the offshore wind farm installation."),
            label='Minimum Depth for Offshore Wind Farm Installation (meters)',
            validator=self.validator)
        self.turbine_group.add_input(self.min_depth)
        self.max_depth = inputs.Text(
            args_key='max_depth',
            helptext=(
                "A floating point value in meters for the maximum "
                "depth of the offshore wind farm installation."),
            label='Maximum Depth for Offshore Wind Farm Installation (meters)',
            validator=self.validator)
        self.turbine_group.add_input(self.max_depth)
        self.min_distance = inputs.Text(
            args_key='min_distance',
            helptext=(
                "A floating point value in meters that represents the "
                "minimum distance from shore for offshore wind farm "
                "installation.  Required for valuation."),
            interactive=False,
            label=(
                'Minimum Distance for Offshore Wind Farm Installation '
                '(meters)'),
            validator=self.validator)
        self.turbine_group.add_input(self.min_distance)
        self.max_distance = inputs.Text(
            args_key='max_distance',
            helptext=(
                "A floating point value in meters that represents the "
                "maximum distance from shore for offshore wind farm "
                "installation.  Required for valuation."),
            interactive=False,
            label=(
                'Maximum Distance for Offshore Wind Farm Installation '
                '(meters)'),
            validator=self.validator)
        self.turbine_group.add_input(self.max_distance)
        self.valuation_container = inputs.Container(
            args_key='valuation_container',
            expandable=True,
            expanded=False,
            label='Valuation')
        self.add_input(self.valuation_container)
        self.foundation_cost = inputs.Text(
            args_key='foundation_cost',
            helptext=(
                "A floating point number for the unit cost of the "
                "foundation type (in millions of dollars). The cost of "
                "a foundation will depend on the type selected, which "
                "itself depends on a variety of factors including "
                "depth and turbine choice.  Please see the User's "
                "Guide for guidance on properly selecting this value."),
            label='Cost of the Foundation Type (USD, in Millions)',
            validator=self.validator)
        self.valuation_container.add_input(self.foundation_cost)
        self.discount_rate = inputs.Text(
            args_key='discount_rate',
            helptext=(
                "The discount rate reflects preferences for immediate "
                "benefits over future benefits (e.g., would an "
                "individual rather receive $10 today or $10 five years "
                "from now?). See the User's Guide for guidance on "
                "selecting this value."),
            label='Discount Rate',
            validator=self.validator)
        self.valuation_container.add_input(self.discount_rate)
        self.grid_points = inputs.File(
            args_key='grid_points_path',
            helptext=(
                "An optional CSV file with grid and land points to "
                "determine cable distances from.  An example:<br/> "
                "<table border='1'> <tr> <th>ID</th> <th>TYPE</th> "
                "<th>LATI</th> <th>LONG</th> </tr> <tr> <td>1</td> "
                "<td>GRID</td> <td>42.957</td> <td>-70.786</td> </tr> "
                "<tr> <td>2</td> <td>LAND</td> <td>42.632</td> "
                "<td>-71.143</td> </tr> <tr> <td>3</td> <td>LAND</td> "
                "<td>41.839</td> <td>-70.394</td> </tr> </table> "
                "<br/><br/>Each point location is represented as a "
                "single row with columns being <b>ID</b>, <b>TYPE</b>, "
                "<b>LATI</b>, and <b>LONG</b>. The <b>LATI</b> and "
                "<b>LONG</b> columns indicate the coordinates for the "
                "point.  The <b>TYPE</b> column relates to whether it "
                "is a land or grid point.  The <b>ID</b> column is a "
                "simple unique integer.  The shortest distance between "
                "respective points is used for calculations.  See the "
                "User's Guide for more information."),
            label='Grid Connection Points (Optional)',
            validator=self.validator)
        self.valuation_container.add_input(self.grid_points)
        self.avg_grid_dist = inputs.Text(
            args_key='avg_grid_distance',
            helptext=(
                "<b>Always required, but NOT used in the model if "
                "Grid Points provided</b><br/><br/>A number in "
                "kilometres that is only used if grid points are NOT "
                "used in valuation.  When running valuation using the "
                "land polygon to compute distances, the model uses an "
                "average distance to the onshore grid from coastal "
                "cable landing points instead of specific grid "
                "connection points.  See the User's Guide for a "
                "description of the approach and the method used to "
                "calculate the default value."),
            label='Average Shore to Grid Distance (Kilometers)',
            validator=self.validator)
        self.valuation_container.add_input(self.avg_grid_dist)
        self.price_table = inputs.Checkbox(
            args_key='price_table',
            helptext=(
                "When checked the model will use the social cost of "
                "wind energy table provided in the input below.  If "
                "not checked the price per year will be determined "
                "using the price of energy input and the annual rate "
                "of change."),
            label='Use Price Table')
        self.valuation_container.add_input(self.price_table)
        self.wind_schedule = inputs.File(
            args_key='wind_schedule',
            helptext=(
                "A CSV file that has the price of wind energy per "
                "kilowatt hour for each year of the wind farms life. "
                "The CSV file should have the following two "
                "columns:<br/><br/><b>Year:</b> a set of integers "
                "indicating each year for the lifespan of the wind "
                "farm.  They can be in date form such as : 2010, 2011, "
                "2012... OR simple time step integers such as : 0, 1, "
                "2... <br/><br/><b>Price:</b> a set of floats "
                "indicating the price of wind energy per kilowatt hour "
                "for a particular year or time step in the wind farms "
                "life.<br/><br/>An example:<br/> <table border='1'> "
                "<tr><th>Year</th> <th>Price</th></tr><tr><td>0</td><t "
                "d>.244</td></tr><tr><td>1</td><td>.255</td></tr><tr>< "
                "td>2</td><td>.270</td></tr><tr><td>3</td><td>.275</td "
                "></tr><tr><td>4</td><td>.283</td></tr><tr><td>5</td>< "
                "td>.290</td></tr></table><br/><br/><b>NOTE:</b> The "
                "number of years or time steps listed must match the "
                "<b>time</b> parameter in the <b>Global Wind Energy "
                "Parameters</b> input file above.  In the above "
                "example we have 6 years for the lifetime of the farm, "
                "year 0 being a construction year and year 5 being the "
                "last year."),
            interactive=False,
            label='Wind Energy Price Table (CSV)',
            validator=self.validator)
        self.valuation_container.add_input(self.wind_schedule)
        self.wind_price = inputs.Text(
            args_key='wind_price',
            helptext=(
                "The price of energy per kilowatt hour.  This is the "
                "price that will be used for year or time step 0 and "
                "will then be adjusted based on the rate of change "
                "percentage from the input below.  See the User's "
                "Guide for guidance about determining this value."),
            label='Price of Energy per Kilowatt Hour ($/kWh)',
            validator=self.validator)
        self.valuation_container.add_input(self.wind_price)
        self.rate_change = inputs.Text(
            args_key='rate_change',
            helptext=(
                "The annual rate of change in the price of wind "
                "energy.  This should be expressed as a decimal "
                "percentage.  For example, 0.1 for a 10% annual price "
                "change."),
            label='Annual Rate of Change in Price of Wind Energy',
            validator=self.validator)
        self.valuation_container.add_input(self.rate_change)

        # Set interactivity, requirement as input sufficiency changes
        self.aoi.sufficiency_changed.connect(
            self.land_polygon.set_interactive)
        self.land_polygon.sufficiency_changed.connect(
            self.min_distance.set_interactive)
        self.land_polygon.sufficiency_changed.connect(
            self.max_distance.set_interactive)
        self.price_table.sufficiency_changed.connect(
            self._toggle_price_options)

    def _toggle_price_options(self, use_price_table):
        self.wind_schedule.set_interactive(use_price_table)
        self.wind_price.set_interactive(not use_price_table)
        self.rate_change.set_interactive(not use_price_table)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.wind_data.args_key: self.wind_data.value(),
            self.bathymetry.args_key: self.bathymetry.value(),
            self.global_wind_parameters.args_key:
                self.global_wind_parameters.value(),
            self.turbine_parameters.args_key: self.turbine_parameters.value(),
            self.number_of_machines.args_key: self.number_of_machines.value(),
            self.min_depth.args_key: self.min_depth.value(),
            self.max_depth.args_key: self.max_depth.value(),
            self.valuation_container.args_key: self.valuation_container.value(),
            self.avg_grid_dist.args_key: self.avg_grid_dist.value(),
        }
        if self.aoi.value():
            args[self.aoi.args_key] = self.aoi.value()
        if self.land_polygon.value():
            args[self.land_polygon.args_key] = self.land_polygon.value()
        if self.min_distance.value():
            args[self.min_distance.args_key] = self.min_distance.value()
        if self.max_distance.value():
            args[self.max_distance.args_key] = self.max_distance.value()
        if self.grid_points.value():
            args[self.grid_points.args_key] = self.grid_points.value()

        # Include these args if valuation is checked.
        if args[self.valuation_container.args_key]:
            args[self.foundation_cost.args_key] = self.foundation_cost.value()
            args[self.discount_rate.args_key] = self.discount_rate.value()
            args[self.price_table.args_key] = self.price_table.value()
            args[self.wind_schedule.args_key] = self.wind_schedule.value()
            args[self.wind_price.args_key] = self.wind_price.value()
            args[self.rate_change.args_key] = self.rate_change.value()

        return args
