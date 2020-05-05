# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.fisheries import fisheries, fisheries_hst


class Fisheries(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Fisheries',
            target=fisheries.execute,
            validator=fisheries.validate,
            localdoc='fisheries.html')

        self.alpha_only = inputs.Label(
            text=(
                "This tool is in an ALPHA testing stage and should "
                "not be used for decision making."))
        self.aoi_vector_path = inputs.File(
            args_key='aoi_vector_path',
            helptext=(
                "An OGR-supported vector file used to display outputs "
                "within the region(s) of interest.<br><br>The layer "
                "should contain one feature for every region of "
                "interest, each feature of which should have a ‘NAME’ "
                "attribute.  The 'NAME' attribute can be numeric or "
                "alphabetic, but must be unique within the given file."),
            label='Area of Interest (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)
        self.total_timesteps = inputs.Text(
            args_key='total_timesteps',
            helptext=(
                "The number of time steps the simulation shall "
                "execute before completion.<br><br>Must be a positive "
                "integer."),
            label='Number of Time Steps for Model Run',
            validator=self.validator)
        self.add_input(self.total_timesteps)
        self.popu_cont = inputs.Container(
            label='Population Parameters')
        self.add_input(self.popu_cont)
        self.population_type = inputs.Dropdown(
            args_key='population_type',
            helptext=(
                "Specifies whether the lifecycle classes provided in "
                "the Population Parameters CSV file represent ages "
                "(uniform duration) or stages.<br><br>Age-based models "
                "(e.g.  Lobster, Dungeness Crab) are separated by "
                "uniform, fixed-length time steps (usually "
                "representing a year).<br><br>Stage-based models (e.g. "
                "White Shrimp) allow lifecycle-classes to have "
                "nonuniform durations based on the assumed resolution "
                "of the provided time step.<br><br>If the stage-based "
                "model is selected, the Population Parameters CSV file "
                "must include a ‘Duration’ vector alongside the "
                "survival matrix that contains the number of time "
                "steps that each stage lasts."),
            label='Population Model Type',
            options=['Age-Based', 'Stage-Based'])
        self.popu_cont.add_input(self.population_type)
        self.sexsp = inputs.Dropdown(
            args_key='sexsp',
            helptext=(
                "Specifies whether or not the lifecycle classes "
                "provided in the Populaton Parameters CSV file are "
                "distinguished by sex."),
            label='Population Classes are Sex-Specific',
            options=['No', 'Yes'])
        self.popu_cont.add_input(self.sexsp)
        self.harvest_units = inputs.Dropdown(
            args_key='harvest_units',
            helptext=(
                "Specifies whether the harvest output values are "
                "calculated in terms of number of individuals or in "
                "terms of biomass (weight).<br><br>If ‘Weight’ is "
                "selected, the Population Parameters CSV file must "
                "include a 'Weight' vector alongside the survival "
                "matrix that contains the weight of each lifecycle "
                "class and sex if model is sex-specific."),
            label='Harvest by Individuals or Weight',
            options=['Individuals', 'Weight'])
        self.popu_cont.add_input(self.harvest_units)
        self.do_batch = inputs.Checkbox(
            args_key='do_batch',
            helptext=(
                "Specifies whether program will perform a single "
                "model run or a batch (set) of model runs.<br><br>For "
                "single model runs, users submit a filepath pointing "
                "to a single Population Parameters CSV file.  For "
                "batch model runs, users submit a directory path "
                "pointing to a set of Population Parameters CSV files."),
            label='Batch Processing')
        self.popu_cont.add_input(self.do_batch)
        self.population_csv_path = inputs.File(
            args_key='population_csv_path',
            helptext=(
                "The provided CSV file should contain all necessary "
                "attributes for the sub-populations based on lifecycle "
                "class, sex, and area - excluding possible migration "
                "information.<br><br>Please consult the documentation "
                "to learn more about what content should be provided "
                "and how the CSV file should be structured."),
            label='Population Parameters File (CSV)',
            validator=self.validator)
        self.popu_cont.add_input(self.population_csv_path)
        self.population_csv_dir = inputs.Folder(
            args_key='population_csv_dir',
            helptext=(
                "The provided CSV folder should contain a set of "
                "Population Parameters CSV files with all necessary "
                "attributes for sub-populations based on lifecycle "
                "class, sex, and area - excluding possible migration "
                "information.<br><br>The name of each file will serve "
                "as the prefix of the outputs created by the model "
                "run.<br><br>Please consult the documentation to learn "
                "more about what content should be provided and how "
                "the CSV file should be structured."),
            interactive=False,
            label='Population Parameters CSV Folder',
            validator=self.validator)
        self.popu_cont.add_input(self.population_csv_dir)
        self.recr_cont = inputs.Container(
            label='Recruitment Parameters')
        self.add_input(self.recr_cont)
        self.total_init_recruits = inputs.Text(
            args_key='total_init_recruits',
            helptext=(
                "The initial number of recruits in the population "
                "model at time equal to zero.<br><br>If the model "
                "contains multiple regions of interest or is "
                "distinguished by sex, this value will be evenly "
                "divided and distributed into each sub-population."),
            label='Total Initial Recruits',
            validator=self.validator)
        self.recr_cont.add_input(self.total_init_recruits)
        self.recruitment_type = inputs.Dropdown(
            args_key='recruitment_type',
            helptext=(
                "The selected equation is used to calculate "
                "recruitment into the subregions at the beginning of "
                "each time step.  Corresponding parameters must be "
                "specified with each function:<br><br>The Beverton- "
                "Holt and Ricker functions both require arguments for "
                "the ‘Alpha’ and ‘Beta’ parameters.<br><br>The "
                "Fecundity function requires a 'Fecundity' vector "
                "alongside the survival matrix in the Population "
                "Parameters CSV file indicating the per-capita "
                "offspring for each lifecycle class.<br><br>The Fixed "
                "function requires an argument for the ‘Total Recruits "
                "per Time Step’ parameter that represents a single "
                "total recruitment value to be distributed into the "
                "population model at the beginning of each time step."),
            label='Recruitment Function Type',
            options=['Beverton-Holt', 'Ricker', 'Fecundity', 'Fixed'])
        self.recr_cont.add_input(self.recruitment_type)
        self.spawn_units = inputs.Dropdown(
            args_key='spawn_units',
            helptext=(
                "Specifies whether the spawner abundance used in the "
                "recruitment function should be calculated in terms of "
                "number of individuals or in terms of biomass "
                "(weight).<br><br>If 'Weight' is selected, the user "
                "must provide a 'Weight' vector alongside the survival "
                "matrix in the Population Parameters CSV file.  The "
                "'Alpha' and 'Beta' parameters provided by the user "
                "should correspond to the selected choice.<br><br>Used "
                "only for the Beverton-Holt and Ricker recruitment "
                "functions."),
            label='Spawners by Individuals or Weight (Beverton-Holt / Ricker)',
            options=['Individuals', 'Weight'])
        self.recr_cont.add_input(self.spawn_units)
        self.alpha = inputs.Text(
            args_key='alpha',
            helptext=(
                "Specifies the shape of the stock-recruit curve. "
                "Used only for the Beverton-Holt and Ricker "
                "recruitment functions.<br><br>Used only for the "
                "Beverton-Holt and Ricker recruitment functions."),
            label='Alpha (Beverton-Holt / Ricker)',
            validator=self.validator)
        self.recr_cont.add_input(self.alpha)
        self.beta = inputs.Text(
            args_key='beta',
            helptext=(
                "Specifies the shape of the stock-recruit "
                "curve.<br><br>Used only for the Beverton-Holt and "
                "Ricker recruitment functions."),
            label='Beta (Beverton-Holt / Ricker)',
            validator=self.validator)
        self.recr_cont.add_input(self.beta)
        self.total_recur_recruits = inputs.Text(
            args_key='total_recur_recruits',
            helptext=(
                "Specifies the total number of recruits that come "
                "into the population at each time step (a fixed "
                "number).<br><br>Used only for the Fixed recruitment "
                "function."),
            label='Total Recruits per Time Step (Fixed)',
            validator=self.validator)
        self.recr_cont.add_input(self.total_recur_recruits)
        self.migr_cont = inputs.Container(
            args_key='migr_cont',
            expandable=True,
            expanded=False,
            label='Migration Parameters')
        self.add_input(self.migr_cont)
        self.migration_dir = inputs.Folder(
            args_key='migration_dir',
            helptext=(
                "The selected folder contain CSV migration matrices "
                "to be used in the simulation.  Each CSV file contains "
                "a single migration matrix corresponding to an "
                "lifecycle class that migrates.  The folder should "
                "contain one CSV file for each lifecycle class that "
                "migrates.<br><br>The files may be named anything, but "
                "must end with an underscore followed by the name of "
                "the age or stage.  The name of the age or stage must "
                "correspond to an age or stage within the Population "
                "Parameters CSV file.  For example, a migration file "
                "might be named 'migration_adult.csv'.<br><br>Each "
                "matrix cell should contain a decimal fraction "
                "indicating the percetage of the population that will "
                "move from one area to another.  Each column should "
                "sum to one."),
            label='Migration Matrix CSV Folder (Optional)',
            validator=self.validator)
        self.migr_cont.add_input(self.migration_dir)
        self.val_cont = inputs.Container(
            args_key='val_cont',
            expandable=True,
            expanded=False,
            label='Valuation Parameters')
        self.add_input(self.val_cont)
        self.frac_post_process = inputs.Text(
            args_key='frac_post_process',
            helptext=(
                "Decimal fraction indicating the percentage of "
                "harvested catch remaining after post-harvest "
                "processing is complete."),
            label='Fraction of Harvest Kept After Processing',
            validator=self.validator)
        self.val_cont.add_input(self.frac_post_process)
        self.unit_price = inputs.Text(
            args_key='unit_price',
            helptext=(
                "Specifies the price per harvest unit.<br><br>If "
                "‘Harvest by Individuals or Weight’ was set to "
                "‘Individuals’, this should be the price per "
                "individual.  If set to ‘Weight’, this should be the "
                "price per unit weight."),
            label='Unit Price',
            validator=self.validator)
        self.val_cont.add_input(self.unit_price)

        # Set interactivity, requirement as input sufficiency changes
        self.do_batch.sufficiency_changed.connect(
            self._toggle_batch_runs)

        # Enable/disable parameters when the recruitment function changes.
        self.recruitment_type.value_changed.connect(
            self._control_recruitment_parameters)

    def _toggle_batch_runs(self, do_batch_runs):
        self.population_csv_path.set_interactive(not do_batch_runs)
        self.population_csv_dir.set_interactive(do_batch_runs)

    def _control_recruitment_parameters(self, recruit_func):
        for parameter in (self.spawn_units, self.alpha, self.beta,
                          self.total_recur_recruits):
            parameter.set_interactive(False)

        if self.recruitment_type.value() in ('Beverton-Holt', 'Ricker'):
            for parameter in (self.spawn_units, self.alpha, self.beta):
                parameter.set_interactive(True)
        elif self.recruitment_type.value() == 'Fixed':
            self.total_recur_recruits.set_interactive(True)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi_vector_path.args_key: self.aoi_vector_path.value(),
            self.population_type.args_key: self.population_type.value(),
            self.sexsp.args_key: self.sexsp.value(),
            self.harvest_units.args_key: self.harvest_units.value(),
            self.do_batch.args_key: self.do_batch.value(),
            self.population_csv_path.args_key: self.population_csv_path.value(),
            self.population_csv_dir.args_key: self.population_csv_dir.value(),
            self.recruitment_type.args_key: self.recruitment_type.value(),
            self.spawn_units.args_key: self.spawn_units.value(),
            self.migr_cont.args_key: self.migr_cont.value(),
            self.val_cont.args_key: self.val_cont.value(),
        }

        # Cast numeric inputs to a float
        for numeric_input in (self.alpha, self.beta, self.total_recur_recruits,
                              self.total_init_recruits, self.total_timesteps):
            if numeric_input.value():
                args[numeric_input.args_key] = numeric_input.value()

        if self.val_cont.value():
            args[self.frac_post_process.args_key] = (
                self.frac_post_process.value())
            args[self.unit_price.args_key] = self.unit_price.value()

        if self.migr_cont.value():
            args[self.migration_dir.args_key] = self.migration_dir.value()

        return args


class FisheriesHST(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Fisheries Habitat Scenario Tool',
            target=fisheries_hst.execute,
            validator=fisheries_hst.validate,
            localdoc='fisheries.html')

        self.alpha_only = inputs.Label(
            text=(
                "This tool is in an ALPHA testing stage and should "
                "not be used for decision making."))
        self.pop_cont = inputs.Container(
            args_key='pop_cont',
            expanded=True,
            label='Population Parameters')
        self.add_input(self.pop_cont)
        self.population_csv_path = inputs.File(
            args_key='population_csv_path',
            helptext=(
                "A CSV file containing all necessary attributes for "
                "population classes based on age/stage, sex, and area "
                "- excluding possible migration "
                "information.<br><br>See the 'Running the Model >> "
                "Core Model >> Population Parameters' section in the "
                "model's documentation for help on how to format this "
                "file."),
            label='Population Parameters File (CSV)',
            validator=self.validator)
        self.pop_cont.add_input(self.population_csv_path)
        self.sexsp = inputs.Dropdown(
            args_key='sexsp',
            helptext=(
                "Specifies whether or not the population classes "
                "provided in the Populaton Parameters CSV file are "
                "distinguished by sex."),
            label='Population Classes are Sex-Specific',
            options=['No', 'Yes'])
        self.pop_cont.add_input(self.sexsp)
        self.hab_cont = inputs.Container(
            args_key='hab_cont',
            expanded=True,
            label='Habitat Parameters')
        self.add_input(self.hab_cont)
        self.habitat_csv_dep_path = inputs.File(
            args_key='habitat_dep_csv_path',
            helptext=(
                "A CSV file containing the habitat dependencies (0-1) "
                "for each life stage or age and for each habitat type "
                "included in the Habitat Change CSV File.<br><br>See "
                "the 'Running the Model >> Habitat Scenario Tool >> "
                "Habitat Parameters' section in the model's "
                "documentation for help on how to format this file."),
            label='Habitat Dependency Parameters File (CSV)',
            validator=self.validator)
        self.hab_cont.add_input(self.habitat_csv_dep_path)
        self.habitat_chg_csv_path = inputs.File(
            args_key='habitat_chg_csv_path',
            helptext=(
                "A CSV file containing the percent changes in habitat "
                "area by subregion (if applicable). The habitats "
                "included should be those which the population depends "
                "on at any life stage.<br><br>See the 'Running the "
                "Model >> Habitat Scenario Tool >> Habitat Parameters' "
                "section in the model's documentation for help on how "
                "to format this file."),
            label='Habitat Area Change File (CSV)',
            validator=self.validator)
        self.hab_cont.add_input(self.habitat_chg_csv_path)
        self.gamma = inputs.Text(
            args_key='gamma',
            helptext=(
                "Gamma describes the relationship between a change in "
                "habitat area and a change in survival of life stages "
                "dependent on that habitat.  Specify a value between 0 "
                "and 1.<br><br>See the documentation for advice on "
                "selecting a gamma value."),
            label='Gamma',
            validator=self.validator)
        self.hab_cont.add_input(self.gamma)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.pop_cont.args_key: self.pop_cont.value(),
            self.population_csv_path.args_key: self.population_csv_path.value(),
            self.sexsp.args_key: self.sexsp.value(),
            self.hab_cont.args_key: self.hab_cont.value(),
            self.habitat_csv_dep_path.args_key: self.habitat_csv_dep_path.value(),
            self.habitat_chg_csv_path.args_key: self.habitat_chg_csv_path.value(),
            self.gamma.args_key: self.gamma.value(),
        }

        return args
