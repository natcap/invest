# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.fisheries import fisheries, fisheries_hst


class Fisheries(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Fisheries',
            target=fisheries.execute,
            validator=fisheries.validate,
            localdoc=u'fisheries.html')

        self.alpha_only = inputs.Label(
            text=(
                u"This tool is in an ALPHA testing stage and should "
                u"not be used for decision making."))
        self.aoi_vector_path = inputs.File(
            args_key=u'aoi_vector_path',
            helptext=(
                u"An OGR-supported vector file used to display outputs "
                u"within the region(s) of interest.<br><br>The layer "
                u"should contain one feature for every region of "
                u"interest, each feature of which should have a ‘NAME’ "
                u"attribute.  The 'NAME' attribute can be numeric or "
                u"alphabetic, but must be unique within the given file."),
            label=u'Area of Interest (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)
        self.total_timesteps = inputs.Text(
            args_key=u'total_timesteps',
            helptext=(
                u"The number of time steps the simulation shall "
                u"execute before completion.<br><br>Must be a positive "
                u"integer."),
            label=u'Number of Time Steps for Model Run',
            validator=self.validator)
        self.add_input(self.total_timesteps)
        self.popu_cont = inputs.Container(
            label=u'Population Parameters')
        self.add_input(self.popu_cont)
        self.population_type = inputs.Dropdown(
            args_key=u'population_type',
            helptext=(
                u"Specifies whether the lifecycle classes provided in "
                u"the Population Parameters CSV file represent ages "
                u"(uniform duration) or stages.<br><br>Age-based models "
                u"(e.g.  Lobster, Dungeness Crab) are separated by "
                u"uniform, fixed-length time steps (usually "
                u"representing a year).<br><br>Stage-based models (e.g. "
                u"White Shrimp) allow lifecycle-classes to have "
                u"nonuniform durations based on the assumed resolution "
                u"of the provided time step.<br><br>If the stage-based "
                u"model is selected, the Population Parameters CSV file "
                u"must include a ‘Duration’ vector alongside the "
                u"survival matrix that contains the number of time "
                u"steps that each stage lasts."),
            label=u'Population Model Type',
            options=[u'Age-Based', u'Stage-Based'])
        self.popu_cont.add_input(self.population_type)
        self.sexsp = inputs.Dropdown(
            args_key=u'sexsp',
            helptext=(
                u"Specifies whether or not the lifecycle classes "
                u"provided in the Populaton Parameters CSV file are "
                u"distinguished by sex."),
            label=u'Population Classes are Sex-Specific',
            options=[u'No', u'Yes'])
        self.popu_cont.add_input(self.sexsp)
        self.harvest_units = inputs.Dropdown(
            args_key=u'harvest_units',
            helptext=(
                u"Specifies whether the harvest output values are "
                u"calculated in terms of number of individuals or in "
                u"terms of biomass (weight).<br><br>If ‘Weight’ is "
                u"selected, the Population Parameters CSV file must "
                u"include a 'Weight' vector alongside the survival "
                u"matrix that contains the weight of each lifecycle "
                u"class and sex if model is sex-specific."),
            label=u'Harvest by Individuals or Weight',
            options=[u'Individuals', u'Weight'])
        self.popu_cont.add_input(self.harvest_units)
        self.do_batch = inputs.Checkbox(
            args_key=u'do_batch',
            helptext=(
                u"Specifies whether program will perform a single "
                u"model run or a batch (set) of model runs.<br><br>For "
                u"single model runs, users submit a filepath pointing "
                u"to a single Population Parameters CSV file.  For "
                u"batch model runs, users submit a directory path "
                u"pointing to a set of Population Parameters CSV files."),
            label=u'Batch Processing')
        self.popu_cont.add_input(self.do_batch)
        self.population_csv_path = inputs.File(
            args_key=u'population_csv_path',
            helptext=(
                u"The provided CSV file should contain all necessary "
                u"attributes for the sub-populations based on lifecycle "
                u"class, sex, and area - excluding possible migration "
                u"information.<br><br>Please consult the documentation "
                u"to learn more about what content should be provided "
                u"and how the CSV file should be structured."),
            label=u'Population Parameters File (CSV)',
            validator=self.validator)
        self.popu_cont.add_input(self.population_csv_path)
        self.population_csv_dir = inputs.Folder(
            args_key=u'population_csv_dir',
            helptext=(
                u"The provided CSV folder should contain a set of "
                u"Population Parameters CSV files with all necessary "
                u"attributes for sub-populations based on lifecycle "
                u"class, sex, and area - excluding possible migration "
                u"information.<br><br>The name of each file will serve "
                u"as the prefix of the outputs created by the model "
                u"run.<br><br>Please consult the documentation to learn "
                u"more about what content should be provided and how "
                u"the CSV file should be structured."),
            interactive=False,
            label=u'Population Parameters CSV Folder',
            validator=self.validator)
        self.popu_cont.add_input(self.population_csv_dir)
        self.recr_cont = inputs.Container(
            label=u'Recruitment Parameters')
        self.add_input(self.recr_cont)
        self.total_init_recruits = inputs.Text(
            args_key=u'total_init_recruits',
            helptext=(
                u"The initial number of recruits in the population "
                u"model at time equal to zero.<br><br>If the model "
                u"contains multiple regions of interest or is "
                u"distinguished by sex, this value will be evenly "
                u"divided and distributed into each sub-population."),
            label=u'Total Initial Recruits',
            validator=self.validator)
        self.recr_cont.add_input(self.total_init_recruits)
        self.recruitment_type = inputs.Dropdown(
            args_key=u'recruitment_type',
            helptext=(
                u"The selected equation is used to calculate "
                u"recruitment into the subregions at the beginning of "
                u"each time step.  Corresponding parameters must be "
                u"specified with each function:<br><br>The Beverton- "
                u"Holt and Ricker functions both require arguments for "
                u"the ‘Alpha’ and ‘Beta’ parameters.<br><br>The "
                u"Fecundity function requires a 'Fecundity' vector "
                u"alongside the survival matrix in the Population "
                u"Parameters CSV file indicating the per-capita "
                u"offspring for each lifecycle class.<br><br>The Fixed "
                u"function requires an argument for the ‘Total Recruits "
                u"per Time Step’ parameter that represents a single "
                u"total recruitment value to be distributed into the "
                u"population model at the beginning of each time step."),
            label=u'Recruitment Function Type',
            options=[u'Beverton-Holt', u'Ricker', u'Fecundity', u'Fixed'])
        self.recr_cont.add_input(self.recruitment_type)
        self.spawn_units = inputs.Dropdown(
            args_key=u'spawn_units',
            helptext=(
                u"Specifies whether the spawner abundance used in the "
                u"recruitment function should be calculated in terms of "
                u"number of individuals or in terms of biomass "
                u"(weight).<br><br>If 'Weight' is selected, the user "
                u"must provide a 'Weight' vector alongside the survival "
                u"matrix in the Population Parameters CSV file.  The "
                u"'Alpha' and 'Beta' parameters provided by the user "
                u"should correspond to the selected choice.<br><br>Used "
                u"only for the Beverton-Holt and Ricker recruitment "
                u"functions."),
            label=u'Spawners by Individuals or Weight (Beverton-Holt / Ricker)',
            options=[u'Individuals', u'Weight'])
        self.recr_cont.add_input(self.spawn_units)
        self.alpha = inputs.Text(
            args_key=u'alpha',
            helptext=(
                u"Specifies the shape of the stock-recruit curve. "
                u"Used only for the Beverton-Holt and Ricker "
                u"recruitment functions.<br><br>Used only for the "
                u"Beverton-Holt and Ricker recruitment functions."),
            label=u'Alpha (Beverton-Holt / Ricker)',
            validator=self.validator)
        self.recr_cont.add_input(self.alpha)
        self.beta = inputs.Text(
            args_key=u'beta',
            helptext=(
                u"Specifies the shape of the stock-recruit "
                u"curve.<br><br>Used only for the Beverton-Holt and "
                u"Ricker recruitment functions."),
            label=u'Beta (Beverton-Holt / Ricker)',
            validator=self.validator)
        self.recr_cont.add_input(self.beta)
        self.total_recur_recruits = inputs.Text(
            args_key=u'total_recur_recruits',
            helptext=(
                u"Specifies the total number of recruits that come "
                u"into the population at each time step (a fixed "
                u"number).<br><br>Used only for the Fixed recruitment "
                u"function."),
            label=u'Total Recruits per Time Step (Fixed)',
            validator=self.validator)
        self.recr_cont.add_input(self.total_recur_recruits)
        self.migr_cont = inputs.Container(
            args_key=u'migr_cont',
            expandable=True,
            expanded=False,
            label=u'Migration Parameters')
        self.add_input(self.migr_cont)
        self.migration_dir = inputs.Folder(
            args_key=u'migration_dir',
            helptext=(
                u"The selected folder contain CSV migration matrices "
                u"to be used in the simulation.  Each CSV file contains "
                u"a single migration matrix corresponding to an "
                u"lifecycle class that migrates.  The folder should "
                u"contain one CSV file for each lifecycle class that "
                u"migrates.<br><br>The files may be named anything, but "
                u"must end with an underscore followed by the name of "
                u"the age or stage.  The name of the age or stage must "
                u"correspond to an age or stage within the Population "
                u"Parameters CSV file.  For example, a migration file "
                u"might be named 'migration_adult.csv'.<br><br>Each "
                u"matrix cell should contain a decimal fraction "
                u"indicating the percetage of the population that will "
                u"move from one area to another.  Each column should "
                u"sum to one."),
            label=u'Migration Matrix CSV Folder (Optional)',
            validator=self.validator)
        self.migr_cont.add_input(self.migration_dir)
        self.val_cont = inputs.Container(
            args_key=u'val_cont',
            expandable=True,
            expanded=False,
            label=u'Valuation Parameters')
        self.add_input(self.val_cont)
        self.frac_post_process = inputs.Text(
            args_key=u'frac_post_process',
            helptext=(
                u"Decimal fraction indicating the percentage of "
                u"harvested catch remaining after post-harvest "
                u"processing is complete."),
            label=u'Fraction of Harvest Kept After Processing',
            validator=self.validator)
        self.val_cont.add_input(self.frac_post_process)
        self.unit_price = inputs.Text(
            args_key=u'unit_price',
            helptext=(
                u"Specifies the price per harvest unit.<br><br>If "
                u"‘Harvest by Individuals or Weight’ was set to "
                u"‘Individuals’, this should be the price per "
                u"individual.  If set to ‘Weight’, this should be the "
                u"price per unit weight."),
            label=u'Unit Price',
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
            label=u'Fisheries Habitat Scenario Tool',
            target=fisheries_hst.execute,
            validator=fisheries_hst.validate,
            localdoc=u'fisheries.html')

        self.alpha_only = inputs.Label(
            text=(
                u"This tool is in an ALPHA testing stage and should "
                u"not be used for decision making."))
        self.pop_cont = inputs.Container(
            args_key=u'pop_cont',
            expanded=True,
            label=u'Population Parameters')
        self.add_input(self.pop_cont)
        self.population_csv_path = inputs.File(
            args_key=u'population_csv_path',
            helptext=(
                u"A CSV file containing all necessary attributes for "
                u"population classes based on age/stage, sex, and area "
                u"- excluding possible migration "
                u"information.<br><br>See the 'Running the Model >> "
                u"Core Model >> Population Parameters' section in the "
                u"model's documentation for help on how to format this "
                u"file."),
            label=u'Population Parameters File (CSV)',
            validator=self.validator)
        self.pop_cont.add_input(self.population_csv_path)
        self.sexsp = inputs.Dropdown(
            args_key=u'sexsp',
            helptext=(
                u"Specifies whether or not the population classes "
                u"provided in the Populaton Parameters CSV file are "
                u"distinguished by sex."),
            label=u'Population Classes are Sex-Specific',
            options=[u'No', u'Yes'])
        self.pop_cont.add_input(self.sexsp)
        self.hab_cont = inputs.Container(
            args_key=u'hab_cont',
            expanded=True,
            label=u'Habitat Parameters')
        self.add_input(self.hab_cont)
        self.habitat_csv_dep_path = inputs.File(
            args_key=u'habitat_dep_csv_path',
            helptext=(
                u"A CSV file containing the habitat dependencies (0-1) "
                u"for each life stage or age and for each habitat type "
                u"included in the Habitat Change CSV File.<br><br>See "
                u"the 'Running the Model >> Habitat Scenario Tool >> "
                u"Habitat Parameters' section in the model's "
                u"documentation for help on how to format this file."),
            label=u'Habitat Dependency Parameters File (CSV)',
            validator=self.validator)
        self.hab_cont.add_input(self.habitat_csv_dep_path)
        self.habitat_chg_csv_path = inputs.File(
            args_key=u'habitat_chg_csv_path',
            helptext=(
                u"A CSV file containing the percent changes in habitat "
                u"area by subregion (if applicable). The habitats "
                u"included should be those which the population depends "
                u"on at any life stage.<br><br>See the 'Running the "
                u"Model >> Habitat Scenario Tool >> Habitat Parameters' "
                u"section in the model's documentation for help on how "
                u"to format this file."),
            label=u'Habitat Area Change File (CSV)',
            validator=self.validator)
        self.hab_cont.add_input(self.habitat_chg_csv_path)
        self.gamma = inputs.Text(
            args_key=u'gamma',
            helptext=(
                u"Gamma describes the relationship between a change in "
                u"habitat area and a change in survival of life stages "
                u"dependent on that habitat.  Specify a value between 0 "
                u"and 1.<br><br>See the documentation for advice on "
                u"selecting a gamma value."),
            label=u'Gamma',
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
