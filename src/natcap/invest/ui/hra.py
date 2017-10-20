# coding=UTF-8
import functools

from natcap.invest.ui import model, inputs
from natcap.invest.habitat_risk_assessment import hra, hra_preprocessor


class HabitatRiskAssessment(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Habitat Risk Assessment',
            target=hra.execute,
            validator=hra.validate,
            localdoc=u'../documentation/habitat_risk_assessment.html')

        self.csv_uri = inputs.Folder(
            args_key=u'csv_uri',
            helptext=(
                u"A folder containing multiple CSV files.  Each file "
                u"refers to the overlap between a habitat and a "
                u"stressor pulled from habitat and stressor shapefiles "
                u"during the run of the HRA Preprocessor."),
            label=u'Criteria Scores CSV Folder',
            validator=self.validator)
        self.add_input(self.csv_uri)
        self.grid_size = inputs.Text(
            args_key=u'grid_size',
            helptext=(
                u"The size that should be used to grid the given "
                u"habitat and stressor shapefiles into rasters.  This "
                u"value will be the pixel size of the completed raster "
                u"files."),
            label=u'Resolution of Analysis (meters)',
            validator=self.validator)
        self.add_input(self.grid_size)
        self.risk_eq = inputs.Dropdown(
            args_key=u'risk_eq',
            helptext=(
                u"Each of these represents an option of a risk "
                u"calculation equation.  This will determine the "
                u"numeric output of risk for every habitat and stressor "
                u"overlap area."),
            label=u'Risk Equation',
            options=[u'Multiplicative', u'Euclidean'])
        self.add_input(self.risk_eq)
        self.decay_eq = inputs.Dropdown(
            args_key=u'decay_eq',
            helptext=(
                u"Each of these represents an option for decay "
                u"equations for the buffered stressors.  If stressor "
                u"buffering is desired, these equtions will determine "
                u"the rate at which stressor data is reduced."),
            label=u'Decay Equation',
            options=[u'None', u'Linear', u'Exponential'])
        self.add_input(self.decay_eq)
        self.max_rating = inputs.Text(
            args_key=u'max_rating',
            helptext=(
                u"This is the highest score that is used to rate a "
                u"criteria within this model run.  These values would "
                u"be placed within the Rating column of the habitat, "
                u"species, and stressor CSVs."),
            label=u'Maximum Criteria Score',
            validator=self.validator)
        self.add_input(self.max_rating)
        self.max_stress = inputs.Text(
            args_key=u'max_stress',
            helptext=(
                u"This is the largest number of stressors that are "
                u"suspected to overlap.  This will be used in order to "
                u"make determinations of low, medium, and high risk for "
                u"any given habitat."),
            label=u'Maximum Overlapping Stressors',
            validator=self.validator)
        self.add_input(self.max_stress)
        self.aoi_tables = inputs.File(
            args_key=u'aoi_tables',
            helptext=(
                u"An OGR-supported vector file containing feature "
                u"subregions.  The program will create additional "
                u"summary outputs across each subregion."),
            label=u'Subregions (Vector)',
            validator=self.validator)
        self.add_input(self.aoi_tables)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.csv_uri.args_key: self.csv_uri.value(),
            self.grid_size.args_key: self.grid_size.value(),
            self.risk_eq.args_key: self.risk_eq.value(),
            self.decay_eq.args_key: self.decay_eq.value(),
            self.max_rating.args_key: self.max_rating.value(),
            self.max_stress.args_key: self.max_stress.value(),
            self.aoi_tables.args_key: self.aoi_tables.value(),
        }

        return args


class HRAPreprocessor(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Habitat Risk Assessment Preprocessor',
            target=hra_preprocessor.execute,
            validator=hra_preprocessor.validate,
            localdoc=u'../documentation/habitat_risk_assessment.html')

        self.habs_dir = inputs.File(
            args_key=u'habitats_dir',
            helptext=(
                u"Checking this box indicates that habitats should be "
                u"used as a base for overlap with provided stressors. "
                u"If checked, the path to the habitat layers folder "
                u"must be provided."),
            hideable=True,
            label=u'Calculate Risk to Habitats?',
            validator=self.validator)
        self.add_input(self.habs_dir)
        self.species_dir = inputs.File(
            args_key=u'species_dir',
            helptext=(
                u"Checking this box indicates that species should be "
                u"used as a base for overlap with provided stressors. "
                u"If checked, the path to the species layers folder "
                u"must be provided."),
            hideable=True,
            label=u'Calculate Risk to Species?',
            validator=self.validator)
        self.add_input(self.species_dir)
        self.stressor_dir = inputs.Folder(
            args_key=u'stressors_dir',
            helptext=u'This is the path to the stressors layers folder.',
            label=u'Stressors Layers Folder',
            validator=self.validator)
        self.add_input(self.stressor_dir)
        self.cur_lulc_box = inputs.Container(
            expandable=False,
            label=u'Criteria')
        self.add_input(self.cur_lulc_box)
        self.help_label = inputs.Label(
            text=(
                u"(Choose at least 1 criteria for each category below, "
                u"and at least 4 total.)"))
        self.exp_crit = inputs.Multi(
            args_key=u'exposure_crits',
            callable_=functools.partial(inputs.Text, label="Input Criteria"),
            label=u'Exposure',
            link_text=u'Add Another')
        self.cur_lulc_box.add_input(self.exp_crit)
        self.sens_crit = inputs.Multi(
            args_key=u'sensitivity_crits',
            callable_=functools.partial(inputs.Text, label="Input Criteria"),
            label=u'Consequence: Sensitivity',
            link_text=u'Add Another')
        self.cur_lulc_box.add_input(self.sens_crit)
        self.res_crit = inputs.Multi(
            args_key=u'resilience_crits',
            callable_=functools.partial(inputs.Text, label="Input Criteria"),
            label=u'Consequence: Resilience',
            link_text=u'Add Another')
        self.cur_lulc_box.add_input(self.res_crit)
        self.crit_dir = inputs.File(
            args_key=u'criteria_dir',
            helptext=(
                u"Checking this box indicates that model should use "
                u"criteria from provided shapefiles.  Each shapefile in "
                u"the folder directories will need to contain a "
                u"'Rating' attribute to be used for calculations in the "
                u"HRA model.  Refer to the HRA User's Guide for "
                u"information about the MANDATORY layout of the "
                u"shapefile directories."),
            hideable=True,
            label=u'Use Spatially Explicit Risk Score in Shapefiles',
            validator=self.validator)
        self.add_input(self.crit_dir)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.habs_dir.args_key: self.habs_dir.value(),
            self.stressor_dir.args_key: self.stressor_dir.value(),
            self.exp_crit.args_key: self.exp_crit.value(),
            self.sens_crit.args_key: self.sens_crit.value(),
            self.res_crit.args_key: self.res_crit.value(),
            self.crit_dir.args_key: self.crit_dir.value(),
        }

        for hideable_input_name in ('habs_dir', 'species_dir', 'crit_dir'):
            hideable_input = getattr(self, hideable_input_name)
            if not hideable_input.hidden:
                args[hideable_input.args_key] = hideable_input.value()

        return args
