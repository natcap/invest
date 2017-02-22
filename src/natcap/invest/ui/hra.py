# coding=UTF-8

from natcap.invest.ui import model
from natcap.ui import inputs
import natcap.invest.habitat_risk_assessment.hra


class HabitatRiskAssessment(model.Model):
    label = u'Habitat Risk Assessment'
    target = staticmethod(natcap.invest.habitat_risk_assessment.hra.execute)
    validator = staticmethod(natcap.invest.habitat_risk_assessment.hra.validate)
    localdoc = u'../documentation/habitat_risk_assessment.html'

    def __init__(self):
        model.Model.__init__(self)

        self.csv_uri = inputs.Folder(
            args_key=u'csv_uri',
            helptext=(
                u"A folder containing multiple CSV files.  Each file "
                u"refers to the overlap between a habitat and a "
                u"stressor pulled from habitat and stressor shapefiles "
                u"during the run of the HRA Preprocessor."),
            label=u'Criteria Scores CSV Folder',
            required=True,
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
            required=True,
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
            required=True,
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
            required=True,
            validator=self.validator)
        self.add_input(self.max_stress)
        self.aoi_tables = inputs.File(
            args_key=u'aoi_tables',
            helptext=(
                u"An OGR-supported vector file containing feature "
                u"subregions.  The program will create additional "
                u"summary outputs across each subregion."),
            label=u'Subregions (Vector)',
            required=True,
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
