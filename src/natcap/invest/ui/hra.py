# coding=UTF-8
from natcap.invest.ui import model, inputs
from natcap.invest import hra


class HabitatRiskAssessment(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Habitat Risk Assessment',
            target=hra.execute,
            validator=hra.validate,
            localdoc=u'../documentation/habitat_risk_assessment.html')

        self.info_table_path = inputs.File(
            args_key=u'info_table_path',
            helptext=(
                u"A CSV or Excel file that contains the name of the habitat "
                u"(H) or stressor (s) on the `NAME` column that matches the "
                u"names in `criteria_table_path`. Each H/S has its "
                u"corresponding vector or raster path on the `PATH` column. "
                u"The `STRESSOR BUFFER (meters)` column should have a buffer "
                u"value if the `TYPE` column is a stressor."),
            label=u'Habitat Stressor Information CSV or Excel File',
            validator=self.validator)
        self.add_input(self.info_table_path)
        self.criteria_table_path = inputs.File(
            args_key=u'criteria_table_path',
            helptext=(
                u"A CSV or Excel file that contains the set of criteria "
                u"ranking  (rating, DQ and weight) of each stressor on each "
                u"habitat, as well as the habitat resilience attributes."),
            label=u'Criteria Scores CSV or Excel File',
            validator=self.validator)
        self.add_input(self.criteria_table_path)
        self.resolution = inputs.Text(
            args_key=u'resolution',
            helptext=(
                u"The size that should be used to grid the given habitat and "
                u"stressor files into rasters. This value will be the pixel "
                u"size of the completed raster files."),
            label=u'Resolution of Analysis (meters)',
            validator=self.validator)
        self.add_input(self.resolution)
        self.max_rating = inputs.Text(
            args_key=u'max_rating',
            helptext=(
                u"This is the highest score that is used to rate a criteria "
                u"within this model run. This value would be used to compare "
                u"with the values within Rating column of the Criteria Scores "
                u"table."),
            label=u'Maximum Criteria Score',
            validator=self.validator)
        self.add_input(self.max_rating)
        self.risk_eq = inputs.Dropdown(
            args_key=u'risk_eq',
            helptext=(
                u"Each of these represents an option of a risk calculation "
                u"equation. This will determine the numeric output of risk "
                u"for every habitat and stressor overlap area."),
            label=u'Risk Equation',
            options=[u'Multiplicative', u'Euclidean'])
        self.add_input(self.risk_eq)
        self.decay_eq = inputs.Dropdown(
            args_key=u'decay_eq',
            helptext=(
                u"Each of these represents an option of a decay equation "
                u"for the buffered stressors. If stressor buffering is "
                u"desired, this equation will determine the rate at which "
                u"stressor data is reduced."),
            label=u'Decay Equation',
            options=[u'None', u'Linear', u'Exponential'])
        self.add_input(self.decay_eq)
        self.aoi_vector_path = inputs.File(
            args_key=u'aoi_vector_path',
            helptext=(
                u"An OGR-supported vector file containing feature containing "
                u"one or more planning regions. subregions. An optional field "
                u"called `name` could be added to compute average risk values "
                u"within each subregion."),
            label=u'Area of Interest (Vector)',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)
        self.visualize_outputs = inputs.Checkbox(
            args_key='visualize_outputs',
            helptext=(
                u"Check to enable the generation of GeoJSON outputs. This "
                u"could be used to visualize the risk scores on a map in the "
                u"HRA visualization web application."),
            label=u'Generate GeoJSONs for Web Visualization')
        self.add_input(self.visualize_outputs)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.info_table_path.args_key: self.info_table_path.value(),
            self.criteria_table_path.args_key: self.criteria_table_path.value(),
            self.resolution.args_key: self.resolution.value(),
            self.risk_eq.args_key: self.risk_eq.value(),
            self.decay_eq.args_key: self.decay_eq.value(),
            self.max_rating.args_key: self.max_rating.value(),
            self.aoi_vector_path.args_key: self.aoi_vector_path.value(),
            self.visualize_outputs.args_key: self.visualize_outputs.value(),
        }

        return args
