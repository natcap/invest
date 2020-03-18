# coding=UTF-8
from natcap.invest.ui import model, inputs
from natcap.invest import hra


class HabitatRiskAssessment(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Habitat Risk Assessment',
            target=hra.execute,
            validator=hra.validate,
            localdoc='../documentation/habitat_risk_assessment.html')

        self.info_table_path = inputs.File(
            args_key='info_table_path',
            helptext=(
                "A CSV or Excel file that contains the name of the habitat "
                "(H) or stressor (s) on the `NAME` column that matches the "
                "names in `criteria_table_path`. Each H/S has its "
                "corresponding vector or raster path on the `PATH` column. "
                "The `STRESSOR BUFFER (meters)` column should have a buffer "
                "value if the `TYPE` column is a stressor."),
            label='Habitat Stressor Information CSV or Excel File',
            validator=self.validator)
        self.add_input(self.info_table_path)
        self.criteria_table_path = inputs.File(
            args_key='criteria_table_path',
            helptext=(
                "A CSV or Excel file that contains the set of criteria "
                "ranking  (rating, DQ and weight) of each stressor on each "
                "habitat, as well as the habitat resilience attributes."),
            label='Criteria Scores CSV or Excel File',
            validator=self.validator)
        self.add_input(self.criteria_table_path)
        self.resolution = inputs.Text(
            args_key='resolution',
            helptext=(
                "The size that should be used to grid the given habitat and "
                "stressor files into rasters. This value will be the pixel "
                "size of the completed raster files."),
            label='Resolution of Analysis (meters)',
            validator=self.validator)
        self.add_input(self.resolution)
        self.max_rating = inputs.Text(
            args_key='max_rating',
            helptext=(
                "This is the highest score that is used to rate a criteria "
                "within this model run. This value would be used to compare "
                "with the values within Rating column of the Criteria Scores "
                "table."),
            label='Maximum Criteria Score',
            validator=self.validator)
        self.add_input(self.max_rating)
        self.risk_eq = inputs.Dropdown(
            args_key='risk_eq',
            helptext=(
                "Each of these represents an option of a risk calculation "
                "equation. This will determine the numeric output of risk "
                "for every habitat and stressor overlap area."),
            label='Risk Equation',
            options=['Multiplicative', 'Euclidean'])
        self.add_input(self.risk_eq)
        self.decay_eq = inputs.Dropdown(
            args_key='decay_eq',
            helptext=(
                "Each of these represents an option of a decay equation "
                "for the buffered stressors. If stressor buffering is "
                "desired, this equation will determine the rate at which "
                "stressor data is reduced."),
            label='Decay Equation',
            options=['None', 'Linear', 'Exponential'])
        self.add_input(self.decay_eq)
        self.aoi_vector_path = inputs.File(
            args_key='aoi_vector_path',
            helptext=(
                "An OGR-supported vector file containing feature containing "
                "one or more planning regions. subregions. An optional field "
                "called `name` could be added to compute average risk values "
                "within each subregion."),
            label='Area of Interest (Vector)',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)
        self.visualize_outputs = inputs.Checkbox(
            args_key='visualize_outputs',
            helptext=(
                "Check to enable the generation of GeoJSON outputs. This "
                "could be used to visualize the risk scores on a map in the "
                "HRA visualization web application."),
            label='Generate GeoJSONs for Web Visualization')
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
