# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.recreation import recmodel_client


class Recreation(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Recreation Model',
            target=recmodel_client.execute,
            validator=recmodel_client.validate,
            localdoc='recreation.html')

        self.internet_warning = inputs.Label(
            text=(
                "Note, this computer must have an Internet connection "
                "in order to run this model."))
        self.aoi_path = inputs.File(
            args_key='aoi_path',
            helptext=(
                "An OGR-supported vector file representing the area "
                "of interest where the model will run the analysis."),
            label='Area of Interest (Vector)',
            validator=self.validator)
        self.add_input(self.aoi_path)
        self.start_year = inputs.Text(
            args_key='start_year',
            helptext='Year to start PUD calculations, date starts on Jan 1st.',
            label='Start Year (inclusive, must be >= 2005)',
            validator=self.validator)
        self.add_input(self.start_year)
        self.end_year = inputs.Text(
            args_key='end_year',
            helptext=(
                'Year to end PUD calculations, date ends and includes '
                'Dec 31st.'),
            label='End Year (inclusive, must be <= 2017)',
            validator=self.validator)
        self.add_input(self.end_year)
        self.regression_container = inputs.Container(
            args_key='compute_regression',
            expandable=True,
            expanded=True,
            label='Compute Regression')
        self.add_input(self.regression_container)
        self.predictor_table_path = inputs.File(
            args_key='predictor_table_path',
            helptext=(
                "A table that maps predictor IDs to files and their "
                "types with required headers of 'id', 'path', and "
                "'type'.  The file paths can be absolute, or relative "
                "to the table."),
            label='Predictor Table',
            validator=self.validator)
        self.regression_container.add_input(self.predictor_table_path)
        self.scenario_predictor_table_path = inputs.File(
            args_key='scenario_predictor_table_path',
            helptext=(
                "A table that maps predictor IDs to files and their "
                "types with required headers of 'id', 'path', and "
                "'type'.  The file paths can be absolute, or relative "
                "to the table."),
            label='Scenario Predictor Table (optional)',
            validator=self.validator)
        self.regression_container.add_input(self.scenario_predictor_table_path)
        self.grid_container = inputs.Container(
            args_key='grid_aoi',
            expandable=True,
            expanded=True,
            label='Grid the AOI')
        self.add_input(self.grid_container)
        self.grid_type = inputs.Dropdown(
            args_key='grid_type',
            label='Grid Type',
            options=['square', 'hexagon'])
        self.grid_container.add_input(self.grid_type)
        self.cell_size = inputs.Text(
            args_key='cell_size',
            helptext=(
                "The size of the grid units measured in the "
                "projection units of the AOI. For example, UTM "
                "projections use meters."),
            label='Cell Size',
            validator=self.validator)
        self.grid_container.add_input(self.cell_size)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi_path.args_key: self.aoi_path.value(),
            self.start_year.args_key: self.start_year.value(),
            self.end_year.args_key: self.end_year.value(),
            self.regression_container.args_key:
                self.regression_container.value(),
            self.grid_container.args_key: self.grid_container.value(),
        }

        if self.regression_container.value():
            args[self.predictor_table_path.args_key] = (
                self.predictor_table_path.value())
            args[self.scenario_predictor_table_path.args_key] = (
                self.scenario_predictor_table_path.value())

        if self.grid_container.value():
            args[self.grid_type.args_key] = self.grid_type.value()
            args[self.cell_size.args_key] = self.cell_size.value()

        return args
