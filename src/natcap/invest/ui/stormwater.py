from natcap.invest.ui import model, inputs
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import stormwater


class Stormwater(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['stormwater'].model_title,
            target=stormwater.execute,
            validator=stormwater.validate,
            localdoc=MODEL_METADATA['stormwater'].userguide)

        self.lulc_path = inputs.File(
            args_key='lulc_path',
            helptext=(
                "A GDAL-supported raster representing land-cover."),
            label='Land Use/Land Cover',
            validator=self.validator)
        self.add_input(self.lulc_path)
        self.soil_group_path = inputs.File(
            args_key='soil_group_path',
            helptext=(
                "A GDAL-supported raster representing hydrologic soil groups."),
            label='Soil Groups',
            validator=self.validator)
        self.add_input(self.soil_group_path)
        self.precipitation_path = inputs.File(
            args_key='precipitation_path',
            helptext=(
                "A GDAL-supported raster showing annual precipitation amounts"),
            label='Precipitation',
            validator=self.validator)
        self.add_input(self.precipitation_path)
        self.biophysical_table = inputs.File(
            args_key='biophysical_table',
            helptext=(
                "A CSV file with runoff coefficient (RC), infiltration "
                "coefficient (IR), and pollutant event mean concentration "
                "(EMC) data for each LULC code."),
            label='Biophysical Table',
            validator=self.validator)
        self.add_input(self.biophysical_table)
        self.adjust_retention_ratios = inputs.Checkbox(
            args_key='adjust_retention_ratios',
            helptext=(
                'If checked, adjust retention ratios using road centerlines.'),
            label='Adjust Retention Ratios')
        self.add_input(self.adjust_retention_ratios)
        self.retention_radius = inputs.Text(
            args_key='retention_radius',
            helptext=('Radius within which to adjust retention ratios'),
            label='Retention Radius',
            validator=self.validator)
        self.add_input(self.retention_radius)
        self.road_centerlines_path = inputs.File(
            args_key='road_centerlines_path',
            helptext=('Polyline vector representing centerlines of roads'),
            label='Road Centerlines',
            validator=self.validator)
        self.add_input(self.road_centerlines_path)
        self.aggregate_areas_path = inputs.File(
            args_key='aggregate_areas_path',
            helptext=(
                'Polygon vector outlining area(s) of interest by which to '
                'aggregate results (typically watersheds or sewersheds).'),
            label='Aggregate Areas',
            validator=self.validator)
        self.add_input(self.aggregate_areas_path)
        self.replacement_cost = inputs.Text(
            args_key='replacement_cost',
            helptext=('Replacement cost of retention per cubic meter'),
            label='Replacement Cost',
            validator=self.validator)
        self.add_input(self.replacement_cost)

        # retention_radius is active when adjust_retention_ratios is checked
        self.adjust_retention_ratios.sufficiency_changed.connect(
            self.retention_radius.set_interactive)
        self.adjust_retention_ratios.sufficiency_changed.connect(
            self.road_centerlines_path.set_interactive)
    

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.lulc_path.args_key: self.lulc_path.value(),
            self.soil_group_path.args_key: self.soil_group_path.value(),
            self.precipitation_path.args_key: self.precipitation_path.value(),
            self.biophysical_table.args_key: self.biophysical_table.value(),
            self.adjust_retention_ratios.args_key: self.adjust_retention_ratios.value(),
            self.retention_radius.args_key: self.retention_radius.value(),
            self.road_centerlines_path.args_key: self.road_centerlines_path.value(),
            self.aggregate_areas_path.args_key: self.aggregate_areas_path.value(),
            self.replacement_cost.args_key: self.replacement_cost.value(),
        }

        return args

