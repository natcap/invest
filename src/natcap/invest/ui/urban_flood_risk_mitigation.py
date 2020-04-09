# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.urban_flood_risk_mitigation


class UrbanFloodRiskMitigation(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='UrbanFloodRiskMitigation',
            target=natcap.invest.urban_flood_risk_mitigation.execute,
            validator=natcap.invest.urban_flood_risk_mitigation.validate,
            localdoc='../documentation/urban_flood_risk_mitigation.html')

        self.aoi_watersheds_path = inputs.File(
            args_key='aoi_watersheds_path',
            helptext=(
                "path to a shapefile of (sub)watersheds or sewersheds used "
                "to indicate spatial area of interest."),
            label='Watershed Vector',
            validator=self.validator)
        self.add_input(self.aoi_watersheds_path)

        self.rainfall_depth = inputs.Text(
            args_key='rainfall_depth',
            label='Depth of rainfall in mm',
            validator=self.validator)
        self.add_input(self.rainfall_depth)

        self.lulc_path = inputs.File(
            args_key='lulc_path',
            helptext="path to a landcover raster",
            label='Landcover Raster',
            validator=self.validator)
        self.add_input(self.lulc_path)

        self.soils_hydrological_group_raster_path = inputs.File(
            args_key='soils_hydrological_group_raster_path',
            helptext=(
                "Raster with values equal to 1, 2, 3, 4, corresponding to "
                "soil hydrologic group A, B, C, or D, respectively (used to "
                "derive the CN number"),
            label='Soils Hydrological Group Raster',
            validator=self.validator)
        self.add_input(self.soils_hydrological_group_raster_path)

        self.curve_number_table_path = inputs.File(
            args_key='curve_number_table_path',
            helptext=(
                "Path to a CSV table that to map landcover codes to curve "
                "numbers and contains at least the headers 'lucode', "
                "'CN_A', 'CN_B', 'CN_C', 'CN_D'"),
            label='Biophysical Table',
            validator=self.validator)
        self.add_input(self.curve_number_table_path)

        self.built_infrastructure_vector_path = inputs.File(
            args_key='built_infrastructure_vector_path',
            helptext=(
                "Path to a vector with built infrastructure footprints. "
                "Attribute table contains a column 'Type' with integers "
                "(e.g. 1=residential, 2=office, etc.)."),
            label='Built Infrastructure Vector (optional)',
            validator=self.validator)
        self.add_input(self.built_infrastructure_vector_path)

        self.infrastructure_damage_loss_table_path = inputs.File(
            args_key='infrastructure_damage_loss_table_path',
            helptext=(
                "path to a a CSV table with columns 'Type' and 'Damage' "
                "with values of built infrastructure type from the 'Type' "
                "field in the 'Built Infrastructure Vector' and potential "
                "damage loss (in $/m^2)."),
            label='Built Infrastructure Damage Loss Table (optional)',
            validator=self.validator)
        self.add_input(self.infrastructure_damage_loss_table_path)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi_watersheds_path.args_key:
                self.aoi_watersheds_path.value(),
            self.rainfall_depth.args_key: self.rainfall_depth.value(),
            self.lulc_path.args_key: self.lulc_path.value(),
            self.soils_hydrological_group_raster_path.args_key:
                self.soils_hydrological_group_raster_path.value(),
            self.curve_number_table_path.args_key:
                self.curve_number_table_path.value(),
            self.built_infrastructure_vector_path.args_key:
                self.built_infrastructure_vector_path.value(),
            self.infrastructure_damage_loss_table_path.args_key:
                self.infrastructure_damage_loss_table_path.value(),
        }

        return args
