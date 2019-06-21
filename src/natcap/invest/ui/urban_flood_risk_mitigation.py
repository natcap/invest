# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.urban_flood_risk_mitigation


class UrbanFloodRiskMitigation(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'UrbanFloodRiskMitigation',
            target=natcap.invest.urban_flood_risk_mitigation.execute,
            validator=natcap.invest.urban_flood_risk_mitigation.validate,
            localdoc=u'../documentation/urban_flood_risk_mitigation.html')

        self.aoi_watersheds_path = inputs.File(
            args_key=u'aoi_watersheds_path',
            helptext=(
                u"path to a shapefile of (sub)watersheds or sewersheds used "
                u"to indicate spatial area of interest."),
            label=u'Watershed Vector',
            validator=self.validator)
        self.add_input(self.aoi_watersheds_path)

        self.rainfall_depth = inputs.Text(
            args_key=u'rainfall_depth',
            label=u'Depth of rainfall in mm',
            validator=self.validator)
        self.add_input(self.rainfall_depth)

        self.lulc_path = inputs.File(
            args_key=u'lulc_path',
            helptext=u"path to a landcover raster",
            label=u'Landcover Raster',
            validator=self.validator)
        self.add_input(self.lulc_path)

        self.soils_hydrological_group_raster_path = inputs.File(
            args_key=u'soils_hydrological_group_raster_path',
            helptext=(
                u"Raster with values equal to 1, 2, 3, 4, corresponding to "
                u"soil hydrologic group A, B, C, or D, respectively (used to "
                u"derive the CN number"),
            label=u'Soils Hydrological Group Raster',
            validator=self.validator)
        self.add_input(self.soils_hydrological_group_raster_path)

        self.curve_number_table_path = inputs.File(
            args_key=u'curve_number_table_path',
            helptext=(
                u"Path to a CSV table that to map landcover codes to curve "
                u"numbers and contains at least the headers 'lucode', "
                u"'CN_A', 'CN_B', 'CN_C', 'CN_D'"),
            label=u'Biophysical Table',
            validator=self.validator)
        self.add_input(self.curve_number_table_path)

        self.built_infrastructure_vector_path = inputs.File(
            args_key=u'built_infrastructure_vector_path',
            helptext=(
                u"Path to a vector with built infrastructure footprints. "
                u"Attribute table contains a column 'Type' with integers "
                u"(e.g. 1=residential, 2=office, etc.)."),
            label=u'Built Infrastructure Vector (optional)',
            validator=self.validator)
        self.add_input(self.built_infrastructure_vector_path)

        self.infrastructure_damage_loss_table_path = inputs.File(
            args_key=u'infrastructure_damage_loss_table_path',
            helptext=(
                u"path to a a CSV table with columns 'Type' and 'Damage' "
                u"with values of built infrastructure type from the 'Type' "
                "field in the 'Built Infrastructure Vector' and potential "
                u"damage loss (in $/m^2)."),
            label=u'Built Infrastructure Damage Loss Table (optional)',
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
