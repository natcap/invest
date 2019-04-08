# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.urban_heat_island_mitigation


class UrbanHeatIslandMitigation(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'UrbanHeatIslandMitigation',
            target=natcap.invest.urban_heat_island_mitigation.execute,
            validator=natcap.invest.urban_heat_island_mitigation.validate,
            localdoc=u'../documentation/urban_heat_island_mitigation.html')

        self.t_air_ref_raster_path = inputs.File(
            args_key='t_air_ref_raster_path',
            helptext=('raster of air temperature.'),
            label='t_air_ref_raster_path',
            validator=self.validator)
        self.add_input(self.t_air_ref_raster_path)

        self.lulc_raster_path = inputs.File(
            args_key='lulc_raster_path',
            helptext=('path to landcover raster.'),
            label='lulc_raster_path',
            validator=self.validator)
        self.add_input(self.lulc_raster_path)

        self.ref_eto_raster_path = inputs.File(
            args_key='ref_eto_raster_path',
            helptext=('path to evapotranspiration raster.'),
            label='ref_eto_raster_path',
            validator=self.validator)
        self.add_input(self.ref_eto_raster_path)

        self.aoi_vector_path = inputs.File(
            args_key='aoi_vector_path',
            helptext=('path to desired AOI.'),
            label='aoi_vector_path',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)

        self.biophysical_table_path = inputs.File(
            args_key='biophysical_table_path',
            helptext=("table to map landcover codes to Shade, Kc, and Albed values. Must contain the fields 'lucode', 'shade', 'kc', and 'albedo'."),
            label='biophysical_table_path',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)

        self.building_vector_path = inputs.File(
            args_key='building_vector_path',
            helptext=("path to a vector of building footprints that contains at least the field 'type'."),
            label='building_vector_path',
            validator=self.validator)
        self.add_input(self.building_vector_path)

        self.energy_consumption_table_path = inputs.File(
            args_key='energy_consumption_table_path',
            helptext=("path to a table that maps building types to energy consumption. Must contain at least the fields 'type' and 'consumption'."),
            label='energy_consumption_table_path',
            validator=self.validator)
        self.add_input(self.energy_consumption_table_path)

        self.uhi_max = inputs.Text(
            args_key='uhi_max',
            label='Magnitude of the UHI effect.',
            validator=self.validator)
        self.add_input(self.uhi_max)
        self.uhi_max.set_value("3.5")

        self.t_air_average_radius = inputs.Text(
            args_key='t_air_average_radius',
            label='T_air moving average radius (m).',
            validator=self.validator)
        self.add_input(self.t_air_average_radius)
        self.t_air_average_radius.set_value("1000")

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.t_air_ref_raster_path.args_key: self.t_air_ref_raster_path.value(),
            self.lulc_raster_path.args_key: self.lulc_raster_path.value(),
            self.ref_eto_raster_path.args_key: self.ref_eto_raster_path.value(),
            self.aoi_vector_path.args_key: self.aoi_vector_path.value(),
            self.biophysical_table_path.args_key: self.biophysical_table_path.value(),
            self.building_vector_path.args_key: self.building_vector_path.value(),
            self.energy_consumption_table_path.args_key: self.energy_consumption_table_path.value(),
            self.uhi_max.args_key: self.uhi_max.value(),
            self.t_air_average_radius.args_key: self.t_air_average_radius.value(),
        }

        return args
