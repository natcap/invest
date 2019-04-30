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

        self.t_ref = inputs.File(
            args_key='t_ref',
            helptext=('Reference air temperature (real).'),
            label='t_ref',
            validator=self.validator)
        self.t_ref.set_value("21.5")
        self.add_input(self.t_ref)

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
        self.t_air_average_radius.set_value("2000")

        self.green_area_cooling_distance = inputs.Text(
            args_key='green_area_cooling_distance',
            label='Green area max cooling distance effect (m).',
            validator=self.validator)
        self.add_input(self.green_area_cooling_distance)
        self.green_area_cooling_distance.set_value("1000")

        self.valuation_container = inputs.Container(
            args_key=u'do_valuation',
            expandable=True,
            expanded=True,
            interactive=True,
            label=u'Run Valuation Model')
        self.add_input(self.valuation_container)

        self.building_vector_path = inputs.File(
            args_key='building_vector_path',
            helptext=("path to a vector of building footprints that contains at least the field 'type'."),
            label='building_vector_path',
            validator=self.validator)
        self.valuation_container.add_input(self.building_vector_path)

        self.avg_rel_humidity = inputs.Text(
            args_key='avg_rel_humidity',
            label='Average relative humidity (0-100%)',
            validator=self.validator)
        self.valuation_container.add_input(self.avg_rel_humidity)
        self.avg_rel_humidity.set_value("30")

        self.energy_consumption_table_path = inputs.File(
            args_key='energy_consumption_table_path',
            helptext=("path to a table that maps building types to energy consumption. Must contain at least the fields 'type' and 'consumption'."),
            label='energy_consumption_table_path',
            validator=self.validator)
        self.valuation_container.add_input(self.energy_consumption_table_path)

        self.cc_weight_shade = inputs.Text(
            args_key='cc_weight_shade',
            helptext=("Shade weight for cooling capacity index. The default value of 0.6 is fine unless you know what you are doing."),
            label='cc_weight_shade',
            validator=self.validator)
        self.add_input(self.cc_weight_shade)
        self.cc_weight_shade.set_value("0.6")

        self.cc_weight_albedo = inputs.Text(
            args_key='cc_weight_albedo',
            helptext=("Albedo weight for cooling capacity index. The default value of 0.2 is fine unless you know what you are doing."),
            label='cc_weight_albedo',
            validator=self.validator)
        self.add_input(self.cc_weight_albedo)
        self.cc_weight_albedo.set_value("0.2")

        self.cc_weight_eti = inputs.Text(
            args_key='cc_weight_eti',
            helptext=("Evapotranspiration index weight for cooling capacity index. The default value of 0.2 is fine unless you know what you are doing."),
            label='cc_weight_eti',
            validator=self.validator)
        self.add_input(self.cc_weight_eti)
        self.cc_weight_eti.set_value("0.2")


    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.t_ref.args_key: self.t_ref.value(),
            self.lulc_raster_path.args_key: self.lulc_raster_path.value(),
            self.ref_eto_raster_path.args_key: self.ref_eto_raster_path.value(),
            self.aoi_vector_path.args_key: self.aoi_vector_path.value(),
            self.biophysical_table_path.args_key: self.biophysical_table_path.value(),
            self.uhi_max.args_key: self.uhi_max.value(),
            self.t_air_average_radius.args_key: self.t_air_average_radius.value(),
            self.green_area_cooling_distance.args_key: self.green_area_cooling_distance.value(),
            self.cc_weight_shade.args_key: self.cc_weight_shade.value(),
            self.cc_weight_albedo.args_key: self.cc_weight_albedo.value(),
            self.cc_weight_eti.args_key: self.cc_weight_eti.value(),
        }
        if self.valuation_container.value():
            args[self.energy_consumption_table_path.args_key] = (
                self.energy_consumption_table_path.value())
            args[self.avg_rel_humidity.args_key] = (
                self.avg_rel_humidity.value())
            args[self.building_vector_path.args_key] = (
                self.building_vector_path.value())

        return args
