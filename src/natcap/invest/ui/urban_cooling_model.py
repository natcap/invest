# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.urban_cooling_model


class UrbanCoolingModel(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Urban Cooling Model',
            target=natcap.invest.urban_cooling_model.execute,
            validator=natcap.invest.urban_cooling_model.validate,
            localdoc='urban_cooling_model.html')

        self.lulc_raster_path = inputs.File(
            args_key='lulc_raster_path',
            helptext=('Path to landcover raster.'),
            label='Land Use / Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.lulc_raster_path)

        self.ref_eto_raster_path = inputs.File(
            args_key='ref_eto_raster_path',
            helptext=('Path to evapotranspiration raster.'),
            label='Reference Evapotranspiration (Raster)',
            validator=self.validator)
        self.add_input(self.ref_eto_raster_path)

        self.aoi_vector_path = inputs.File(
            args_key='aoi_vector_path',
            helptext=('Path to desired AOI.'),
            label='Area of Interest (Vector)',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)

        self.biophysical_table_path = inputs.File(
            args_key='biophysical_table_path',
            helptext=(
                "Table to map landcover codes to Shade, Kc, and Albedo "
                "values. Must contain the fields 'lucode', 'shade', 'kc', "
                "and 'albedo'."),
            label='Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)

        self.t_ref = inputs.Text(
            args_key='t_ref',
            helptext=('Reference air temperature (real).'),
            label='Baseline air temperature (°C)',
            validator=self.validator)
        self.t_ref.set_value("21.5")
        self.add_input(self.t_ref)

        self.uhi_max = inputs.Text(
            args_key='uhi_max',
            label='Magnitude of the UHI effect (°C)',
            helptext=(
                "The magnitude of the urban heat island effect, in degrees "
                "C.  Example: the difference between the rural reference "
                "temperature and the maximum temperature observed in the "
                "city."),
            validator=self.validator)
        self.add_input(self.uhi_max)
        self.uhi_max.set_value("3.5")

        self.t_air_average_radius = inputs.Text(
            args_key='t_air_average_radius',
            label='Air Temperature Maximum Blending Distance (m).',
            helptext=(
                "Radius of the averaging filter for turning T_air_nomix "
                "into T_air"),
            validator=self.validator)
        self.add_input(self.t_air_average_radius)
        self.t_air_average_radius.set_value("2000")

        self.green_area_cooling_distance = inputs.Text(
            args_key='green_area_cooling_distance',
            label='Green Area Maximum Cooling Distance (m).',
            helptext=(
                "Distance (in m) over which large green areas (> 2 ha) "
                "will have a cooling effect."),
            validator=self.validator)
        self.add_input(self.green_area_cooling_distance)
        self.green_area_cooling_distance.set_value("1000")

        self.cc_method = inputs.Dropdown(
            label='Cooling Capacity Calculation Method',
            args_key='cc_method',
            helptext=(
                'The method selected here determines the predictor used for '
                'air temperature.  If <b>"Weighted Factors"</b> is '
                'selected, the Cooling Capacity calculations will use the '
                'weighted factors for shade, albedo and ETI as a predictor '
                'for daytime temperatures. <br/>'
                'Alternatively, if <b>"Building Intensity"</b> is selected, '
                'building intensity will be used as a predictor for nighttime '
                'temperature instead of shade, albedo and ETI.'
            ),
            options=('Weighted Factors', 'Building Intensity'),
            return_value_map={
                'Weighted Factors': 'factors',
                'Building Intensity': 'intensity',
            })
        self.cc_method.value_changed.connect(
            self._enable_cc_options)
        self.add_input(self.cc_method)

        self.valuation_container = inputs.Container(
            args_key='do_valuation',
            expandable=True,
            expanded=True,
            interactive=True,
            label='Run Valuation Model')
        self.add_input(self.valuation_container)

        self.building_vector_path = inputs.File(
            args_key='building_vector_path',
            helptext=(
                "Path to a vector of building footprints that contains at "
                "least the field 'type'."),
            label='Building Footprints (Vector)',
            validator=self.validator)
        self.valuation_container.add_input(self.building_vector_path)

        self.avg_rel_humidity = inputs.Text(
            args_key='avg_rel_humidity',
            label='Average relative humidity (0-100%)',
            helptext=(
                "The average relative humidity (0-100%)."),
            validator=self.validator)
        self.valuation_container.add_input(self.avg_rel_humidity)
        self.avg_rel_humidity.set_value("30")

        self.energy_consumption_table_path = inputs.File(
            args_key='energy_consumption_table_path',
            helptext=(
                "Path to a table that maps building types to energy "
                "consumption. Must contain at least the fields 'type' "
                "and 'consumption'."),
            label='Energy Consumption Table (CSV)',
            validator=self.validator)
        self.valuation_container.add_input(self.energy_consumption_table_path)

        self.cooling_capacity_container = inputs.Container(
            expandable=True,
            expanded=True,
            interactive=True,
            label='Manually Adjust Cooling Capacity Index Weights'
        )
        self.add_input(self.cooling_capacity_container)

        self.cc_weight_shade = inputs.Text(
            args_key='cc_weight_shade',
            helptext=("Shade weight for cooling capacity index. "
                      "Default: 0.6"),
            label='Shade',
            validator=self.validator)
        self.cooling_capacity_container.add_input(self.cc_weight_shade)
        self.cc_weight_shade.set_value("0.6")

        self.cc_weight_albedo = inputs.Text(
            args_key='cc_weight_albedo',
            helptext=("Albedo weight for cooling capacity index. "
                      "Default: 0.2"),
            label='Albedo',
            validator=self.validator)
        self.cooling_capacity_container.add_input(self.cc_weight_albedo)
        self.cc_weight_albedo.set_value("0.2")

        self.cc_weight_eti = inputs.Text(
            args_key='cc_weight_eti',
            helptext=("Evapotranspiration index weight for cooling capacity. "
                      "Default: 0.2"),
            label='Evapotranspiration Index',
            validator=self.validator)
        self.cooling_capacity_container.add_input(self.cc_weight_eti)
        self.cc_weight_eti.set_value("0.2")

    def _enable_cc_options(self, new_value=None):
        """Enable the Cooling Capacity options based on the CC method."""
        self.cooling_capacity_container.set_interactive(
            self.cc_method.value() == 'factors')

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.t_ref.args_key: self.t_ref.value(),
            self.lulc_raster_path.args_key: self.lulc_raster_path.value(),
            self.ref_eto_raster_path.args_key: self.ref_eto_raster_path.value(),
            self.aoi_vector_path.args_key: self.aoi_vector_path.value(),
            self.biophysical_table_path.args_key: self.biophysical_table_path.value(),
            self.uhi_max.args_key: self.uhi_max.value(),
            self.t_air_average_radius.args_key: self.t_air_average_radius.value(),
            self.green_area_cooling_distance.args_key: self.green_area_cooling_distance.value(),
            self.cc_method.args_key: self.cc_method.value(),
            self.cc_weight_shade.args_key: self.cc_weight_shade.value(),
            self.cc_weight_albedo.args_key: self.cc_weight_albedo.value(),
            self.cc_weight_eti.args_key: self.cc_weight_eti.value(),
            self.valuation_container.args_key: self.valuation_container.value(),
        }
        if self.valuation_container.value():
            args[self.energy_consumption_table_path.args_key] = (
                self.energy_consumption_table_path.value())
            args[self.avg_rel_humidity.args_key] = (
                self.avg_rel_humidity.value())
            args[self.building_vector_path.args_key] = (
                self.building_vector_path.value())

        return args
