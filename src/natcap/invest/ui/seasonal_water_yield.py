# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.seasonal_water_yield import seasonal_water_yield


class SeasonalWaterYield(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Seasonal Water Yield',
            target=seasonal_water_yield.execute,
            validator=seasonal_water_yield.validate,
            localdoc='seasonal_water_yield.html')

        self.threshold_flow_accumulation = inputs.Text(
            args_key='threshold_flow_accumulation',
            helptext=(
                "The number of upstream cells that must flow into a "
                "cell before it's considered part of a stream such "
                "that retention stops and the remaining export is "
                "exported to the stream.  Used to define streams from "
                "the DEM."),
            label='Threshold Flow Accumulation',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.et0_dir = inputs.Folder(
            args_key='et0_dir',
            helptext=(
                "The selected folder has a list of ET0 files with a "
                "specified format."),
            label='ET0 Directory',
            validator=self.validator)
        self.add_input(self.et0_dir)
        self.precip_dir = inputs.Folder(
            args_key='precip_dir',
            helptext=(
                "The selected folder has a list of monthly "
                "precipitation files with a specified format."),
            label='Precipitation Directory',
            validator=self.validator)
        self.add_input(self.precip_dir)
        self.dem_raster_path = inputs.File(
            args_key='dem_raster_path',
            helptext=(
                "A GDAL-supported raster file with an elevation value "
                "for each cell.  Make sure the DEM is corrected by "
                "filling in sinks, and if necessary burning "
                "hydrographic features into the elevation model "
                "(recommended when unusual streams are observed.) See "
                "the 'Working with the DEM' section of the InVEST "
                "User's Guide for more information."),
            label='Digital Elevation Model (Raster)',
            validator=self.validator)
        self.add_input(self.dem_raster_path)
        self.lulc_raster_path = inputs.File(
            args_key='lulc_raster_path',
            helptext=(
                "A GDAL-supported raster file, with an integer LULC "
                "code for each cell."),
            label='Land-Use/Land-Cover (Raster)',
            validator=self.validator)
        self.add_input(self.lulc_raster_path)
        self.soil_group_path = inputs.File(
            args_key='soil_group_path',
            helptext=(
                "Map of SCS soil groups (A, B, C, or D) mapped to "
                "integer values (1, 2, 3, or 4) used in combination of "
                "the LULC map to compute the CN map."),
            label='Soil Group (Raster)',
            validator=self.validator)
        self.add_input(self.soil_group_path)
        self.aoi_path = inputs.File(
            args_key='aoi_path',
            label='AOI/Watershed (Vector)',
            validator=self.validator)
        self.add_input(self.aoi_path)
        self.biophysical_table_path = inputs.File(
            args_key='biophysical_table_path',
            helptext=(
                "A CSV table containing model information "
                "corresponding to each of the land use classes in the "
                "LULC raster input.  It must contain the fields "
                "'lucode', and 'Kc'."),
            label='Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.rain_events_table_path = inputs.File(
            args_key='rain_events_table_path',
            label='Rain Events Table (CSV)',
            validator=self.validator)
        self.add_input(self.rain_events_table_path)
        self.alpha_m = inputs.Text(
            args_key='alpha_m',
            label='alpha_m Parameter',
            validator=self.validator)
        self.add_input(self.alpha_m)
        self.beta_i = inputs.Text(
            args_key='beta_i',
            label='beta_i Parameter',
            validator=self.validator)
        self.add_input(self.beta_i)
        self.gamma = inputs.Text(
            args_key='gamma',
            label='gamma Parameter',
            validator=self.validator)
        self.add_input(self.gamma)
        self.climate_zone_container = inputs.Container(
            args_key='user_defined_climate_zones',
            expandable=True,
            label='Climate Zones (Advanced)',
            expanded=False)
        self.add_input(self.climate_zone_container)
        self.climate_zone_table_path = inputs.File(
            args_key='climate_zone_table_path',
            label='Climate Zone Table (CSV)',
            validator=self.validator)
        self.climate_zone_container.add_input(self.climate_zone_table_path)
        self.climate_zone_raster_path = inputs.File(
            args_key='climate_zone_raster_path',
            helptext=(
                "Map of climate zones that are found in the Climate "
                "Zone Table input.  Pixel values correspond to cz_id."),
            label='Climate Zone (Raster)',
            validator=self.validator)
        self.climate_zone_container.add_input(self.climate_zone_raster_path)
        self.user_defined_local_recharge_container = inputs.Container(
            args_key='user_defined_local_recharge',
            expandable=True,
            label='User Defined Recharge Layer (Advanced)',
            expanded=False)
        self.add_input(self.user_defined_local_recharge_container)
        self.l_path = inputs.File(
            args_key='l_path',
            label='Local Recharge (Raster)',
            validator=self.validator)
        self.user_defined_local_recharge_container.add_input(self.l_path)
        self.monthly_alpha_container = inputs.Container(
            args_key='monthly_alpha',
            expandable=True,
            label='Monthly Alpha Table (Advanced)',
            expanded=False)
        self.add_input(self.monthly_alpha_container)
        self.monthly_alpha_path = inputs.File(
            args_key='monthly_alpha_path',
            label='Monthly Alpha Table (csv)',
            validator=self.validator)
        self.monthly_alpha_container.add_input(self.monthly_alpha_path)

        # Set interactivity, requirement as input sufficiency changes
        self.user_defined_local_recharge_container.sufficiency_changed.connect(
            self._toggle_user_defined_local_recharge)
        self.monthly_alpha_container.sufficiency_changed.connect(
            self._toggle_monthly_alpha)
        self.climate_zone_container.sufficiency_changed.connect(
            self._toggle_climate_zone)

    def _toggle_climate_zone(self, use_climate_zones):
        self.rain_events_table_path.set_interactive(not use_climate_zones)

    def _toggle_user_defined_local_recharge(self, use_local_recharge):
        self.et0_dir.set_interactive(not use_local_recharge)
        self.precip_dir.set_interactive(not use_local_recharge)
        self.soil_group_path.set_interactive(not use_local_recharge)

    def _toggle_monthly_alpha(self, use_monthly_alpha):
        self.alpha_m.set_interactive(not use_monthly_alpha)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.threshold_flow_accumulation.args_key:
                self.threshold_flow_accumulation.value(),
            self.et0_dir.args_key: self.et0_dir.value(),
            self.precip_dir.args_key: self.precip_dir.value(),
            self.dem_raster_path.args_key: self.dem_raster_path.value(),
            self.lulc_raster_path.args_key: self.lulc_raster_path.value(),
            self.soil_group_path.args_key: self.soil_group_path.value(),
            self.aoi_path.args_key: self.aoi_path.value(),
            self.biophysical_table_path.args_key:
                self.biophysical_table_path.value(),
            self.rain_events_table_path.args_key:
                self.rain_events_table_path.value(),
            self.alpha_m.args_key: self.alpha_m.value(),
            self.beta_i.args_key: self.beta_i.value(),
            self.gamma.args_key: self.gamma.value(),
            self.climate_zone_container.args_key:
                self.climate_zone_container.value(),
            self.user_defined_local_recharge_container.args_key:
                self.user_defined_local_recharge_container.value(),
            self.monthly_alpha_container.args_key:
                self.monthly_alpha_container.value(),
        }

        if self.user_defined_local_recharge_container.value():
            args[self.l_path.args_key] = self.l_path.value()

        if self.climate_zone_container.value():
            args[self.climate_zone_table_path.args_key] = (
                self.climate_zone_table_path.value())
            args[self.climate_zone_raster_path.args_key] = (
                self.climate_zone_raster_path.value())

        if self.monthly_alpha_container.value():
            args[self.monthly_alpha_path.args_key] = (
                self.monthly_alpha_path.value())

        return args
