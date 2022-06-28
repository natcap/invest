# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import wave_energy


class WaveEnergy(model.InVESTModel):
    def __init__(self):
        analysis_area_options = {
            val['display_name']: key for key, val in
            wave_energy.ARGS_SPEC['args']['analysis_area']['options'].items()
        }
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['wave_energy'].model_title,
            target=wave_energy.execute,
            validator=wave_energy.validate,
            localdoc=MODEL_METADATA['wave_energy'].userguide)

        self.wave_base_data = inputs.Folder(
            args_key='wave_base_data_path',
            helptext=(
                'Select the folder that has the packaged Wave Energy '
                'Data.'),
            label='Wave Base Data Folder',
            validator=self.validator)
        self.add_input(self.wave_base_data)
        self.analysis_area = inputs.Dropdown(
            args_key='analysis_area',
            helptext=(
                "A list of analysis areas for which the model can "
                "currently be run.  All the wave energy data needed "
                "for these areas are pre-packaged in the WaveData "
                "folder."),
            label='Analysis Area',
            options=(
                'West Coast of North America and Hawaii',
                'East Coast of North America and Puerto Rico',
                'North Sea 4 meter resolution',
                'North Sea 10 meter resolution',
                'Australia',
                'Global'),
            return_value_map=analysis_area_options)
        self.add_input(self.analysis_area)
        self.aoi = inputs.File(
            args_key='aoi_path',
            helptext=(
                "An OGR-supported vector file containing a single "
                "polygon representing the area of interest.  This "
                "input is required for computing valuation and is "
                "recommended for biophysical runs as well.  The AOI "
                "should be projected in linear units of meters."),
            label='Area of Interest (Vector)',
            validator=self.validator)
        self.add_input(self.aoi)
        self.machine_perf_table = inputs.File(
            args_key='machine_perf_path',
            helptext=(
                "A CSV Table that has the performance of a particular "
                "wave energy machine at certain sea state conditions."),
            label='Machine Performance Table (CSV)',
            validator=self.validator)
        self.add_input(self.machine_perf_table)
        self.machine_param_table = inputs.File(
            args_key='machine_param_path',
            helptext=(
                "A CSV Table that has parameter values for a wave "
                "energy machine.  This includes information on the "
                "maximum capacity of the device and the upper limits "
                "for wave height and period."),
            label='Machine Parameter Table (CSV)',
            validator=self.validator)
        self.add_input(self.machine_param_table)
        self.dem = inputs.File(
            args_key='dem_path',
            helptext=(
                "A GDAL-supported raster file containing a digital "
                "elevation model dataset that has elevation values in "
                "meters.  Used to get the cable distance for wave "
                "energy transmission."),
            label='Global Digital Elevation Model (Raster)',
            validator=self.validator)
        self.add_input(self.dem)
        self.valuation_container = inputs.Container(
            args_key='valuation_container',
            expandable=True,
            expanded=False,
            label='Valuation')
        self.add_input(self.valuation_container)
        self.land_grid_points = inputs.File(
            args_key='land_gridPts_path',
            helptext=(
                "A CSV Table that has the landing points and grid "
                "points locations for computing cable distances."),
            label='Grid Connection Points File (CSV)',
            validator=self.validator)
        self.valuation_container.add_input(self.land_grid_points)
        self.machine_econ_table = inputs.File(
            args_key='machine_econ_path',
            helptext=(
                "A CSV Table that has the economic parameters for the "
                "wave energy machine."),
            label='Machine Economic Table (CSV)',
            validator=self.validator)
        self.valuation_container.add_input(self.machine_econ_table)
        self.number_of_machines = inputs.Text(
            args_key='number_of_machines',
            helptext=(
                "An integer for how many wave energy machines will be "
                "in the wave farm."),
            label='Number of Machines',
            validator=self.validator)
        self.valuation_container.add_input(self.number_of_machines)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.wave_base_data.args_key: self.wave_base_data.value(),
            self.analysis_area.args_key: self.analysis_area.value(),
            self.machine_perf_table.args_key: self.machine_perf_table.value(),
            self.machine_param_table.args_key: self.machine_param_table.value(),
            self.dem.args_key: self.dem.value(),
            self.valuation_container.args_key: self.valuation_container.value(),
        }

        if self.aoi.value():
            args[self.aoi.args_key] = self.aoi.value()
        if self.valuation_container.value():
            args[self.land_grid_points.args_key] = self.land_grid_points.value()
            args[self.machine_econ_table.args_key] = (
                self.machine_econ_table.value())
            args[self.number_of_machines.args_key] = (
                self.number_of_machines.value())

        return args
