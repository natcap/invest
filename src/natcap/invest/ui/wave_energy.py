# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.wave_energy


class WaveEnergy(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Wave Energy',
            target=natcap.invest.wave_energy.execute,
            validator=natcap.invest.wave_energy.validate,
            localdoc=u'wave_energy.html')

        self.wave_base_data = inputs.Folder(
            args_key=u'wave_base_data_path',
            helptext=(
                u'Select the folder that has the packaged Wave Energy '
                u'Data.'),
            label=u'Wave Base Data Folder',
            validator=self.validator)
        self.add_input(self.wave_base_data)
        self.analysis_area = inputs.Dropdown(
            args_key=u'analysis_area_path',
            helptext=(
                u"A list of analysis areas for which the model can "
                u"currently be run.  All the wave energy data needed "
                u"for these areas are pre-packaged in the WaveData "
                u"folder."),
            label=u'Analysis Area',
            options=(
                u'West Coast of North America and Hawaii',
                u'East Coast of North America and Puerto Rico',
                u'North Sea 4 meter resolution',
                u'North Sea 10 meter resolution',
                u'Australia',
                u'Global'))
        self.add_input(self.analysis_area)
        self.aoi = inputs.File(
            args_key=u'aoi_path',
            helptext=(
                u"An OGR-supported vector file containing a single "
                u"polygon representing the area of interest.  This "
                u"input is required for computing valuation and is "
                u"recommended for biophysical runs as well.  The AOI "
                u"should be projected in linear units of meters."),
            label=u'Area of Interest (Vector)',
            validator=self.validator)
        self.add_input(self.aoi)
        self.machine_perf_table = inputs.File(
            args_key=u'machine_perf_path',
            helptext=(
                u"A CSV Table that has the performance of a particular "
                u"wave energy machine at certain sea state conditions."),
            label=u'Machine Performance Table (CSV)',
            validator=self.validator)
        self.add_input(self.machine_perf_table)
        self.machine_param_table = inputs.File(
            args_key=u'machine_param_path',
            helptext=(
                u"A CSV Table that has parameter values for a wave "
                u"energy machine.  This includes information on the "
                u"maximum capacity of the device and the upper limits "
                u"for wave height and period."),
            label=u'Machine Parameter Table (CSV)',
            validator=self.validator)
        self.add_input(self.machine_param_table)
        self.dem = inputs.File(
            args_key=u'dem_path',
            helptext=(
                u"A GDAL-supported raster file containing a digital "
                u"elevation model dataset that has elevation values in "
                u"meters.  Used to get the cable distance for wave "
                u"energy transmission."),
            label=u'Global Digital Elevation Model (Raster)',
            validator=self.validator)
        self.add_input(self.dem)
        self.valuation_container = inputs.Container(
            args_key=u'valuation_container',
            expandable=True,
            expanded=False,
            label=u'Valuation')
        self.add_input(self.valuation_container)
        self.land_grid_points = inputs.File(
            args_key=u'land_gridPts_path',
            helptext=(
                u"A CSV Table that has the landing points and grid "
                u"points locations for computing cable distances."),
            label=u'Grid Connection Points File (CSV)',
            validator=self.validator)
        self.valuation_container.add_input(self.land_grid_points)
        self.machine_econ_table = inputs.File(
            args_key=u'machine_econ_path',
            helptext=(
                u"A CSV Table that has the economic parameters for the "
                u"wave energy machine."),
            label=u'Machine Economic Table (CSV)',
            validator=self.validator)
        self.valuation_container.add_input(self.machine_econ_table)
        self.number_of_machines = inputs.Text(
            args_key=u'number_of_machines',
            helptext=(
                u"An integer for how many wave energy machines will be "
                u"in the wave farm."),
            label=u'Number of Machines',
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
