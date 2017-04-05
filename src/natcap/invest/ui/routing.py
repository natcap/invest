# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.routing import delineateit, routedem


class Delineateit(model.Model):
    label = u'DelineateIT: Watershed Delineation'
    target = staticmethod(delineateit.execute)
    validator = staticmethod(delineateit.validate)
    localdoc = u'../documentation/delineateit.html'

    def __init__(self):
        model.Model.__init__(self)

        self.dem_uri = inputs.File(
            args_key=u'dem_uri',
            helptext=(
                u"A GDAL-supported raster file with an elevation value "
                u"for each cell.  Make sure the DEM is corrected by "
                u"filling in sinks, and if necessary burning "
                u"hydrographic features into the elevation model "
                u"(recommended when unusual streams are observed.) See "
                u"the 'Working with the DEM' section of the InVEST "
                u"User's Guide for more information."),
            label=u'Digital Elevation Model (Raster)',
            required=True,
            validator=self.validator)
        self.add_input(self.dem_uri)
        self.outlet_shapefile_uri = inputs.File(
            args_key=u'outlet_shapefile_uri',
            helptext=(
                u"This is a layer of points representing outlet points "
                u"that the watersheds should be built around."),
            label=u'Outlet Points (Vector)',
            required=True,
            validator=self.validator)
        self.add_input(self.outlet_shapefile_uri)
        self.flow_threshold = inputs.Text(
            args_key=u'flow_threshold',
            helptext=(
                u"The number of upstream cells that must flow into a "
                u"cell before it's considered part of a stream such "
                u"that retention stops and the remaining export is "
                u"exported to the stream.  Used to define streams from "
                u"the DEM."),
            label=u'Threshold Flow Accumulation',
            required=True,
            validator=self.validator)
        self.add_input(self.flow_threshold)
        self.snap_distance = inputs.Text(
            args_key=u'snap_distance',
            label=u'Pixel Distance to Snap Outlet Points',
            required=True,
            validator=self.validator)
        self.add_input(self.snap_distance)

        # Set interactivity, requirement as input sufficiency changes

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.dem_uri.args_key: self.dem_uri.value(),
            self.outlet_shapefile_uri.args_key: (
                self.outlet_shapefile_uri.value()),
            self.flow_threshold.args_key: self.flow_threshold.value(),
            self.snap_distance.args_key: self.snap_distance.value(),
        }

        return args


class RouteDEM(model.Model):
    label = u'RouteDEM'
    target = staticmethod(routedem.execute)
    validator = staticmethod(routedem.validate)
    localdoc = u'../documentation/routedem.html'

    def __init__(self):
        model.Model.__init__(self)

        self.dem = inputs.File(
            args_key=u'dem_uri',
            helptext=(
                u"A GDAL-supported raster file containing a base "
                u"Digital Elevation Model to execute the routing "
                u"functionality across."),
            label=u'Digital Elevation Model (Raster)',
            required=True,
            validator=self.validator)
        self.add_input(self.dem)
        self.pit_filled_filename = inputs.Text(
            args_key=u'pit_filled_filename',
            helptext=(
                u"The filename of the output raster with pits filled "
                u"in.  It will go in the project workspace."),
            label=u'Pit Filled DEM Filename',
            required=True,
            validator=self.validator)
        self.add_input(self.pit_filled_filename)
        self.flow_direction_filename = inputs.Text(
            args_key=u'flow_direction_filename',
            helptext=(
                u"The filename of the flow direction raster.  It will "
                u"go in the project workspace."),
            label=u'Flow Direction Filename',
            required=True,
            validator=self.validator)
        self.add_input(self.flow_direction_filename)
        self.flow_accumulation_filename = inputs.Text(
            args_key=u'flow_accumulation_filename',
            helptext=(
                u"The filename of the flow accumulation raster.  It "
                u"will go in the project workspace."),
            label=u'Flow Accumulation Filename',
            required=True,
            validator=self.validator)
        self.add_input(self.flow_accumulation_filename)
        self.threshold_flow_accumulation = inputs.Text(
            args_key=u'threshold_flow_accumulation',
            helptext=(
                u"The number of upstream cells that must flow into a "
                u"cell before it's classified as a stream."),
            label=u'Threshold Flow Accumulation',
            required=True,
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.multiple_stream_thresholds = inputs.Checkbox(
            args_key=u'multiple_stream_thresholds',
            helptext=(
                u"Select to calculate multiple stream maps.  If "
                u"enabled set stream threshold to lowest amount, then "
                u"set upper and step size thresholds."),
            label=u'Calculate Multiple Stream Thresholds')
        self.add_input(self.multiple_stream_thresholds)
        self.threshold_flow_accumulation_upper = inputs.Text(
            args_key=u'threshold_flow_accumulation_upper',
            helptext=(
                u"The number of upstream cells that must flow into a "
                u"cell before it's classified as a stream."),
            interactive=False,
            label=u'Threshold Flow Accumulation Upper Limit',
            required=True,
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation_upper)
        self.threshold_flow_accumulation_stepsize = inputs.Text(
            args_key=u'threshold_flow_accumulation_stepsize',
            helptext=(
                u'The number cells to step up from lower to upper threshold '
                u'range.'),
            interactive=False,
            label=u'Threshold Flow Accumulation Range Stepsize',
            required=True,
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation_stepsize)
        self.calculate_slope = inputs.Checkbox(
            args_key=u'calculate_slope',
            helptext=u'Select to output a slope raster.',
            label=u'Calculate Slope')
        self.add_input(self.calculate_slope)
        self.slope_filename = inputs.Text(
            args_key=u'slope_filename',
            helptext=(
                u"The filename of the output slope raster.  It will go "
                u"in the project workspace."),
            interactive=False,
            label=u'Slope Filename',
            required=True,
            validator=self.validator)
        self.add_input(self.slope_filename)
        self.calculate_downstream_distance = inputs.Checkbox(
            args_key=u'calculate_downstream_distance',
            helptext=(
                u"Select to calculate a distance to stream raster, "
                u"based on uppper threshold limit."),
            label=u'Calculate Distance to stream')
        self.add_input(self.calculate_downstream_distance)
        self.downstream_distance_filename = inputs.Text(
            args_key=u'downstream_distance_filename',
            helptext=(
                u"The filename of the output slope raster.  It will go "
                u"in the project workspace."),
            interactive=False,
            label=u'Downstream Distance Filename',
            required=True,
            validator=self.validator)
        self.add_input(self.downstream_distance_filename)

        # Set interactivity, requirement as input sufficiency changes
        self.multiple_stream_thresholds.sufficiency_changed.connect(
            self.threshold_flow_accumulation_upper.set_interactive)
        self.multiple_stream_thresholds.sufficiency_changed.connect(
            self.threshold_flow_accumulation_stepsize.set_interactive)
        self.calculate_slope.sufficiency_changed.connect(
            self.slope_filename.set_interactive)
        self.calculate_downstream_distance.sufficiency_changed.connect(
            self.downstream_distance_filename.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.dem.args_key: self.dem.value(),
            self.pit_filled_filename.args_key: self.pit_filled_filename.value(),
            self.flow_direction_filename.args_key:
                self.flow_direction_filename.value(),
            self.flow_accumulation_filename.args_key:
                self.flow_accumulation_filename.value(),
            self.threshold_flow_accumulation.args_key:
                self.threshold_flow_accumulation.value(),
            self.multiple_stream_thresholds.args_key:
                self.multiple_stream_thresholds.value(),
            self.threshold_flow_accumulation_upper.args_key:
                self.threshold_flow_accumulation_upper.value(),
            self.threshold_flow_accumulation_stepsize.args_key:
                self.threshold_flow_accumulation_stepsize.value(),
            self.calculate_slope.args_key: self.calculate_slope.value(),
            self.slope_filename.args_key: self.slope_filename.value(),
            self.calculate_downstream_distance.args_key:
                self.calculate_downstream_distance.value(),
            self.downstream_distance_filename.args_key:
                self.downstream_distance_filename.value(),
        }

        return args
