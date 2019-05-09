# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest import routedem

class RouteDEM(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'RouteDEM',
            target=routedem.execute,
            validator=routedem.validate,
            localdoc=u'routedem.html')

        self.dem_path = inputs.File(
            args_key=u'dem_path',
            helptext=(
                u"A GDAL-supported raster file containing a base "
                u"Digital Elevation Model to execute the routing "
                u"functionality across."),
            label=u'Digital Elevation Model (Raster)',
            validator=self.validator)
        self.add_input(self.dem_path)
        self.dem_band_index = inputs.Text(
            args_key=u'dem_band_index',
            helptext=(
                u'The band index to use from the raster. '
                u'This positive integer is 1-based.'
                u'Default: 1'),
            label='Band Index (optional)',
            validator=self.validator)
        self.dem_band_index.set_value(1)
        self.add_input(self.dem_band_index)
        self.calculate_slope = inputs.Checkbox(
            args_key=u'calculate_slope',
            helptext=u'If selected, calculates slope raster.',
            label=u'Calculate Slope')
        self.add_input(self.calculate_slope)
        self.algorithm = inputs.Dropdown(
            args_key=u'algorithm',
            label=u'Routing Algorithm',
            helptext=(
                u'The routing algorithm to use. '
                u'<ul><li>D8: all water flows directly into the most downhill '
                u'of each of the 8 neighbors of a cell.</li>'
                u'<li>MFD: Multiple Flow Direction. Fractional flow is '
                u'modelled between pixels.</li></ul>'),
            options=('D8', 'MFD'))
        self.add_input(self.algorithm)
        self.calculate_flow_direction = inputs.Checkbox(
            args_key=u'calculate_flow_direction',
            helptext=u'Select to calculate flow direction',
            label=u'Calculate Flow Direction')
        self.add_input(self.calculate_flow_direction)
        self.calculate_flow_accumulation = inputs.Checkbox(
            args_key=u'calculate_flow_accumulation',
            helptext=u'Select to calculate flow accumulation.',
            label=u'Calculate Flow Accumulation',
            interactive=False)
        self.add_input(self.calculate_flow_accumulation)
        self.calculate_stream_threshold = inputs.Checkbox(
            args_key=u'calculate_stream_threshold',
            helptext=u'Select to calculate a stream threshold to flow accumulation.',
            interactive=False,
            label=u'Calculate Stream Thresholds')
        self.add_input(self.calculate_stream_threshold)
        self.threshold_flow_accumulation = inputs.Text(
            args_key=u'threshold_flow_accumulation',
            helptext=(
                u"The number of upstream cells that must flow into a "
                u"cell before it's classified as a stream."),
            interactive=False,
            label=u'Threshold Flow Accumulation Limit',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.calculate_downstream_distance = inputs.Checkbox(
            args_key=u'calculate_downstream_distance',
            helptext=(
                u"If selected, creates a downstream distance raster "
                u"based on the thresholded flow accumulation stream "
                u"classification."),
            interactive=False,
            label=u'Calculate Distance to stream')
        self.add_input(self.calculate_downstream_distance)

        # Set interactivity, requirement as input sufficiency changes
        self.calculate_flow_direction.sufficiency_changed.connect(
            self.calculate_flow_accumulation.set_interactive)
        self.calculate_flow_accumulation.sufficiency_changed.connect(
            self.calculate_stream_threshold.set_interactive)
        self.calculate_stream_threshold.sufficiency_changed.connect(
            self.threshold_flow_accumulation.set_interactive)
        self.calculate_stream_threshold.sufficiency_changed.connect(
            self.calculate_downstream_distance.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.dem_path.args_key: self.dem_path.value(),
            self.dem_band_index.args_key: self.dem_band_index.value(),
            self.algorithm.args_key: self.algorithm.value(),
            self.calculate_slope.args_key: self.calculate_slope.value(),
            self.calculate_flow_direction.args_key:
                self.calculate_flow_direction.value(),
            self.calculate_flow_accumulation.args_key:
                self.calculate_flow_accumulation.value(),
            self.calculate_stream_threshold.args_key:
                self.calculate_stream_threshold.value(),
            self.threshold_flow_accumulation.args_key:
                self.threshold_flow_accumulation.value(),
            self.calculate_downstream_distance.args_key:
                self.calculate_downstream_distance.value(),
        }
        return args
