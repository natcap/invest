# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest import routedem

class RouteDEM(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='RouteDEM',
            target=routedem.execute,
            validator=routedem.validate,
            localdoc='routedem.html')

        self.dem_path = inputs.File(
            args_key='dem_path',
            helptext=(
                "A GDAL-supported raster file containing a base "
                "Digital Elevation Model to execute the routing "
                "functionality across."),
            label='Digital Elevation Model (Raster)',
            validator=self.validator)
        self.add_input(self.dem_path)
        self.dem_band_index = inputs.Text(
            args_key='dem_band_index',
            helptext=(
                'The band index to use from the raster. '
                'This positive integer is 1-based.'
                'Default: 1'),
            label='Band Index (optional)',
            validator=self.validator)
        self.dem_band_index.set_value(1)
        self.add_input(self.dem_band_index)
        self.calculate_slope = inputs.Checkbox(
            args_key='calculate_slope',
            helptext='If selected, calculates slope raster.',
            label='Calculate Slope')
        self.add_input(self.calculate_slope)
        self.algorithm = inputs.Dropdown(
            args_key='algorithm',
            label='Routing Algorithm',
            helptext=(
                'The routing algorithm to use. '
                '<ul><li>D8: all water flows directly into the most downhill '
                'of each of the 8 neighbors of a cell.</li>'
                '<li>MFD: Multiple Flow Direction. Fractional flow is '
                'modelled between pixels.</li></ul>'),
            options=('D8', 'MFD'))
        self.add_input(self.algorithm)
        self.calculate_flow_direction = inputs.Checkbox(
            args_key='calculate_flow_direction',
            helptext='Select to calculate flow direction',
            label='Calculate Flow Direction')
        self.add_input(self.calculate_flow_direction)
        self.calculate_flow_accumulation = inputs.Checkbox(
            args_key='calculate_flow_accumulation',
            helptext='Select to calculate flow accumulation.',
            label='Calculate Flow Accumulation',
            interactive=False)
        self.add_input(self.calculate_flow_accumulation)
        self.calculate_stream_threshold = inputs.Checkbox(
            args_key='calculate_stream_threshold',
            helptext='Select to calculate a stream threshold to flow accumulation.',
            interactive=False,
            label='Calculate Stream Thresholds')
        self.add_input(self.calculate_stream_threshold)
        self.threshold_flow_accumulation = inputs.Text(
            args_key='threshold_flow_accumulation',
            helptext=(
                "The number of upstream cells that must flow into a "
                "cell before it's classified as a stream."),
            interactive=False,
            label='Threshold Flow Accumulation Limit',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.calculate_downstream_distance = inputs.Checkbox(
            args_key='calculate_downstream_distance',
            helptext=(
                "If selected, creates a downstream distance raster "
                "based on the thresholded flow accumulation stream "
                "classification."),
            interactive=False,
            label='Calculate Distance to stream')
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
