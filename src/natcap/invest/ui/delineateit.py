# encoding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest import delineateit


class Delineateit(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='DelineateIt: Watershed Delineation',
            target=delineateit.execute,
            validator=delineateit.validate,
            localdoc='delineateit.html')

        self.dem_path = inputs.File(
            label='Digital Elevation Model (Raster)',
            args_key='dem_path',
            helptext=(
                "A GDAL-supported raster file with an elevation value "
                "for each cell."),
            validator=self.validator)
        self.add_input(self.dem_path)
        self.outlet_vector_path = inputs.File(
            args_key='outlet_vector_path',
            helptext=(
                "This is a layer of geometries representing watershed "
                "outlets such as municipal water intakes or lakes."),
            label='Outlet Features (Vector)',
            validator=self.validator)
        self.add_input(self.outlet_vector_path)
        self.outlet_vector_path.value_changed.connect(
            self._enable_point_snapping_container)
        self.skip_invalid_geometry = inputs.Checkbox(
            args_key='skip_invalid_geometry',
            helptext=(
                'If this box is checked, any invalid geometries encountered '
                'in the outlet vector will not be included in the '
                'delineation.  If this box is unchecked, an invalid '
                'geometry will cause DelineateIt to error.'),
            label='Skip invalid geometries')
        self.add_input(self.skip_invalid_geometry)
        self.skip_invalid_geometry.set_value(True)

        self.snap_points_container = inputs.Container(
            label='Snap points to the nearest stream',
            expandable=True,
            expanded=False,
            interactive=False,
            args_key='snap_points')
        self.add_input(self.snap_points_container)
        self.flow_threshold = inputs.Text(
            args_key='flow_threshold',
            helptext=(
                "The number of upstream cells that must flow into a "
                "cell before it's considered part of a stream such "
                "that retention stops and the remaining export is "
                "exported to the stream.  Used to define streams from "
                "the DEM."),
            label='Threshold Flow Accumulation',
            validator=self.validator)
        self.snap_points_container.add_input(self.flow_threshold)
        self.snap_distance = inputs.Text(
            args_key='snap_distance',
            label='Pixel Distance to Snap Outlet Points',
            helptext=(
                "If provided, the maximum search radius in pixels to look "
                "for stream pixels.  If a stream pixel is found within the "
                "snap distance, the outflow point will be snapped to the "
                "center of the nearest stream pixel.  Geometries that are "
                "not points (such as Lines and Polygons) will not be "
                "snapped.  MultiPoints will also not be snapped."),
            validator=self.validator)
        self.snap_points_container.add_input(self.snap_distance)

    def _enable_point_snapping_container(self, input_valid):
        outlet_vector_path = self.outlet_vector_path.value()
        if delineateit._vector_may_contain_points(outlet_vector_path):
            self.snap_points_container.set_interactive(True)
        else:
            self.snap_points_container.set_interactive(False)
            self.snap_points_container.expanded = False

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.dem_path.args_key: self.dem_path.value(),
            self.outlet_vector_path.args_key: (
                self.outlet_vector_path.value()),
            self.snap_points_container.args_key: (
                self.snap_points_container.value()),
            self.flow_threshold.args_key: self.flow_threshold.value(),
            self.snap_distance.args_key: self.snap_distance.value(),
            self.skip_invalid_geometry.args_key: (
                self.skip_invalid_geometry.value()),
        }

        return args
