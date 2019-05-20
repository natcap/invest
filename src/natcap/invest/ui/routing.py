# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.routing import delineateit


class Delineateit(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'DelineateIT: Watershed Delineation',
            target=delineateit.execute,
            validator=delineateit.validate,
            localdoc=u'delineateit.html')

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
            validator=self.validator)
        self.add_input(self.dem_uri)
        self.outlet_shapefile_uri = inputs.File(
            args_key=u'outlet_shapefile_uri',
            helptext=(
                u"This is a layer of points representing outlet points "
                u"that the watersheds should be built around."),
            label=u'Outlet Points (Vector)',
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
            validator=self.validator)
        self.add_input(self.flow_threshold)
        self.snap_distance = inputs.Text(
            args_key=u'snap_distance',
            label=u'Pixel Distance to Snap Outlet Points',
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
