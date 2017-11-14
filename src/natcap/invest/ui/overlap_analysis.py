# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.overlap_analysis import (overlap_analysis,
                                            overlap_analysis_mz)


class OverlapAnalysis(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Overlap Analysis Model: Fisheries and Recreation',
            target=overlap_analysis.execute,
            validator=overlap_analysis.validate,
            localdoc=u'../documentation/overlap_analysis.html')

        self.aoi = inputs.File(
            args_key=u'zone_layer_uri',
            helptext=(
                u"An OGR-supported vector file.  If Management Zones "
                u"is being used to analyze overlap data, this should be "
                u"a polygon shapefile containing multiple polygons. "
                u"If, on the other hand, gridding is being used in "
                u"order to separate the area, this can be a single "
                u"polygon shapefile."),
            label=u'Analysis Zones Layer (Vector)',
            validator=self.validator)
        self.add_input(self.aoi)
        self.grid_size = inputs.Text(
            args_key=u'grid_size',
            helptext=(
                u"By specifying a number in the interface, an analysis "
                u"grid within the AOI of size x size will be created."),
            label=u'Analysis Cell Size (meters)',
            validator=self.validator)
        self.add_input(self.grid_size)
        self.data_dir = inputs.Folder(
            args_key=u'overlap_data_dir_uri',
            helptext=(
                u"Users are required to specify the path on their "
                u"system to a folder containing only the input data for "
                u"the Overlap Analysis model.  Input data can be point, "
                u"line or polygon data layers indicating where in the "
                u"coastal and marine environment the human use activity "
                u"takes place."),
            label=u'Overlap Analysis Data Directory',
            validator=self.validator)
        self.add_input(self.data_dir)
        self.intra = inputs.Checkbox(
            args_key=u'do_intra',
            helptext=(
                u"Checking this box indicates that intra-activity "
                u"valuation of the data should be used.  These weights "
                u"will be retrieved from the column in the attribute "
                u"table of the shapefile specified in 'Analysis Zones "
                u"Layer' that bears the name specified in the 'Intra- "
                u"Activity Field Name' field below."),
            label=u'Intra-Activity Weighting?')
        self.add_input(self.intra)
        self.IS_field_name = inputs.Text(
            args_key=u'intra_name',
            helptext=(
                u"The column heading to look for in the activity "
                u"layers' attribute tables that gives intra-activity "
                u"weights."),
            interactive=False,
            label=u'Intra-Activity Attribute Name',
            validator=self.validator)
        self.add_input(self.IS_field_name)
        self.inter = inputs.Checkbox(
            args_key=u'do_inter',
            helptext=(
                u"Checking this box indicates that inter-activity "
                u"valuation of the data should be used.  These weights "
                u"will be derived from the data included in the CSV "
                u"provided in the 'Overlap Analysis Importance Score "
                u"Table' field."),
            label=u'Inter-Activity Weighting?')
        self.add_input(self.inter)
        self.IS_tbl = inputs.File(
            args_key=u'overlap_layer_tbl',
            helptext=(
                u"The name of the CSV table that links each provided "
                u"activity layer to the desired inter-activity weight."),
            interactive=False,
            label=u'Inter-Activity Weight Table (CSV)',
            validator=self.validator)
        self.add_input(self.IS_tbl)
        self.HU_Hubs = inputs.Checkbox(
            args_key=u'do_hubs',
            helptext=(
                u"Checking this box indicates taht a layer of human "
                u"use hubs should be used to weight the raster file "
                u"output.  This input should be in the form of a point "
                u"shapefile."),
            label=u'Human Use Hubs?')
        self.add_input(self.HU_Hubs)
        self.HU_Hub_URI = inputs.File(
            args_key=u'hubs_uri',
            helptext=(
                u"An OGR-supported vector file.  If human use hubs are "
                u"desired, this is the file that shows the hubs "
                u"themselves.  This should be a shapefile of points "
                u"where each point is a hub."),
            interactive=False,
            label=u'Points Layer of Human Use Hubs (Vector)',
            validator=self.validator)
        self.add_input(self.HU_Hub_URI)
        self.hub_decay = inputs.Text(
            args_key=u'decay_amt',
            helptext=(
                u"This number is the rate (r) of interest decay from "
                u"each of the human use hubs for use in the final "
                u"weighted raster for the function exp(-r*d) where d is "
                u"the distance from the closest hub."),
            interactive=False,
            label=u'Distance Decay Rate',
            validator=self.validator)
        self.add_input(self.hub_decay)

        # Set interactivity, requirement as input sufficiency changes
        self.intra.sufficiency_changed.connect(
            self.IS_field_name.set_interactive)
        self.inter.sufficiency_changed.connect(
            self.IS_tbl.set_interactive)
        self.HU_Hubs.sufficiency_changed.connect(
            self.HU_Hub_URI.set_interactive)
        self.HU_Hubs.sufficiency_changed.connect(
            self.hub_decay.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi.args_key: self.aoi.value(),
            self.grid_size.args_key: self.grid_size.value(),
            self.data_dir.args_key: self.data_dir.value(),
            self.intra.args_key: self.intra.value(),
            self.IS_field_name.args_key: self.IS_field_name.value(),
            self.inter.args_key: self.inter.value(),
            self.IS_tbl.args_key: self.IS_tbl.value(),
            self.HU_Hubs.args_key: self.HU_Hubs.value(),
            self.HU_Hub_URI.args_key: self.HU_Hub_URI.value(),
            self.hub_decay.args_key: self.hub_decay.value(),
        }

        return args


class OverlapAnalysisMZ(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=(u'Overlap Analysis Management Zone Model: Fisheries and '
                   u'Recreation'),
            target=overlap_analysis_mz.execute,
            validator=overlap_analysis_mz.validate,
            localdoc=u'../documentation/overlap_analysis.html')

        self.aoi = inputs.File(
            args_key=u'zone_layer_loc',
            helptext=(
                u"An OGR-supported vector file.  This should be a "
                u"vector file containing multiple polygons since the "
                u"Management Zones tool is used to analyze overlap "
                u"data."),
            label=u'Analysis Zones Layer (Vector)',
            validator=self.validator)
        self.add_input(self.aoi)
        self.data_dir = inputs.Folder(
            args_key=u'overlap_data_dir_loc',
            helptext=(
                u"The path to a folder containing ONLY the input data "
                u"for the Overlap Analysis model.  Input data can be "
                u"point, line or polygon data layers indicating where "
                u"in the coastal and marine environment the human use "
                u"activity takes place."),
            label=u'Overlap Analysis Data Directory',
            validator=self.validator)
        self.add_input(self.data_dir)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi.args_key: self.aoi.value(),
            self.data_dir.args_key: self.data_dir.value(),
        }

        return args
