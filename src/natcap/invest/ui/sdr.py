# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.sdr.sdr


class SDR(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Sediment Delivery Ratio Model (SDR)',
            target=natcap.invest.sdr.sdr.execute,
            validator=natcap.invest.sdr.sdr.validate,
            localdoc=u'../documentation/sdr.html')
        self.dem_path = inputs.File(
            args_key=u'dem_path',
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
        self.add_input(self.dem_path)
        self.erosivity_path = inputs.File(
            args_key=u'erosivity_path',
            helptext=(
                u"A GDAL-supported raster file, with an erosivity "
                u"index value for each cell.  This variable depends on "
                u"the intensity and duration of rainfall in the area of "
                u"interest.  The greater the intensity and duration of "
                u"the rain storm, the higher the erosion potential. "
                u"The erosivity index is widely used, but in case of "
                u"its absence, there are methods and equations to help "
                u"generate a grid using climatic data.  The units are "
                u"MJ*mm/(ha*h*yr)."),
            label=u'Rainfall Erosivity Index (R) (Raster)',
            validator=self.validator)
        self.add_input(self.erosivity_path)
        self.erodibility_path = inputs.File(
            args_key=u'erodibility_path',
            helptext=(
                u"A GDAL-supported raster file, with a soil "
                u"erodibility value for each cell which is a measure of "
                u"the susceptibility of soil particles to detachment "
                u"and transport by rainfall and runoff.  Units are in "
                u"T*ha*h/(ha*MJ*mm)."),
            label=u'Soil Erodibility (Raster)',
            validator=self.validator)
        self.add_input(self.erodibility_path)
        self.lulc_path = inputs.File(
            args_key=u'lulc_path',
            helptext=(
                u"A GDAL-supported raster file, with an integer LULC "
                u"code for each cell."),
            label=u'Land-Use/Land-Cover (Raster)',
            validator=self.validator)
        self.add_input(self.lulc_path)
        self.watersheds_path = inputs.File(
            args_key=u'watersheds_path',
            helptext=(
                u"This is a layer of polygons representing watersheds "
                u"such that each watershed contributes to a point of "
                u"interest where water quality will be analyzed.  It "
                u"must have the integer field 'ws_id' where the values "
                u"uniquely identify each watershed."),
            label=u'Watersheds (Vector)',
            validator=self.validator)
        self.add_input(self.watersheds_path)
        self.biophysical_table_path = inputs.File(
            args_key=u'biophysical_table_path',
            helptext=(
                u"A CSV table containing model information "
                u"corresponding to each of the land use classes in the "
                u"LULC raster input.  It must contain the fields "
                u"'lucode', 'usle_c', and 'usle_p'.  See the InVEST "
                u"Sediment User's Guide for more information about "
                u"these fields."),
            label=u'Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.threshold_flow_accumulation = inputs.Text(
            args_key=u'threshold_flow_accumulation',
            helptext=(
                u"The number of upstream cells that must flow into a "
                u"cell before it's considered part of a stream such "
                u"that retention stops and the remaining export is "
                u"exported to the stream.  Used to define streams from "
                u"the DEM."),
            label=u'Threshold Flow Accumulation',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.drainage_path = inputs.File(
            args_key=u'drainage_path',
            helptext=(
                u"An optional GDAL-supported raster file mask, that "
                u"indicates areas that drain to the watershed.  Format "
                u"is that 1's indicate drainage areas and 0's or nodata "
                u"indicate areas with no additional drainage.  This "
                u"model is most accurate when the drainage raster "
                u"aligns with the DEM."),
            label=u'Drainages (Raster) (Optional)',
            validator=self.validator)
        self.add_input(self.drainage_path)
        self.k_param = inputs.Text(
            args_key=u'k_param',
            helptext=u'Borselli k parameter.',
            label=u'Borselli k Parameter',
            validator=self.validator)
        self.add_input(self.k_param)
        self.ic_0_param = inputs.Text(
            args_key=u'ic_0_param',
            helptext=u'Borselli IC0 parameter.',
            label=u'Borselli IC0 Parameter',
            validator=self.validator)
        self.add_input(self.ic_0_param)
        self.sdr_max = inputs.Text(
            args_key=u'sdr_max',
            helptext=u'Maximum SDR value.',
            label=u'Max SDR Value',
            validator=self.validator)
        self.add_input(self.sdr_max)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.dem_path.args_key: self.dem_path.value(),
            self.erosivity_path.args_key: self.erosivity_path.value(),
            self.erodibility_path.args_key: self.erodibility_path.value(),
            self.lulc_path.args_key: self.lulc_path.value(),
            self.watersheds_path.args_key: self.watersheds_path.value(),
            self.biophysical_table_path.args_key:
                self.biophysical_table_path.value(),
            self.threshold_flow_accumulation.args_key:
                self.threshold_flow_accumulation.value(),
            self.drainage_path.args_key: self.drainage_path.value(),
            self.k_param.args_key: self.k_param.value(),
            self.ic_0_param.args_key: self.ic_0_param.value(),
            self.sdr_max.args_key: self.sdr_max.value(),
        }

        return args
