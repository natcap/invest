# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.sdr.sdr
from natcap.invest.model_metadata import MODEL_METADATA


class SDR(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['sdr'].model_title,
            target=natcap.invest.sdr.sdr.execute,
            validator=natcap.invest.sdr.sdr.validate,
            localdoc=MODEL_METADATA['sdr'].userguide)
        self.dem_path = inputs.File(
            args_key='dem_path',
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
        self.add_input(self.dem_path)
        self.erosivity_path = inputs.File(
            args_key='erosivity_path',
            helptext=(
                "A GDAL-supported raster file, with an erosivity "
                "index value for each cell.  This variable depends on "
                "the intensity and duration of rainfall in the area of "
                "interest.  The greater the intensity and duration of "
                "the rain storm, the higher the erosion potential. "
                "The erosivity index is widely used, but in case of "
                "its absence, there are methods and equations to help "
                "generate a grid using climatic data.  The units are "
                "MJ*mm/(ha*h*yr)."),
            label='Rainfall Erosivity Index (R) (Raster)',
            validator=self.validator)
        self.add_input(self.erosivity_path)
        self.erodibility_path = inputs.File(
            args_key='erodibility_path',
            helptext=(
                "A GDAL-supported raster file, with a soil "
                "erodibility value for each cell which is a measure of "
                "the susceptibility of soil particles to detachment "
                "and transport by rainfall and runoff.  Units are in "
                "T*ha*h/(ha*MJ*mm)."),
            label='Soil Erodibility (Raster)',
            validator=self.validator)
        self.add_input(self.erodibility_path)
        self.lulc_path = inputs.File(
            args_key='lulc_path',
            helptext=(
                "A GDAL-supported raster file, with an integer LULC "
                "code for each cell."),
            label='Land-Use/Land-Cover (Raster)',
            validator=self.validator)
        self.add_input(self.lulc_path)
        self.watersheds_path = inputs.File(
            args_key='watersheds_path',
            helptext=(
                "This is a layer of polygons representing watersheds "
                "such that each watershed contributes to a point of "
                "interest where water quality will be analyzed.  It "
                "must have the integer field 'ws_id' where the values "
                "uniquely identify each watershed."),
            label='Watersheds (Vector)',
            validator=self.validator)
        self.add_input(self.watersheds_path)
        self.biophysical_table_path = inputs.File(
            args_key='biophysical_table_path',
            helptext=(
                "A CSV table containing model information "
                "corresponding to each of the land use classes in the "
                "LULC raster input.  It must contain the fields "
                "'lucode', 'usle_c', and 'usle_p'.  See the InVEST "
                "Sediment User's Guide for more information about "
                "these fields."),
            label='Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.threshold_flow_accumulation = inputs.Text(
            args_key='threshold_flow_accumulation',
            helptext=(
                "The number of upslope cells that must flow into a "
                "cell before it's considered part of a stream such "
                "that retention stops and the remaining export is "
                "exported to the stream.  Used to define streams from "
                "the DEM."),
            label='Threshold Flow Accumulation',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.drainage_path = inputs.File(
            args_key='drainage_path',
            helptext=(
                "An optional GDAL-supported raster file mask, that "
                "indicates areas that drain to the watershed.  Format "
                "is that 1's indicate drainage areas and 0's or nodata "
                "indicate areas with no additional drainage.  This "
                "model is most accurate when the drainage raster "
                "aligns with the DEM."),
            label='Drainages (Raster) (Optional)',
            validator=self.validator)
        self.add_input(self.drainage_path)
        self.k_param = inputs.Text(
            args_key='k_param',
            helptext='Borselli k parameter.',
            label='Borselli k Parameter',
            validator=self.validator)
        self.add_input(self.k_param)
        self.ic_0_param = inputs.Text(
            args_key='ic_0_param',
            helptext='Borselli IC0 parameter.',
            label='Borselli IC0 Parameter',
            validator=self.validator)
        self.add_input(self.ic_0_param)
        self.sdr_max = inputs.Text(
            args_key='sdr_max',
            helptext='Maximum SDR value.',
            label='Max SDR Value',
            validator=self.validator)
        self.add_input(self.sdr_max)
        self.l_max = inputs.Text(
            args_key='l_max',
            helptext=(
                'L will not exceed this value. Ranges of 122-333 (unitless) '
                'are found in relevant literature.'),
            label='Max L Value',
            validator=self.validator)
        self.add_input(self.l_max)

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
            self.l_max.args_key: self.l_max.value(),
        }

        return args
