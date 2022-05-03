# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.ndr.ndr
from natcap.invest.model_metadata import MODEL_METADATA


class Nutrient(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['ndr'].model_title,
            target=natcap.invest.ndr.ndr.execute,
            validator=natcap.invest.ndr.ndr.validate,
            localdoc=MODEL_METADATA['ndr'].userguide)

        self.dem_path = inputs.File(
            args_key='dem_path',
            helptext=(
                "A GDAL-supported raster file containing elevation "
                "values for each cell.  Make sure the DEM is corrected "
                "by filling in sinks, and if necessary burning "
                "hydrographic features into the elevation model "
                "(recommended when unusual streams are observed.) See "
                "the Working with the DEM section of the InVEST User's "
                "Guide for more information."),
            label='DEM (Raster)',
            validator=self.validator)
        self.add_input(self.dem_path)
        self.land_use = inputs.File(
            args_key='lulc_path',
            helptext=(
                "A GDAL-supported raster file containing integer "
                "values representing the LULC code for each cell.  The "
                "LULC code should be an integer."),
            label='Land Use (Raster)',
            validator=self.validator)
        self.add_input(self.land_use)
        self.runoff_proxy = inputs.File(
            args_key='runoff_proxy_path',
            helptext=(
                "Weighting factor to nutrient loads.  Internally this "
                "value is normalized by its average values so a "
                "variety of data can be used including precipitation "
                "or quickflow."),
            label='Nutrient Runoff Proxy (Raster)',
            validator=self.validator)
        self.add_input(self.runoff_proxy)
        self.watersheds_path = inputs.File(
            args_key='watersheds_path',
            helptext=(
                "An OGR-supported vector file containing watersheds "
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
                "'lucode', 'load_n' (or p), 'eff_n' (or p), and "
                "'crit_len_n' (or p) depending on which nutrients are "
                "selected."),
            label='Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.calc_p = inputs.Checkbox(
            args_key='calc_p',
            helptext='Select to calculate phosphorus export.',
            label='Calculate phosphorus retention')
        self.add_input(self.calc_p)
        self.calc_n = inputs.Checkbox(
            args_key='calc_n',
            helptext='Select to calcualte nitrogen export.',
            label='Calculate Nitrogen Retention')
        self.add_input(self.calc_n)
        self.threshold_flow_accumulation = inputs.Text(
            args_key='threshold_flow_accumulation',
            helptext=(
                "The number of upslope cells that must flow into a "
                "cell before it's considered part of a stream such "
                "that retention stops and the remaining export is "
                "exported to the stream.  Used to define streams from "
                "the DEM."),
            label='Threshold Flow Accumluation',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.k_param = inputs.Text(
            args_key='k_param',
            helptext='Borselli k parameter.',
            label='Borselli k Parameter',
            validator=self.validator)
        self.add_input(self.k_param)
        self.subsurface_critical_length_n = inputs.Text(
            args_key='subsurface_critical_length_n',
            helptext='',
            interactive=False,
            label='Subsurface Critical Length (Nitrogen)',
            validator=self.validator)
        self.add_input(self.subsurface_critical_length_n)
        self.subsurface_eff_n = inputs.Text(
            args_key='subsurface_eff_n',
            helptext='',
            interactive=False,
            label='Subsurface Maximum Retention Efficiency (Nitrogen)',
            validator=self.validator)
        self.add_input(self.subsurface_eff_n)

        # Set interactivity, requirement as input sufficiency changes
        self.calc_n.sufficiency_changed.connect(
            self.subsurface_critical_length_n.set_interactive)
        self.calc_n.sufficiency_changed.connect(
            self.subsurface_eff_n.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.dem_path.args_key: self.dem_path.value(),
            self.land_use.args_key: self.land_use.value(),
            self.runoff_proxy.args_key: self.runoff_proxy.value(),
            self.watersheds_path.args_key: self.watersheds_path.value(),
            self.biophysical_table_path.args_key:
                self.biophysical_table_path.value(),
            self.calc_p.args_key: self.calc_p.value(),
            self.calc_n.args_key: self.calc_n.value(),
            self.threshold_flow_accumulation.args_key:
                self.threshold_flow_accumulation.value(),
            self.k_param.args_key: self.k_param.value(),
            self.subsurface_critical_length_n.args_key:
                self.subsurface_critical_length_n.value(),
            self.subsurface_eff_n.args_key: self.subsurface_eff_n.value(),
        }

        return args
