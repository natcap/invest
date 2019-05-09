# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.ndr.ndr


class Nutrient(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Nutrient Delivery Ratio Model (NDR)',
            target=natcap.invest.ndr.ndr.execute,
            validator=natcap.invest.ndr.ndr.validate,
            localdoc=u'waterpurification.html')

        self.dem_path = inputs.File(
            args_key=u'dem_path',
            helptext=(
                u"A GDAL-supported raster file containing elevation "
                u"values for each cell.  Make sure the DEM is corrected "
                u"by filling in sinks, and if necessary burning "
                u"hydrographic features into the elevation model "
                u"(recommended when unusual streams are observed.) See "
                u"the Working with the DEM section of the InVEST User's "
                u"Guide for more information."),
            label=u'DEM (Raster)',
            validator=self.validator)
        self.add_input(self.dem_path)
        self.land_use = inputs.File(
            args_key=u'lulc_path',
            helptext=(
                u"A GDAL-supported raster file containing integer "
                u"values representing the LULC code for each cell.  The "
                u"LULC code should be an integer."),
            label=u'Land Use (Raster)',
            validator=self.validator)
        self.add_input(self.land_use)
        self.runoff_proxy = inputs.File(
            args_key=u'runoff_proxy_path',
            helptext=(
                u"Weighting factor to nutrient loads.  Internally this "
                u"value is normalized by its average values so a "
                u"variety of data can be used including precipitation "
                u"or quickflow."),
            label=u'Nutrient Runoff Proxy (Raster)',
            validator=self.validator)
        self.add_input(self.runoff_proxy)
        self.watersheds_path = inputs.File(
            args_key=u'watersheds_path',
            helptext=(
                u"An OGR-supported vector file containing watersheds "
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
                u"'lucode', 'load_n' (or p), 'eff_n' (or p), and "
                u"'crit_len_n' (or p) depending on which nutrients are "
                u"selected."),
            label=u'Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.calc_p = inputs.Checkbox(
            args_key=u'calc_p',
            helptext=u'Select to calculate phosphorous export.',
            label=u'Calculate phosphorous retention')
        self.add_input(self.calc_p)
        self.calc_n = inputs.Checkbox(
            args_key=u'calc_n',
            helptext=u'Select to calcualte nitrogen export.',
            label=u'Calculate Nitrogen Retention')
        self.add_input(self.calc_n)
        self.threshold_flow_accumulation = inputs.Text(
            args_key=u'threshold_flow_accumulation',
            helptext=(
                u"The number of upstream cells that must flow into a "
                u"cell before it's considered part of a stream such "
                u"that retention stops and the remaining export is "
                u"exported to the stream.  Used to define streams from "
                u"the DEM."),
            label=u'Threshold Flow Accumluation',
            validator=self.validator)
        self.add_input(self.threshold_flow_accumulation)
        self.k_param = inputs.Text(
            args_key=u'k_param',
            helptext=u'Borselli k parameter.',
            label=u'Borselli k Parameter',
            validator=self.validator)
        self.add_input(self.k_param)
        self.subsurface_critical_length_n = inputs.Text(
            args_key=u'subsurface_critical_length_n',
            helptext=u'',
            interactive=False,
            label=u'Subsurface Critical Length (Nitrogen)',
            validator=self.validator)
        self.add_input(self.subsurface_critical_length_n)
        self.subsurface_critical_length_p = inputs.Text(
            args_key=u'subsurface_critical_length_p',
            helptext=u'',
            interactive=False,
            label=u'Subsurface Critical Length (Phosphorous)',
            validator=self.validator)
        self.add_input(self.subsurface_critical_length_p)
        self.subsurface_eff_n = inputs.Text(
            args_key=u'subsurface_eff_n',
            helptext=u'',
            interactive=False,
            label=u'Subsurface Maximum Retention Efficiency (Nitrogen)',
            validator=self.validator)
        self.add_input(self.subsurface_eff_n)
        self.subsurface_eff_p = inputs.Text(
            args_key=u'subsurface_eff_p',
            helptext=u'',
            interactive=False,
            label=u'Subsurface Maximum Retention Efficiency (Phosphorous)',
            validator=self.validator)
        self.add_input(self.subsurface_eff_p)

        # Set interactivity, requirement as input sufficiency changes
        self.calc_n.sufficiency_changed.connect(
            self.subsurface_critical_length_n.set_interactive)
        self.calc_p.sufficiency_changed.connect(
            self.subsurface_critical_length_p.set_interactive)
        self.calc_n.sufficiency_changed.connect(
            self.subsurface_eff_n.set_interactive)
        self.calc_p.sufficiency_changed.connect(
            self.subsurface_eff_p.set_interactive)

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
            self.subsurface_critical_length_p.args_key:
                self.subsurface_critical_length_p.value(),
            self.subsurface_eff_n.args_key: self.subsurface_eff_n.value(),
            self.subsurface_eff_p.args_key: self.subsurface_eff_p.value(),
        }

        return args
