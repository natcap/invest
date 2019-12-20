# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.habitat_quality


class HabitatQuality(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Habitat Quality',
            target=natcap.invest.habitat_quality.execute,
            validator=natcap.invest.habitat_quality.validate,
            localdoc=u'../documentation/habitat_quality.html')
        self.current_landcover = inputs.File(
            args_key=u'lulc_cur_path',
            helptext=(
                u"A GDAL-supported raster file.  The current LULC must "
                u"have its' own threat rasters, where each threat "
                u"raster file path has a suffix of <b>_c</b>.<br/><br/> "
                u"Each cell should represent a LULC code as an Integer. "
                u"The dataset should be in a projection where the units "
                u"are in meters and the projection used should be "
                u"defined.  <b>The LULC codes must match the codes in "
                u"the Sensitivity table</b>."),
            label=u'Current Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.current_landcover)
        self.future_landcover = inputs.File(
            args_key=u'lulc_fut_path',
            helptext=(
                u"Optional.  A GDAL-supported raster file.  Inputting "
                u"a future LULC will generate degradation, habitat "
                u"quality, and habitat rarity (If baseline is input) "
                u"outputs.  The future LULC must have it's own threat "
                u"rasters, where each threat raster file path has a "
                u"suffix of <b>_f</b>.<br/><br/>Each cell should "
                u"represent a LULC code as an Integer.  The dataset "
                u"should be in a projection where the units are in "
                u"meters and the projection used should be defined. "
                u"<b>The LULC codes must match the codes in the "
                u"Sensitivity table</b>."),
            label=u'Future Land Cover (Raster) (Optional)',
            validator=self.validator)
        self.add_input(self.future_landcover)
        self.baseline_landcover = inputs.File(
            args_key=u'lulc_bas_path',
            helptext=(
                u"Optional.  A GDAL-supported raster file.  If the "
                u"baseline LULC is provided, rarity outputs will be "
                u"created for the current and future LULC. The baseline "
                u"LULC can have it's own threat rasters (optional), "
                u"where each threat raster file path has a suffix of "
                u"<b>_b</b>. If no threat rasters are found, "
                u"degradation and habitat quality outputs will not be "
                u"generated for the baseline LULC.<br/><br/> Each cell "
                u"should  represent a LULC code as an Integer.  The "
                u"dataset should be in a projection where the units are "
                u"in meters and the projection used should be defined. "
                u"The LULC codes must match the codes in the "
                u"Sensitivity table.  If possible the baseline map "
                u"should refer to a time when intensive management of "
                u"the landscape was relatively rare."),
            label=u'Baseline Land Cover (Raster) (Optional)',
            validator=self.validator)
        self.add_input(self.baseline_landcover)
        self.threat_rasters = inputs.Folder(
            args_key=u'threat_raster_folder',
            helptext=(
                u"The selected folder is used as the location to find "
                u"all threat rasters for the threats listed in the "
                u"below table."),
            label=u'Folder Containing Threat Rasters',
            validator=self.validator)
        self.add_input(self.threat_rasters)
        self.threats_data = inputs.File(
            args_key=u'threats_table_path',
            helptext=(
                u"A CSV file of all the threats for the model to "
                u"consider.  Each row in the table is a degradation "
                u"source and each column contains a different attribute "
                u"of each degradation source (THREAT, MAX_DIST, "
                u"WEIGHT).<br/><br/><b>THREAT:</b> The name of the "
                u"threat source and this name must match exactly to the "
                u"name of the threat raster and to the name of it's "
                u"corresponding column in the sensitivity table. "
                u"<b>NOTE:</b> The threat raster path should have a "
                u"suffix indicator ( _c, _f, _b ) and the sensitivity "
                u"column should have a prefix indicator (L_). The "
                u"THREAT name in the threat table should not include "
                u"either the suffix or prefix. "
                u"<br/><br/><b>MAX_DIST:</b> A number in kilometres "
                u"(km) for the maximum distance a threat has an "
                u"affect.<br/><br/><b>WEIGHT:</b> A floating point "
                u"value between 0 and 1 for the the threats weight "
                u"relative to the other threats.  Depending on the type "
                u"of habitat under review, certain threats may cause "
                u"greater degradation than other "
                u"threats.<br/><br/><b>DECAY:</b> A string value of "
                u"either <b>exponential</b> or <b>linear</b> "
                u"representing the type of decay over space for the "
                u"threat.<br/><br/>See the user's guide for valid "
                u"values for these columns."),
            label=u'Threats Data',
            validator=self.validator)
        self.add_input(self.threats_data)
        self.accessibility_threats = inputs.File(
            args_key=u'access_vector_path',
            helptext=(
                u"An OGR-supported vector file.  The input contains "
                u"data on the relative protection that legal / "
                u"institutional / social / physical barriers provide "
                u"against threats.  The vector file should contain "
                u"polygons with a field <b>ACCESS</b>. The "
                u"<b>ACCESS</b> values should range from 0 - 1, where 1 "
                u"is fully accessible.  Any cells not covered by a "
                u"polygon will be set to 1."),
            label=u'Accessibility to Threats (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.accessibility_threats)
        self.sensitivity_data = inputs.File(
            args_key=u'sensitivity_table_path',
            helptext=(
                u"A CSV file of LULC types, whether or not the are "
                u"considered habitat, and, for LULC types that are "
                u"habitat, their specific sensitivity to each threat. "
                u"Each row is a LULC type with the following columns: "
                u"<b>LULC, HABITAT, L_THREAT1, L_THREAT2, "
                u"...</b><br/><br/><b>LULC:</b> Integer values that "
                u"reflect each LULC code found in current, future, and "
                u"baseline rasters.<br/><br/><b>HABITAT:</b> A value of "
                u"0 or 1 (presence / absence) or a value between 0 and "
                u"1 (continuum) depicting the suitability of "
                u"habitat.<br/><br/><b>L_THREATN:</b> Each L_THREATN "
                u"should match exactly with the threat names given in "
                u"the threat CSV file, where the THREATN is the name "
                u"that matches.  This is an floating point value "
                u"between 0 and 1 that represents the sensitivity of a "
                u"habitat to a threat.  <br/><br/>.Please see the users "
                u"guide for more detailed information on proper column "
                u"values and column names for each threat."),
            label=u'Sensitivity of Land Cover Types to Each Threat, File (CSV)',
            validator=self.validator)
        self.add_input(self.sensitivity_data)
        self.half_saturation_constant = inputs.Text(
            args_key=u'half_saturation_constant',
            helptext=(
                u"A positive floating point value that is defaulted at "
                u"0.5. This is the value of the parameter k in equation "
                u"(4). In general, set k to half of the highest grid "
                u"cell degradation value on the landscape.  To perform "
                u"this model calibration the model must be run once in "
                u"order to find the highest degradation value and set k "
                u"for the provided landscape.  Note that the choice of "
                u"k only determines the spread and central tendency of "
                u"habitat quality cores and does not affect the rank."),
            label=u'Half-Saturation Constant',
            validator=self.validator)
        self.add_input(self.half_saturation_constant)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.current_landcover.args_key: self.current_landcover.value(),
            self.threat_rasters.args_key: self.threat_rasters.value(),
            self.threats_data.args_key: self.threats_data.value(),
            self.sensitivity_data.args_key: self.sensitivity_data.value(),
            self.half_saturation_constant.args_key:
                self.half_saturation_constant.value(),
        }
        if self.future_landcover.value():
            args[self.future_landcover.args_key] = self.future_landcover.value()
        if self.baseline_landcover.value():
            args[self.baseline_landcover.args_key] = (
                self.baseline_landcover.value())
        if self.accessibility_threats.value():
            args[self.accessibility_threats.args_key] = (
                self.accessibility_threats.value())

        return args
