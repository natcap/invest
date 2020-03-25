# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.habitat_quality


class HabitatQuality(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Habitat Quality',
            target=natcap.invest.habitat_quality.execute,
            validator=natcap.invest.habitat_quality.validate,
            localdoc='../documentation/habitat_quality.html')
        self.current_landcover = inputs.File(
            args_key='lulc_cur_path',
            helptext=(
                "A GDAL-supported raster file.  The current LULC must "
                "have its' own threat rasters, where each threat "
                "raster file path has a suffix of <b>_c</b>.<br/><br/> "
                "Each cell should represent a LULC code as an Integer. "
                "The dataset should be in a projection where the units "
                "are in meters and the projection used should be "
                "defined.  <b>The LULC codes must match the codes in "
                "the Sensitivity table</b>."),
            label='Current Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.current_landcover)
        self.future_landcover = inputs.File(
            args_key='lulc_fut_path',
            helptext=(
                "Optional.  A GDAL-supported raster file.  Inputting "
                "a future LULC will generate degradation, habitat "
                "quality, and habitat rarity (If baseline is input) "
                "outputs.  The future LULC must have it's own threat "
                "rasters, where each threat raster file path has a "
                "suffix of <b>_f</b>.<br/><br/>Each cell should "
                "represent a LULC code as an Integer.  The dataset "
                "should be in a projection where the units are in "
                "meters and the projection used should be defined. "
                "<b>The LULC codes must match the codes in the "
                "Sensitivity table</b>."),
            label='Future Land Cover (Raster) (Optional)',
            validator=self.validator)
        self.add_input(self.future_landcover)
        self.baseline_landcover = inputs.File(
            args_key='lulc_bas_path',
            helptext=(
                "Optional.  A GDAL-supported raster file.  If the "
                "baseline LULC is provided, rarity outputs will be "
                "created for the current and future LULC. The baseline "
                "LULC can have it's own threat rasters (optional), "
                "where each threat raster file path has a suffix of "
                "<b>_b</b>. If no threat rasters are found, "
                "degradation and habitat quality outputs will not be "
                "generated for the baseline LULC.<br/><br/> Each cell "
                "should  represent a LULC code as an Integer.  The "
                "dataset should be in a projection where the units are "
                "in meters and the projection used should be defined. "
                "The LULC codes must match the codes in the "
                "Sensitivity table.  If possible the baseline map "
                "should refer to a time when intensive management of "
                "the landscape was relatively rare."),
            label='Baseline Land Cover (Raster) (Optional)',
            validator=self.validator)
        self.add_input(self.baseline_landcover)
        self.threat_rasters = inputs.Folder(
            args_key='threat_raster_folder',
            helptext=(
                "The selected folder is used as the location to find "
                "all threat rasters for the threats listed in the "
                "below table."),
            label='Folder Containing Threat Rasters',
            validator=self.validator)
        self.add_input(self.threat_rasters)
        self.threats_data = inputs.File(
            args_key='threats_table_path',
            helptext=(
                "A CSV file of all the threats for the model to "
                "consider.  Each row in the table is a degradation "
                "source and each column contains a different attribute "
                "of each degradation source (THREAT, MAX_DIST, "
                "WEIGHT).<br/><br/><b>THREAT:</b> The name of the "
                "threat source and this name must match exactly to the "
                "name of the threat raster and to the name of it's "
                "corresponding column in the sensitivity table. "
                "<b>NOTE:</b> The threat raster path should have a "
                "suffix indicator ( _c, _f, _b ) and the sensitivity "
                "column should have a prefix indicator (L_). The "
                "THREAT name in the threat table should not include "
                "either the suffix or prefix. "
                "<br/><br/><b>MAX_DIST:</b> A number in kilometres "
                "(km) for the maximum distance a threat has an "
                "affect.<br/><br/><b>WEIGHT:</b> A floating point "
                "value between 0 and 1 for the the threats weight "
                "relative to the other threats.  Depending on the type "
                "of habitat under review, certain threats may cause "
                "greater degradation than other "
                "threats.<br/><br/><b>DECAY:</b> A string value of "
                "either <b>exponential</b> or <b>linear</b> "
                "representing the type of decay over space for the "
                "threat.<br/><br/>See the user's guide for valid "
                "values for these columns."),
            label='Threats Data',
            validator=self.validator)
        self.add_input(self.threats_data)
        self.accessibility_threats = inputs.File(
            args_key='access_vector_path',
            helptext=(
                "An OGR-supported vector file.  The input contains "
                "data on the relative protection that legal / "
                "institutional / social / physical barriers provide "
                "against threats.  The vector file should contain "
                "polygons with a field <b>ACCESS</b>. The "
                "<b>ACCESS</b> values should range from 0 - 1, where 1 "
                "is fully accessible.  Any cells not covered by a "
                "polygon will be set to 1."),
            label='Accessibility to Threats (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.accessibility_threats)
        self.sensitivity_data = inputs.File(
            args_key='sensitivity_table_path',
            helptext=(
                "A CSV file of LULC types, whether or not the are "
                "considered habitat, and, for LULC types that are "
                "habitat, their specific sensitivity to each threat. "
                "Each row is a LULC type with the following columns: "
                "<b>LULC, HABITAT, L_THREAT1, L_THREAT2, "
                "...</b><br/><br/><b>LULC:</b> Integer values that "
                "reflect each LULC code found in current, future, and "
                "baseline rasters.<br/><br/><b>HABITAT:</b> A value of "
                "0 or 1 (presence / absence) or a value between 0 and "
                "1 (continuum) depicting the suitability of "
                "habitat.<br/><br/><b>L_THREATN:</b> Each L_THREATN "
                "should match exactly with the threat names given in "
                "the threat CSV file, where the THREATN is the name "
                "that matches.  This is an floating point value "
                "between 0 and 1 that represents the sensitivity of a "
                "habitat to a threat.  <br/><br/>.Please see the users "
                "guide for more detailed information on proper column "
                "values and column names for each threat."),
            label='Sensitivity of Land Cover Types to Each Threat, File (CSV)',
            validator=self.validator)
        self.add_input(self.sensitivity_data)
        self.half_saturation_constant = inputs.Text(
            args_key='half_saturation_constant',
            helptext=(
                "A positive floating point value that is defaulted at "
                "0.5. This is the value of the parameter k in equation "
                "(4). In general, set k to half of the highest grid "
                "cell degradation value on the landscape.  To perform "
                "this model calibration the model must be run once in "
                "order to find the highest degradation value and set k "
                "for the provided landscape.  Note that the choice of "
                "k only determines the spread and central tendency of "
                "habitat quality cores and does not affect the rank."),
            label='Half-Saturation Constant',
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
