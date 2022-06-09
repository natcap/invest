# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import habitat_quality


class HabitatQuality(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['habitat_quality'].model_title,
            target=habitat_quality.execute,
            validator=habitat_quality.validate,
            localdoc=MODEL_METADATA['habitat_quality'].userguide)
        self.current_landcover = inputs.File(
            args_key='lulc_cur_path',
            helptext=(
                "A GDAL-supported raster file.  The current LULC must "
                "have its' own threat rasters, where each threat "
                "raster file path is defined in the <b>Threats Data</b> "
                "CSV..<br/><br/> "
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
                "rasters, where each threat raster file path is defined "
                "in the <b>Threats Data</b> CSV.<br/><br/>Each cell should "
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
                "where each threat raster file path is defined in the "
                "<b>Threats Data</b> CSV. If there are no threat rasters and "
                "the threat paths are left blank in the CSV column, "
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
        self.threats_data = inputs.File(
            args_key='threats_table_path',
            helptext=(
                "A CSV file of all the threats for the model to consider. "
                "Each row in the table is a degradation source. The columns "
                "(THREAT, MAX_DIST, WEIGHT, DECAY) are different attributes "
                "of each degradation source. The columns (BASE_PATH, CUR_PATH, "
                "FUT_PATH) specify the filepath name for the degradation "
                "source where the path is relative to the THREAT CSV. "
                "Column names are case-insensitive. "
                "<br/><br/><b>THREAT:</b> The name of the "
                "threat source must match exactly to the name "
                "of it's corresponding column in the sensitivity table. "
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
                "threat. <br/><br/><b>BASE_PATH:</b> Required if baseline "
                "LULC input. The THREAT raster filepath for the base scenario "
                "where the filepath is relative to the THREAT CSV input. "
                "Entries can be left empty if there is no baseline scenario "
                "or if using the baseline LULC for rarity calculations only. "
                "<br/><br/><b>CUR_PATH:</b> "
                "The THREAT raster filepath for the current scenario "
                "where the filepath is relative to the THREAT CSV input. "
                "<br/><br/><b>FUT_PATH:</b> Required if future LULC input. "
                "The THREAT raster filepath for the future scenario where the "
                "filepath is relative to the THREAT CSV input."
                "<br/><br/>See the user's guide for valid values for these "
                "columns."),
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
                "<b>LULC, HABITAT, THREAT1, THREAT2, "
                "...</b><br/><br/>. Column names are case-insensitive. "
                "<b>LULC:</b> Integer values that "
                "reflect each LULC code found in current, future, and "
                "baseline rasters.<br/><br/><b>HABITAT:</b> A value of "
                "0 or 1 (presence / absence) or a value between 0 and "
                "1 (continuum) depicting the suitability of "
                "habitat.<br/><br/><b>THREATN:</b> Each THREATN "
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
                "0.05. This is the value of the parameter k in equation "
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
