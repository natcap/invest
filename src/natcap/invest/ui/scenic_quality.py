# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.scenic_quality import scenic_quality
from natcap.invest.model_metadata import MODEL_METADATA


class ScenicQuality(model.InVESTModel):
    def __init__(self):
        val_func_options = {
            val['display_name']: key for key, val in
            scenic_quality.ARGS_SPEC['args']['valuation_function']['options'].items()
        }
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['scenic_quality'].model_title,
            target=scenic_quality.execute,
            validator=scenic_quality.validate,
            localdoc=MODEL_METADATA['scenic_quality'].userguide)

        self.general_tab = inputs.Container(
            interactive=True,
            label='General')
        self.add_input(self.general_tab)
        self.aoi_path = inputs.File(
            args_key='aoi_path',
            helptext=(
                "An OGR-supported vector file.  This AOI instructs "
                "the model where to clip the input data and the extent "
                "of analysis.  Users will create a polygon feature "
                "layer that defines their area of interest.  The AOI "
                "must intersect the Digital Elevation Model (DEM)."),
            label='Area of Interest (Vector) (Required)',
            validator=self.validator)
        self.general_tab.add_input(self.aoi_path)
        self.structure_path = inputs.File(
            args_key='structure_path',
            helptext=(
                "An OGR-supported vector file.  The user must specify "
                "a point feature layer that indicates locations of "
                "objects that contribute to negative scenic quality, "
                "such as aquaculture netpens or wave energy "
                "facilities.  In order for the viewshed analysis to "
                "run correctly, the projection of this input must be "
                "consistent with the project of the DEM input."),
            label='Features Impacting Scenic Quality (Vector) (Required)',
            validator=self.validator)
        self.general_tab.add_input(self.structure_path)
        self.dem_path = inputs.File(
            args_key='dem_path',
            helptext=(
                "A GDAL-supported raster file.  An elevation raster "
                "layer is required to conduct viewshed analysis. "
                "Elevation data allows the model to determine areas "
                "within the AOI's land-seascape where point features "
                "contributing to negative scenic quality are visible."),
            label='Digital Elevation Model (Raster) (Required)',
            validator=self.validator)
        self.general_tab.add_input(self.dem_path)
        self.refraction = inputs.Text(
            args_key='refraction',
            helptext=(
                "The earth curvature correction option corrects for "
                "the curvature of the earth and refraction of visible "
                "light in air.  Changes in air density curve the light "
                "downward causing an observer to see further and the "
                "earth to appear less curved.  While the magnitude of "
                "this effect varies with atmospheric conditions, a "
                "standard rule of thumb is that refraction of visible "
                "light reduces the apparent curvature of the earth by "
                "one-seventh.  By default, this model corrects for the "
                "curvature of the earth and sets the refractivity "
                "coefficient to 0.13."),
            label='Refractivity Coefficient (Required)',
            validator=self.validator)
        self.general_tab.add_input(self.refraction)
        self.valuation_container = inputs.Container(
            args_key='do_valuation',
            expandable=True,
            expanded=False,
            interactive=True,
            label='Valuation')
        self.add_input(self.valuation_container)
        self.valuation_function = inputs.Dropdown(
            args_key='valuation_function',
            helptext=(
                "This field indicates the functional form f(x) the "
                "model will use to value the visual impact for each "
                "viewpoint."),
            label='Valuation Function',
            options=['linear: a + bx',
                     'logarithmic: a + b log(x+1)',
                     'exponential: a * e^(-bx)'],
            return_value_map=val_func_options)
        self.valuation_container.add_input(self.valuation_function)
        self.a_coefficient = inputs.Text(
            args_key='a_coef',
            helptext=(
                "First coefficient used by the valuation function"),
            label="'a' Coefficient (Required)",
            validator=self.validator)
        self.valuation_container.add_input(self.a_coefficient)
        self.b_coefficient = inputs.Text(
            args_key='b_coef',
            helptext=(
                "Second coefficient used by the valuation function"),
            label="'b' Coefficient (Required)",
            validator=self.validator)
        self.valuation_container.add_input(self.b_coefficient)
        self.max_valuation_radius = inputs.Text(
            args_key='max_valuation_radius',
            helptext=(
                "Radius beyond which the valuation is set to zero. "
                "The valuation function 'f' cannot be negative at the "
                "radius 'r' (f(r)>=0)."),
            label='Maximum Valuation Radius (meters) (Required)',
            validator=self.validator)
        self.valuation_container.add_input(self.max_valuation_radius)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi_path.args_key: self.aoi_path.value(),
            self.structure_path.args_key: self.structure_path.value(),
            self.dem_path.args_key: self.dem_path.value(),
            self.refraction.args_key: self.refraction.value(),
            self.valuation_container.args_key: self.valuation_container.value(),
            self.valuation_function.args_key: self.valuation_function.value(),
            self.a_coefficient.args_key: self.a_coefficient.value(),
            self.b_coefficient.args_key: self.b_coefficient.value(),
            self.max_valuation_radius.args_key:
                self.max_valuation_radius.value(),
        }

        return args
