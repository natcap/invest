# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.scenic_quality import scenic_quality


class ScenicQuality(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Scenic Quality',
            target=scenic_quality.execute,
            validator=scenic_quality.validate,
            localdoc=u'../documentation/scenic_quality.html')

        self.beta_only = inputs.Label(
            text=(
                u"This tool is considered UNSTABLE.  Users may "
                u"experience performance issues and unexpected errors."))
        self.general_tab = inputs.Container(
            interactive=True,
            label=u'General')
        self.add_input(self.general_tab)
        self.aoi_uri = inputs.File(
            args_key=u'aoi_uri',
            helptext=(
                u"An OGR-supported vector file.  This AOI instructs "
                u"the model where to clip the input data and the extent "
                u"of analysis.  Users will create a polygon feature "
                u"layer that defines their area of interest.  The AOI "
                u"must intersect the Digital Elevation Model (DEM)."),
            label=u'Area of Interest (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.aoi_uri)
        self.cell_size = inputs.Text(
            args_key=u'cell_size',
            helptext=u'Length (in meters) of each side of the (square) cell.',
            label=u'Cell Size (meters)',
            validator=self.validator)
        self.general_tab.add_input(self.cell_size)
        self.structure_uri = inputs.File(
            args_key=u'structure_uri',
            helptext=(
                u"An OGR-supported vector file.  The user must specify "
                u"a point feature layer that indicates locations of "
                u"objects that contribute to negative scenic quality, "
                u"such as aquaculture netpens or wave energy "
                u"facilities.  In order for the viewshed analysis to "
                u"run correctly, the projection of this input must be "
                u"consistent with the project of the DEM input."),
            label=u'Features Impacting Scenic Quality (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.structure_uri)
        self.dem_uri = inputs.File(
            args_key=u'dem_uri',
            helptext=(
                u"A GDAL-supported raster file.  An elevation raster "
                u"layer is required to conduct viewshed analysis. "
                u"Elevation data allows the model to determine areas "
                u"within the AOI's land-seascape where point features "
                u"contributing to negative scenic quality are visible."),
            label=u'Digital Elevation Model (Raster)',
            validator=self.validator)
        self.general_tab.add_input(self.dem_uri)
        self.refraction = inputs.Text(
            args_key=u'refraction',
            helptext=(
                u"The earth curvature correction option corrects for "
                u"the curvature of the earth and refraction of visible "
                u"light in air.  Changes in air density curve the light "
                u"downward causing an observer to see further and the "
                u"earth to appear less curved.  While the magnitude of "
                u"this effect varies with atmospheric conditions, a "
                u"standard rule of thumb is that refraction of visible "
                u"light reduces the apparent curvature of the earth by "
                u"one-seventh.  By default, this model corrects for the "
                u"curvature of the earth and sets the refractivity "
                u"coefficient to 0.13."),
            label=u'Refractivity Coefficient',
            validator=self.validator)
        self.general_tab.add_input(self.refraction)
        self.pop_uri = inputs.File(
            args_key=u'pop_uri',
            helptext=(
                u"A GDAL-supported raster file.  A population raster "
                u"layer is required to determine population within the "
                u"AOI's land-seascape where point features contributing "
                u"to negative scenic quality are visible and not "
                u"visible."),
            label=u'Population (Raster)',
            validator=self.validator)
        self.general_tab.add_input(self.pop_uri)
        self.overlap_uri = inputs.File(
            args_key=u'overlap_uri',
            helptext=(
                u"An OGR-supported vector file.  The user has the "
                u"option of providing a polygon feature layer where "
                u"they would like to determine the impact of objects on "
                u"visual quality.  This input must be a polygon and "
                u"projected in meters.  The model will use this layer "
                u"to determine what percent of the total area of each "
                u"polygon feature can see at least one of the point "
                u"features impacting scenic quality."),
            label=u'Overlap Analysis Features (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.overlap_uri)
        self.valuation_tab = inputs.Container(
            interactive=True,
            label=u'Valuation')
        self.add_input(self.valuation_tab)
        self.valuation_function = inputs.Dropdown(
            args_key=u'valuation_function',
            helptext=(
                u"This field indicates the functional form f(x) the "
                u"model will use to value the visual impact for each "
                u"viewpoint.  For distances less than 1 km (x<1), the "
                u"model uses a linear form g(x) where the line passes "
                u"through f(1) (i.e.  g(1) == f(1)) and extends to zero "
                u"with the same slope as f(1) (i.e.  g'(x) == f'(1))."),
            label=u'Valuation Function',
            options=[u'polynomial: a + bx + cx^2 + dx^3',
                     u'logarithmic: a + b ln(x)'])
        self.valuation_tab.add_input(self.valuation_function)
        self.a_coefficient = inputs.Text(
            args_key=u'a_coefficient',
            helptext=(
                u"First coefficient used either by the polynomial or "
                u"by the logarithmic valuation function."),
            label=u"'a' Coefficient (polynomial/logarithmic)",
            validator=self.validator)
        self.valuation_tab.add_input(self.a_coefficient)
        self.b_coefficient = inputs.Text(
            args_key=u'b_coefficient',
            helptext=(
                u"Second coefficient used either by the polynomial or "
                u"by the logarithmic valuation function."),
            label=u"'b' Coefficient (polynomial/logarithmic)",
            validator=self.validator)
        self.valuation_tab.add_input(self.b_coefficient)
        self.c_coefficient = inputs.Text(
            args_key=u'c_coefficient',
            helptext=u"Third coefficient for the polynomial's quadratic term.",
            label=u"'c' Coefficient (polynomial only)",
            validator=self.validator)
        self.valuation_tab.add_input(self.c_coefficient)
        self.d_coefficient = inputs.Text(
            args_key=u'd_coefficient',
            helptext=u"Fourth coefficient for the polynomial's cubic exponent.",
            label=u"'d' Coefficient (polynomial only)",
            validator=self.validator)
        self.valuation_tab.add_input(self.d_coefficient)
        self.max_valuation_radius = inputs.Text(
            args_key=u'max_valuation_radius',
            helptext=(
                u"Radius beyond which the valuation is set to zero. "
                u"The valuation function 'f' cannot be negative at the "
                u"radius 'r' (f(r)>=0)."),
            label=u'Maximum Valuation Radius (meters)',
            validator=self.validator)
        self.valuation_tab.add_input(self.max_valuation_radius)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.aoi_uri.args_key: self.aoi_uri.value(),
            self.structure_uri.args_key: self.structure_uri.value(),
            self.dem_uri.args_key: self.dem_uri.value(),
            self.refraction.args_key: self.refraction.value(),
            self.valuation_function.args_key: self.valuation_function.value(),
            self.a_coefficient.args_key: self.a_coefficient.value(),
            self.b_coefficient.args_key: self.b_coefficient.value(),
            self.c_coefficient.args_key: self.c_coefficient.value(),
            self.d_coefficient.args_key: self.d_coefficient.value(),
            self.max_valuation_radius.args_key:
                self.max_valuation_radius.value(),
        }
        if self.cell_size.value():
            args[self.cell_size.args_key] = self.cell_size.value()
        if self.pop_uri.value():
            args[self.pop_uri.args_key] = self.pop_uri.value()
        if self.overlap_uri.value():
            args[self.overlap_uri.args_key] = self.overlap_uri.value()

        return args
