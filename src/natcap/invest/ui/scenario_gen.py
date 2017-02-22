# coding=UTF-8

from natcap.invest.ui import model
from natcap.ui import inputs
import natcap.invest.scenario_gen_proximity


class ScenarioGenProximity(model.Model):
    label = u'Scenario Generator: Proximity Based'
    target = staticmethod(natcap.invest.scenario_gen_proximity.execute)
    validator = staticmethod(natcap.invest.scenario_gen_proximity.validate)
    localdoc = u'../documentation/scenario_gen_proximity.html'

    def __init__(self):
        model.Model.__init__(self)

        self.base_lulc_path = inputs.File(
            args_key=u'base_lulc_path',
            label=u'Base Land Use/Cover (Raster)',
            required=True,
            validator=self.validator)
        self.add_input(self.base_lulc_path)
        self.aoi_path = inputs.File(
            args_key=u'aoi_path',
            helptext=(
                u"This is a set of polygons that will be used to "
                u"aggregate carbon values at the end of the run if "
                u"provided."),
            label=u'Area of interest (Vector) (optional)',
            required=False,
            validator=self.validator)
        self.add_input(self.aoi_path)
        self.area_to_convert = inputs.Text(
            args_key=u'area_to_convert',
            label=u'Max area to convert (Ha)',
            required=True,
            validator=self.validator)
        self.add_input(self.area_to_convert)
        self.focal_landcover_codes = inputs.Text(
            args_key=u'focal_landcover_codes',
            label=u'Focal Landcover Codes (list)',
            required=True,
            validator=self.validator)
        self.add_input(self.focal_landcover_codes)
        self.convertible_landcover_codes = inputs.Text(
            args_key=u'convertible_landcover_codes',
            label=u'Convertible Landcover Codes (list)',
            required=True,
            validator=self.validator)
        self.add_input(self.convertible_landcover_codes)
        self.replacment_lucode = inputs.Text(
            args_key=u'replacment_lucode',
            label=u'Replacement Landcover Code (int)',
            required=True,
            validator=self.validator)
        self.add_input(self.replacment_lucode)
        self.convert_farthest_from_edge = inputs.Checkbox(
            args_key=u'convert_farthest_from_edge',
            helptext=(
                u"This scenario converts the convertible landcover "
                u"codes starting at the furthest pixel from the closest "
                u"base landcover codes and moves inward."),
            label=u'Farthest from edge')
        self.add_input(self.convert_farthest_from_edge)
        self.convert_nearest_to_edge = inputs.Checkbox(
            args_key=u'convert_nearest_to_edge',
            helptext=(
                u"This scenario converts the convertible landcover "
                u"codes starting at the closest pixel in the base "
                u"landcover codes and moves outward."),
            label=u'Nearest to edge')
        self.add_input(self.convert_nearest_to_edge)
        self.n_fragmentation_steps = inputs.Text(
            args_key=u'n_fragmentation_steps',
            helptext=(
                u"This parameter is used to divide the conversion "
                u"simulation into equal subareas of the requested max "
                u"area.  During each sub-step the distance transform is "
                u"recalculated from the base landcover codes.  This can "
                u"affect the final result if the base types are also "
                u"convertible types."),
            label=u'Number of Steps in Conversion',
            required=True,
            validator=self.validator)
        self.add_input(self.n_fragmentation_steps)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.base_lulc_path.args_key: self.base_lulc_path.value(),
            self.aoi_path.args_key: self.aoi_path.value(),
            self.area_to_convert.args_key: self.area_to_convert.value(),
            self.focal_landcover_codes.args_key:
                self.focal_landcover_codes.value(),
            self.convertible_landcover_codes.args_key:
                self.convertible_landcover_codes.value(),
            self.replacment_lucode.args_key: self.replacment_lucode.value(),
            self.convert_farthest_from_edge.args_key:
                self.convert_farthest_from_edge.value(),
            self.convert_nearest_to_edge.args_key:
                self.convert_nearest_to_edge.value(),
            self.n_fragmentation_steps.args_key:
                self.n_fragmentation_steps.value(),
        }

        return args
