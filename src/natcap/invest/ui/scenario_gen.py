# coding=UTF-8
import logging

from natcap.invest.ui import model, inputs
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import scenario_gen_proximity


LOGGER = logging.getLogger(__name__)


class ScenarioGenProximity(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['scenario_generator_proximity'].model_title,
            target=scenario_gen_proximity.execute,
            validator=scenario_gen_proximity.validate,
            localdoc=MODEL_METADATA['scenario_generator_proximity'].userguide)

        self.base_lulc_path = inputs.File(
            args_key='base_lulc_path',
            label='Base Land Use/Cover (Raster)',
            validator=self.validator)
        self.add_input(self.base_lulc_path)
        self.aoi_path = inputs.File(
            args_key='aoi_path',
            helptext=(
                "This is a set of polygons that will be used to "
                "aggregate carbon values at the end of the run if "
                "provided."),
            label='Area of interest (Vector) (optional)',
            validator=self.validator)
        self.add_input(self.aoi_path)
        self.area_to_convert = inputs.Text(
            args_key='area_to_convert',
            label='Max area to convert (Ha)',
            validator=self.validator)
        self.add_input(self.area_to_convert)
        self.focal_landcover_codes = inputs.Text(
            args_key='focal_landcover_codes',
            label='Focal Landcover Codes (list)',
            validator=self.validator)
        self.add_input(self.focal_landcover_codes)
        self.convertible_landcover_codes = inputs.Text(
            args_key='convertible_landcover_codes',
            label='Convertible Landcover Codes (list)',
            validator=self.validator)
        self.add_input(self.convertible_landcover_codes)
        self.replacement_lucode = inputs.Text(
            args_key='replacement_lucode',
            label='Replacement Landcover Code (int)',
            validator=self.validator)
        self.add_input(self.replacement_lucode)
        self.convert_farthest_from_edge = inputs.Checkbox(
            args_key='convert_farthest_from_edge',
            helptext=(
                "This scenario converts the convertible landcover "
                "codes starting at the furthest pixel from the closest "
                "base landcover codes and moves inward."),
            label='Farthest from edge')
        self.add_input(self.convert_farthest_from_edge)
        self.convert_nearest_to_edge = inputs.Checkbox(
            args_key='convert_nearest_to_edge',
            helptext=(
                "This scenario converts the convertible landcover "
                "codes starting at the closest pixel in the base "
                "landcover codes and moves outward."),
            label='Nearest to edge')
        self.add_input(self.convert_nearest_to_edge)
        self.n_fragmentation_steps = inputs.Text(
            args_key='n_fragmentation_steps',
            helptext=(
                "This parameter is used to divide the conversion "
                "simulation into equal subareas of the requested max "
                "area.  During each sub-step the distance transform is "
                "recalculated from the base landcover codes.  This can "
                "affect the final result if the base types are also "
                "convertible types."),
            label='Number of Steps in Conversion',
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
            self.replacement_lucode.args_key: self.replacement_lucode.value(),
            self.convert_farthest_from_edge.args_key:
                self.convert_farthest_from_edge.value(),
            self.convert_nearest_to_edge.args_key:
                self.convert_nearest_to_edge.value(),
            self.n_fragmentation_steps.args_key:
                self.n_fragmentation_steps.value(),
        }

        return args
