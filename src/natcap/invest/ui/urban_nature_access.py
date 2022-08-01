# coding=UTF-8
import functools

from natcap.invest import urban_nature_access
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest.ui import cbc
from natcap.invest.ui import inputs
from natcap.invest.ui import model


class UrbanNatureAccess(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['urban_nature_access'].model_title,
            target=urban_nature_access.execute,
            validator=urban_nature_access.validate,
            localdoc=MODEL_METADATA['urban_nature_access'].userguide)

        _ui_keys = functools.partial(
            cbc._create_input_kwargs_from_args_spec,
            args_spec=urban_nature_access.ARGS_SPEC,
            validator=self.validator)

        self.lulc = inputs.File(**_ui_keys('lulc_raster_path'))
        self.add_input(self.lulc)

        self.lulc_attribute_table = inputs.File(
            **_ui_keys('lulc_attribute_table'))
        self.add_input(self.lulc_attribute_table)

        self.population = inputs.File(
            **_ui_keys('population_raster_path'))
        self.add_input(self.population)

        self.admin_units = inputs.File(
            **_ui_keys('admin_unit_vector_path'))
        self.add_input(self.admin_units)

        self.greenspace_demand = inputs.Text(
            **_ui_keys('greenspace_demand'))
        self.add_input(self.greenspace_demand)

        decay_function_spec = urban_nature_access.ARGS_SPEC[
            'args']['decay_function']
        self.decay_function = inputs.Dropdown(
            label=decay_function_spec['name'],
            helptext=decay_function_spec['about'],
            options=list(decay_function_spec['options'].keys()),
            args_key='decay_function'
        )
        self.add_input(self.decay_function)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.lulc.args_key: self.lulc.value(),
            self.lulc_attribute_table.args_key:
                self.lulc_attribute_table.value(),
            self.population.args_key: self.population.value(),
            self.admin_units.args_key: self.admin_units.value(),
            self.greenspace_demand.args_key: self.greenspace_demand.value(),
            self.decay_function.args_key: self.decay_function.value(),
        }
        return args
