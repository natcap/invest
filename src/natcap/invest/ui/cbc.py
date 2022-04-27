# coding=UTF-8

import functools

from natcap.invest.ui import model, inputs
from natcap.invest.coastal_blue_carbon import coastal_blue_carbon, preprocessor
from natcap.invest.model_metadata import MODEL_METADATA


def _create_input_kwargs_from_args_spec(
        args_key, args_spec, validator):
    """Helper function to return kwargs for most model inputs.

    Args:
        args_key: The args key of the input from which a kwargs
            dict is being built.
        args_spec: The ARGS_SPEC object to reference.
        validator: The validator callable to provide to the ``validator`` kwarg
            for the input.

    Returns:
        A dict of ``kwargs`` to explode to an ``inputs.GriddedInput``
        object at creation time.
    """
    model_spec = args_spec['args']
    return {
        'args_key': args_key,
        'helptext': model_spec[args_key]['about'],
        'label': model_spec[args_key]['name'],
        'validator': validator,
    }


class CoastalBlueCarbonPreprocessor(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['coastal_blue_carbon_preprocessor'].model_title,
            target=preprocessor.execute,
            validator=preprocessor.validate,
            localdoc=MODEL_METADATA['coastal_blue_carbon_preprocessor'].userguide)

        _ui_keys = functools.partial(
            _create_input_kwargs_from_args_spec,
            args_spec=preprocessor.ARGS_SPEC,
            validator=self.validator)

        self.lulc_snapshot_csv = inputs.File(
            **_ui_keys('landcover_snapshot_csv'))
        self.add_input(self.lulc_snapshot_csv)

        self.lulc_lookup_table_path = inputs.File(
            **_ui_keys('lulc_lookup_table_path'))
        self.add_input(self.lulc_lookup_table_path)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.lulc_lookup_table_path.args_key:
                self.lulc_lookup_table_path.value(),
            self.lulc_snapshot_csv.args_key: self.lulc_snapshot_csv.value(),
        }
        return args


class CoastalBlueCarbon(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['coastal_blue_carbon'].model_title,
            target=coastal_blue_carbon.execute,
            validator=coastal_blue_carbon.validate,
            localdoc=MODEL_METADATA['coastal_blue_carbon'].userguide)

        _ui_keys = functools.partial(
            _create_input_kwargs_from_args_spec,
            args_spec=coastal_blue_carbon.ARGS_SPEC,
            validator=self.validator)

        self.snapshots_table = inputs.File(
            **_ui_keys('landcover_snapshot_csv'))
        self.add_input(self.snapshots_table)

        self.biophysical_table_path = inputs.File(
            **_ui_keys('biophysical_table_path'))
        self.add_input(self.biophysical_table_path)

        self.landcover_transitions_table = inputs.File(
            **_ui_keys('landcover_transitions_table'))
        self.add_input(self.landcover_transitions_table)

        self.analysis_year = inputs.Text(**_ui_keys('analysis_year'))
        self.add_input(self.analysis_year)

        self.do_economic_analysis = inputs.Container(
            args_key='do_economic_analysis',
            expandable=True,
            expanded=True,
            label='Calculate Net Present Value of Sequestered Carbon')
        self.add_input(self.do_economic_analysis)

        self.use_price_table = inputs.Checkbox(
            args_key='use_price_table',
            helptext='',
            label='Use Price Table')
        self.do_economic_analysis.add_input(self.use_price_table)
        self.price = inputs.Text(**_ui_keys('price'))
        self.do_economic_analysis.add_input(self.price)

        self.inflation_rate = inputs.Text(**_ui_keys('inflation_rate'))
        self.do_economic_analysis.add_input(self.inflation_rate)

        self.price_table_path = inputs.File(**_ui_keys('price_table_path'))
        self.do_economic_analysis.add_input(self.price_table_path)

        self.discount_rate = inputs.Text(**_ui_keys('discount_rate'))
        self.do_economic_analysis.add_input(self.discount_rate)

        # Set interactivity, requirement as input sufficiency changes
        self.use_price_table.sufficiency_changed.connect(
            self._price_table_sufficiency_changed)

    def _price_table_sufficiency_changed(self, new_sufficiency):
        self.price.set_interactive(not new_sufficiency)
        self.inflation_rate.set_interactive(not new_sufficiency)
        self.price_table_path.set_interactive(new_sufficiency)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.snapshots_table.args_key:
                self.snapshots_table.value(),
            self.landcover_transitions_table.args_key:
                self.landcover_transitions_table.value(),
            self.analysis_year.args_key: self.analysis_year.value(),
            self.biophysical_table_path.args_key:
                self.biophysical_table_path.value(),
            self.landcover_transitions_table.args_key:
                self.landcover_transitions_table.value(),
            self.do_economic_analysis.args_key:
                self.do_economic_analysis.value(),
            }

        if self.do_economic_analysis.value():
            args[self.price.args_key] = self.price.value()
            args[self.inflation_rate.args_key] = self.inflation_rate.value()
            args[self.discount_rate.args_key] = self.discount_rate.value()

            args[self.use_price_table.args_key] = self.use_price_table.value()
            if self.use_price_table.value():
                args[self.price_table_path.args_key] = (
                    self.price_table_path.value())

        return args
