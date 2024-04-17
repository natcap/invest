import importlib
import pkgutil

import natcap.invest

def is_invest_compliant_model(module):
    """Check if a python module is an invest model.

    Args:
        module (module): python module to check

    Returns:
        True if the module has a ``MODEL_SPEC`` dictionary attribute and
        ``execute`` and ``validate`` functions, False otherwise
    """
    return (
        hasattr(module, "execute") and callable(module.execute) and
        hasattr(module, "validate") and callable(module.validate) and
        # could also validate model spec structure
        hasattr(module, "MODEL_SPEC") and isinstance(module.MODEL_SPEC, dict))

# pyname: importable name e.g. natcap.invest.carbon, natcap.invest.sdr.sdr
# model id: identifier e.g. coastal_blue_carbon
# model name: title e.g. Coastal Blue Carbon
# Build up an index mapping aliases to model_name.
pyname_to_module = {}
for _, _name, _ispkg in pkgutil.iter_modules(natcap.invest.__path__):
    if _name in {'__main__', 'cli', 'ui_server'}:
        continue  # avoid a circular import
    _module = importlib.import_module(f'natcap.invest.{_name}')
    if _ispkg:
        for _, _sub_name, _ in pkgutil.iter_modules(_module.__path__):
            _submodule = importlib.import_module(f'natcap.invest.{_name}.{_sub_name}')
            if is_invest_compliant_model(_submodule):
                pyname_to_module[f'natcap.invest.{_name}.{_sub_name}'] = _submodule
    else:
        if is_invest_compliant_model(_module):
            pyname_to_module[f'natcap.invest.{_name}'] = _module

model_id_to_pyname = {}
model_id_to_spec = {}
model_alias_to_id = {}
for _pyname, _model in pyname_to_module.items():
    model_id_to_pyname[_model.MODEL_SPEC['model_id']] = _pyname
    model_id_to_spec[_model.MODEL_SPEC['model_id']] = _model.MODEL_SPEC
    for _alias in _model.MODEL_SPEC['aliases']:
        model_alias_to_id[_alias] = _model.MODEL_SPEC['model_id']
