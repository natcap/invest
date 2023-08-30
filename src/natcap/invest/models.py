import importlib
import pkgutil

import natcap.invest


def is_invest_model(module):
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
for _, name, ispkg in pkgutil.iter_modules(natcap.invest.__path__, 'natcap.invest.'):
    module = importlib.import_module(name)
    if ispkg:
        for _, sub_name, _ in pkgutil.iter_modules(module.__path__):
            submodule = importlib.import_module(f'{name}.{sub_name}')
            if is_invest_model(submodule):
                pyname_to_module[f'{name}.{sub_name}'] = submodule
    else:
        if is_invest_model(module):
            pyname_to_module[name] = module

model_id_to_pyname = {}
model_id_to_spec = {}
model_alias_to_id = {}
for pyname, model in pyname_to_module.items():
    model_id_to_pyname[model.MODEL_SPEC['model_id']] = pyname
    model_id_to_spec[model.MODEL_SPEC['model_id']] = model.MODEL_SPEC
    for alias in model.MODEL_SPEC['aliases']:
        model_alias_to_id[alias] = model.MODEL_SPEC['model_id']
