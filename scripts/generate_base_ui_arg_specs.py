"""
This script makes boilerplate UI spec files for each invest model.
They are not meant to be checked and used directly in the new UI,
but are meant to be edited manually from here. 

Manual edits to the resulting JSON files:
* `order` values should be added
* `ui_control` and `ui_option` values should be reviewed
 and modified as needed.

It's unlikely we'll ever need to run this script again,
but it could become useful if an model is added or an ARGS_SPEC 
is heavily overhauled.
"""

import importlib
import json
import os
from natcap.invest import cli


for model_ui in cli._MODEL_UIS:
    model_module = importlib.import_module(name=cli._MODEL_UIS[model_ui].pyname)
    model_spec = model_module.ARGS_SPEC
    model_ui_spec = {}
    for arg in model_spec['args']:
        if arg == 'n_workers':
            continue
        model_ui_spec[arg] = {'order': None}
        if isinstance(model_spec['args'][arg]['required'], str):
            model_ui_spec[arg]['ui_option'] = 'disable'
            controller_list = [x for x in model_spec['args'] if x in model_spec['args'][arg]['required']]
            for controller in controller_list:
                if controller not in model_ui_spec:
                    model_ui_spec[controller] = {}
                if 'ui_control' not in model_ui_spec[controller]:
                    model_ui_spec[controller]['ui_control'] = set([arg])
                else:
                    model_ui_spec[controller]['ui_control'].add(arg)

    for key in model_ui_spec:
        try:
            model_ui_spec[key]['ui_control'] = list(model_ui_spec[key]['ui_control'])
        except KeyError:
            pass
    jsonfile_path = os.path.join('../ui_data', model_spec['module'] + '.json')
    with open(jsonfile_path, 'w') as file:
        json.dump(model_ui_spec, file, indent=4)
