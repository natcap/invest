import sys
import importlib
import re
import pprint
import json
import collections


from natcap.invest import cli
from natcap.invest import validation
from natcap.invest.ui import model, inputs

from docstring_parser import parse


def main(modelname):
    # objective: build up the args spec dict from various places around InVEST.

    model_components = cli._MODEL_UIS[modelname]

    model_module = importlib.import_module(model_components.pyname)
    docstring = parse(getattr(model_module, 'execute').__doc__)

    gui_class = model_components.gui
    module_name, classname = gui_class.split('.')
    ui_module = importlib.import_module(
        name='.ui.%s' % module_name,
        package='natcap.invest')
    model_form = getattr(ui_module, classname)()

    # add some general model info from the UI class.
    model_details = collections.OrderedDict()
    model_details['model_name'] = model_form.label
    model_details['module'] = model_components.pyname
    model_details['userguide_html'] = model_form.localdoc

    model_details['args_with_spatial_overlap'] = {
        'spatial_keys': [],
        'reference_key': ''
    }

    args_spec = collections.OrderedDict({
        "workspace_dir": "validation.WORKSPACE_SPEC",
        "results_suffix": "validation.SUFFIX_SPEC",
        "n_workers": "validation.N_WORKERS_SPEC",
    })
    model_details['args'] = args_spec

    # build a lookup of {args_key: model UI}
    ui_windows = {}
    for ui_object in model_form.inputs:
        ui_windows[ui_object.args_key] = {
            "helptext": ui_object.helptext,
            "label": ui_object.label,
            "required": (True if "required" in ui_object.label else False),
            'object': ui_object,
        }

    for parameter in docstring.params:
        key_name = parameter.arg_name
        matches = re.search("args\[['\"]([0-9a-zA-Z_-]+)['\"]\]", key_name)
        if matches:
            # matches.group(0) would be args['workspace_dir']
            # matches.group(1) would be workspace_dir.
            key_name = matches.group(1)

        if key_name in args_spec:
            continue

        args_spec[key_name] = {}
        args_spec[key_name]['validation_options'] = {}

        # attempt to determine the type of the input.
        if 'raster' in parameter.description.lower():
            input_type = 'raster'
        elif 'vector' in parameter.description.lower():
            input_type = 'vector'
        elif 'csv' in parameter.description.lower():
            input_type = 'csv'
        elif parameter.type_name is None:
            input_type = 'UNKNOWN'
        elif parameter.type_name.lower() in ('int', 'float', 'number'):
            input_type = 'number'
        elif parameter.type_name.lower() in ('bool', 'boolean'):
            input_type = 'boolean'
        elif key_name not in ui_windows:
            input_type = 'UNKNOWN'
        elif hasattr(ui_windows[key_name]['object'], 'options'):
            input_type = 'option_string'
            args_spec[key_name]['validation_options']['options'] = (
                ui_windows[key_name]['object'].user_options)
        else:
            input_type = 'UNKNOWN'
        args_spec[key_name]['type'] = input_type

        # Attempt to determine if an input is required
        required = False
        if '(required)' in parameter.description.lower():
            required = True
        elif key_name not in ui_windows:
            requried = 'UNKNOWN'
        elif ui_windows[key_name]['required']:
            required = True
        args_spec[key_name]['required'] = required

        # Grab the helptext and other attributes from the UI
        # If the UI doesn't have any about text, use the args key description.
        if key_name in ui_windows:
            args_spec[key_name]['about'] = ui_windows[key_name]['helptext']
            if args_spec[key_name]['about'] is None:
                args_spec[key_name]['about'] = parameter.description
            args_spec[key_name]['name'] = (
                ui_windows[key_name]['label'].replace('(Required)', '').strip())

    print(json.dumps(model_details, indent=4))


if __name__ == '__main__':
    main(sys.argv[1])
