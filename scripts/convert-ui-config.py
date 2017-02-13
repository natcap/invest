"""Convert an IUI json file to the natcap.invest.ui model structure."""

import json
import sys
import warnings

import autopep8


UI_CLASS_TEMPLATE = """
from natcap.ui import inputs
import {target}

class {classname}(inputs.Form):
    label = {label}
    target = staticmethod({target})
    validator = staticmethod({validator})
    localdoc = {localdoc}

    def __init__(self):
        inputs.Form.__init__(self)

{input_attributes}

        # Set interactivity, requirement as input sufficiency changes
{sufficiency_connections}

    def assemble_args(self):
        args = {{
{args_key_map}
        }}
{args_to_maybe_skip}

        return args
"""

INPUT_ATTRIBUTES_TEMPLATE = "       self.{name} = {classname}({kwargs})\n"


def format_kwargs(kwargs):
    def _convert(param):
        if isinstance(param, basestring):
            return "\"%s\"" % param
        return param

    return ', '.join(sorted("%s=%s" % (key, _convert(value))
                            for (key, value) in kwargs.iteritems()))


def convert_ui_structure(json_file, out_python_file):
    print 'Converting', json_file, out_python_file
    json_dict = json.load(open(json_file))

    # Create and open the output file.
    # extract top-level information about the model
    # Recurse through root's elements list
    #   * extract relevant information about the element
    #   * record element info to a list for later or write to a file.
    # Close the file and flush it.
    # Run autopep8 on the finished file.

    input_attributes = []  # list of strings to be written
    sufficiency_links = []  # EnabledBy/DisabledBy links
    args_values = [  # these inputs are already provided by the Form.
        '            self.workspace.args_key: self.workspace.value()',
        '            self.results_suffix.args_key: self.results_suffix.value()',
    ]
    args_to_maybe_skip = []

    def recurse(obj, container_key=None):
        if isinstance(obj, dict):
            if obj['type'].lower() == 'tab':
                obj['type'] = 'container'

            if obj['type'].lower() == 'container':
                kwargs = {}
                kwargs['label'] = obj['label']

                for target_key, source_key in (
                        ('interactive', 'enabled'),
                        ('collapsible', 'collapsible'),
                        ('args_key', 'args_id')):
                    try:
                        kwargs[target_key] = obj[source_key]
                    except KeyError:
                        pass

                input_attributes.append(
                    INPUT_ATTRIBUTES_TEMPLATE.format(
                        name=obj['id'],
                        classname='inputs.Container',
                        kwargs=format_kwargs(kwargs)))
                input_attributes.append(
                    "        self.add_input(self.{name})\n".format(
                        name=obj['id']))
                if 'args_id' in obj:
                    args_values.append(
                        '            self.{name}.args_key: self.{name}.isChecked(),'.format(
                            name=obj['id']))
                recurse(obj['elements'], container_key=obj['id'])
            elif obj['type'].lower() in ('list', 'tabbedgroup'):
                recurse(obj['elements'], container_key)
            elif obj['type'].lower() in ('sliderspinbox', 'embeddedui', 'scrollgroup'):
                warnings.warn('Type %s has been deprecated' % obj['type'], DeprecationWarning)
            else:  # object is a primitive.
                try:
                    if obj['args_id'] in ('workspace_dir', 'results_suffix'):
                        return
                except KeyError:
                    pass

                kwargs = {}
                if obj['type'].lower() == 'hideablefileentry':
                    kwargs['hideable'] = True
                    kwargs['hidden'] = True
                    obj['type'] = 'file'

                classname = obj['type'].capitalize()
                if classname == 'Multi':
                    # additional kwargs needed for Multi elements:
                    kwargs['callable'] = (
                        'functools.partial(inputs.{classname}, '
                        'label="{label}")').format(
                            classname=obj['sampleElement']['type'].capitalize(),
                            label=obj['sampleElement']['label'])
                    kwargs['link_text'] = repr(obj['linkText'])


                kwargs['label'] = repr(obj['label'])

                for target_key, source_key in (
                        ('interactive', 'enabled'),
                        ('helptext', 'helpText'),
                        ('required', 'required'),
                        ('args_key', 'args_id')):
                    try:
                        kwargs[target_key] = repr(obj[source_key])
                    except KeyError:
                        pass

                # Handle EnabledBy/DisabledBy in terms of sufficiency
                for (key, slot_name) in [('enabledBy', 'set_enabled'),
                                         ('disabledBy', 'set_disabled')]:
                    try:
                        enabling_id = obj[key]
                        sufficiency_links.append((
                            '        self.{enabling_id}.sufficiency_changed.connect(\n'
                            '            self.{id}.{funcname})').format(
                                id=obj['id'],
                                enabling_id=enabling_id,
                                funcname=slot_name))
                    except KeyError:
                        pass

                # Handle requiredIf in terms of sufficiency
                try:
                    for requiring_id in obj['requiredIf']:
                        sufficiency_links.append((
                            '        self.{requiring_id}.sufficiency_changed.connect(\n'
                            '            self.{id}.set_required)').format(
                                id=obj['id'],
                                requiring_id=requiring_id))
                except KeyError:
                    pass

                input_attributes.append(
                    INPUT_ATTRIBUTES_TEMPLATE.format(
                        name=obj['id'],
                        classname='inputs.%s' % classname,
                        kwargs=format_kwargs(kwargs)))

                if container_key:
                    add_string = '        self.{container}.add_input(self.{name})\n'
                else:
                    add_string = '        self.add_input(self.{name})\n'

                input_attributes.append(add_string.format(
                    container=container_key, name=obj['id']))

                if 'args_id' in obj:
                    try:
                        if (obj['type'].lower() != 'dropdown' and
                                obj['returns']['ifEmpty'].lower() == 'pass'):
                            args_to_maybe_skip.append((
                                '        if self.{name}.value():\n'
                                '            args[self.{name}.args_key] = '
                                'self.{name}.value()').format(name=obj['id']))
                    except KeyError:
                        args_values.append(
                            '            self.{name}.args_key: self.{name}.value(),'.format(
                                name=obj['id']))


        elif isinstance(obj, list):
            for item in obj:
                recurse(item, container_key)

    recurse(json_dict['elements'])

    input_attributes = [autopep8.reindent(
        autopep8.fix_code(line, options={'aggressive': 1}), 8).rstrip() for line in input_attributes]


#    for line in input_attributes:
#        print autopep8.fix_code(line, options={'aggressive': 1}).rstrip()


    with open(out_python_file, 'w') as out_file:
        out_file.write(UI_CLASS_TEMPLATE.format(
            label=repr(json_dict['label']),
            target='%s.execute' % json_dict['targetScript'],
            validator='%s.validate' % json_dict['targetScript'],
            localdoc=repr(json_dict['localDocURI']),
            input_attributes='\n'.join(input_attributes),
            classname=json_dict['modelName'].capitalize(),
            args_key_map='\n'.join(args_values),
            sufficiency_connections='\n'.join(sufficiency_links),
            args_to_maybe_skip='\n'.join(args_to_maybe_skip)
        ))


if __name__ == '__main__':
    convert_ui_structure(sys.argv[1], sys.argv[2])
