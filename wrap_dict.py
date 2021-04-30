"""
Script to wrap multi-line text blocks in dictionaries.
Made this for InVEST model ARGS_SPECs "about" properties since I'm 
moving things around and it's tedious to wrap them all by hand.
This could be generalized to any multi-line text block or an entire file.
"""
import ast
import importlib
import textwrap

max_width = 80
tab = '    '
space = ' '

def wrap_str(key, val, before, after, linebreak, indent=0):
    """Wrap a key-value pair from a dictionary.

    Args:
        key (str): the key of the key-value pair. e.g. "about" 
        val (str):
        linebreak (str): the linebreak character(s), typically \n or \r\n
        indent (int): how many spaces to indent the first line by.
            wrapped lines are indented by one additional tab.
        trailing_comma (bool): whether to append a comma to the end of
            the last line

    Returns:
        list[str]: list of complete wrapped lines (without newline chars)
    """
    print(':'.join([str(indent), before, key, val, after]))
    # see if it can all fit in one line
    one_line = f'{space * indent}{before}"{key}": "{val}"'
    if len(one_line) <= max_width:
        return [f'{one_line}{after}']

    wrapper = textwrap.TextWrapper()
    # allow room for a space and quote on the end (for the wrapped text)
    wrapper.width = max_width - 2
    # start the first line indented, with key: "...
    wrapper.initial_indent = f'{space * indent}{before}"{key}": ("'
    # start subsequent lines indented one more level 
    # and with a quote for the wrapped text block
    wrapper.subsequent_indent = f'{space * indent}{tab}"'
    # drop whitespace on the ends of wrapped lines
    # we will replace it to make sure each line ends with a space
    wrapper.drop_whitespace = True

    lines = wrapper.wrap(val)
    # add a quote to the end of each line of wrapped text
    for i in range(len(lines) - 1):
        lines[i] += ' "'
    # close the wrapped text block with a quote and parenthesis at the end
    lines[-1] += f'"{")" if after[0] != ")" else ""}{after}'
    # add the linebreak to the end of each line
    for i in range(len(lines) - 1):
        lines[i] += linebreak

    # check that they are all as short as expected
    for line in lines:
        if len(line) > max_width + len(linebreak):
            print(line, len(line))

    return lines

def wrap_str2(text, initial_indent=0, subsequent_indent=0):
    """Wrap a key-value pair from a dictionary.

    Args:
        key (str): the key of the key-value pair. e.g. "about" 
        val (str):
        indent (int): how many spaces to indent the first line by.
            wrapped lines are indented by one additional tab.

    Returns:
        list[str]: list of complete wrapped lines (without newline chars)
    """

    wrapper = textwrap.TextWrapper()
    # allow room for a space and quote on the end (for the wrapped text)
    wrapper.width = max_width - 2
    # start the first line indented, with "(...
    wrapper.initial_indent = f'{space * initial_indent}("'
    # start subsequent lines indented one more level 
    # and with a quote for the wrapped text block
    wrapper.subsequent_indent = f'{space * subsequent_indent}"'
    # drop whitespace on the ends of wrapped lines
    # we will replace it to make sure each line ends with a space
    wrapper.drop_whitespace = True

    lines = wrapper.wrap(text)
    # add a quote to the end of each line of wrapped text
    for i in range(len(lines) - 1):
        lines[i] += ' "'
    # close the wrapped text block with a quote and parenthesis at the end
    lines[-1] += '")'
    return lines


def wrap_dict(dic, indent=1):

    lines = []
    for key, val in dic.items():
        if type(val) is str:
            lines.append(wrap_str(key, val, indent=indent))
        elif type(val) is dict:
            dict_lines = []
            dict_lines.append(f'{tab * indent}"{key}": {{')
            dict_lines += wrap_dict(val, indent=indent + 1)
            dict_lines.append(tab * indent + '}')
            lines.append(dict_lines)
        else:
            lines.append([rf'{tab * indent}"{key}": {val}'])
    # add a comma to the end of all but the last dictionary item
    for i in range(len(lines) - 1):
        lines[i][-1] += ','

    lines = [l for line in lines for l in line]
    return lines




def wrap_strings2(ast_node, path):

    with open(path, newline='') as f:
        lines = f.readlines()

    # identify the linebreak (newline or carriage return)
    # avoid changing the linebreak unnecessarily (minimize diff size)
    if lines[0][-2:] == '\r\n':
        linebreak = '\r\n'
    elif lines[0][-1] == '\n':
        linebreak = '\n'
    else:
        raise ValueError(f'Unknown linebreak: {repr(lines[0][-1])}')


    changes = []
    for node in ast.walk(ast_node):
        if type(node) is ast.Dict:
            print(ast.dump(node))
            dict_node = node

            for i, (key, val) in enumerate(zip(dict_node.keys, dict_node.values)):
                if (isinstance(key, ast.Constant) and 
                    isinstance(val, ast.Constant) and
                    isinstance(val.value, str)):

                    initial_indent = len(lines[key.lineno - 1]) - len(lines[key.lineno - 1].lstrip())
                    subsequent_indent = initial_indent + len(tab)

                    stuff_before = lines[key.lineno - 1][initial_indent : key.col_offset]
                    stuff_after = lines[val.end_lineno - 1][val.end_col_offset:]

                    is_not_last_entry = i != len(dict_node.keys) - 1
                    wrapped = wrap_str(
                        key.value, 
                        val.value,
                        stuff_before,
                        stuff_after,
                        linebreak,
                        indent=initial_indent)
                    
                    print('\n\n\n')
                    for line in lines[key.lineno - 1 : val.end_lineno]:
                        print(line)
                    for line in wrapped:
                        print(line)
                    if wrapped != lines[key.lineno - 1 : val.end_lineno]:
                        changes.append((key.lineno - 1, val.end_lineno, wrapped))

    line_offset = 0
    for start, end, changed_lines in sorted(changes):
        print(start, end, (start - line_offset), (end - line_offset))
    
        print('overwriting:')
        for line in lines[(start - line_offset):(end - line_offset)]:
            print(line)
        print('with:')
        for line in changed_lines:
            print(line)
        # overwrite that section of the code
        lines = (
            lines[:start - line_offset] + 
            changed_lines + 
            lines[end - line_offset:])

        difference = (end - start) - len(changed_lines)
        line_offset += difference
        print('offset:', line_offset)


    # print out if any are longer than expected
    for i, line in enumerate(lines):
        if len(line) > max_width + len(linebreak):
            print(len(line), line)


    # overwrite the module file with the modified lines
    with open(path, 'w', newline='') as f:
        for line in lines:
            f.write(line)





def format_dict(dict_node, indent_width=0):
    """Format a dictionary nicely"""
    quote = '"'
    indent = ' ' * indent_width
    items = [
        f'{indent}{{'
    ]
    for i, (key, value) in enumerate(zip(dict_node.keys, dict_node.values)):
        # should always be a constant
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            key_string = f'{quote}{key.value}{quote}: '

        trailing_comma = ',' if (i != len(dict_node.keys) - 1) else ''

        if isinstance(val, ast.Constant):
            if isinstance(val.value, str):
                one_line = f'{indent}{key_string}"{val.value}"{trailing_comma}{linebreak}'
                if len(one_line) <= max_width + len(linebreak):
                    items.append(one_line)

                else:
                    wrapper = textwrap.TextWrapper()
                    # allow room for a space and quote on the end (for the wrapped text)
                    wrapper.width = max_width - 2
                    # start the first line indented, with key: "...
                    wrapper.initial_indent = f'{indent}{key_string}("'
                    # start subsequent lines indented one more level 
                    # and with a quote for the wrapped text block
                    wrapper.subsequent_indent = f'{indent}{tab}"'
                    # drop whitespace on the ends of wrapped lines
                    # we will replace it to make sure each line ends with a space
                    wrapper.drop_whitespace = True

                    lines = wrapper.wrap(val)
                    # add a space and quote to the end of each line of wrapped text
                    for i in range(len(lines) - 1):
                        lines[i] += f' "{linebreak}'
                    # close the wrapped text block with a quote and parenthesis at the end
                    lines[-1] += f'"){trailing_comma}{linebreak}'

                    # check that they are all as short as expected
                    for line in lines:
                        if len(line) > max_width + len(linebreak):
                            print(line, len(line))
                    items += lines


        elif isinstance(val.value, list):
            f'{key_string}: ['
            elif isinstance(val.value, tuple):

        elif isinstance(val, ast.Dict):
            items += format_dict(val)

    items.append(f'{indent}}}')
    return items


    def format_list()







if __name__ == '__main__':
   
    for model in [
        'carbon',
        'coastal_blue_carbon/coastal_blue_carbon',
        'coastal_blue_carbon/preprocessor',
        'coastal_vulnerability',
        'crop_production_regression',
        'crop_production_percentile',
        'delineateit/delineateit',
        'finfish_aquaculture/finfish_aquaculture',
        'fisheries/fisheries',
        'fisheries/fisheries_hst',
        'forest_carbon_edge_effect',
        'globio',
        'habitat_quality',
        'hra',
        'hydropower/hydropower_water_yield',
        'ndr/ndr',
        'pollination',
        'recreation/recmodel_client',
        'routedem',
        'scenic_quality/scenic_quality',
        'scenario_gen_proximity',
        'sdr/sdr',
        'seasonal_water_yield/seasonal_water_yield',
        'urban_cooling_model',
        'urban_flood_risk_mitigation',
        'wave_energy',
        'wind_energy'
    ]:
        path = f'src/natcap/invest/{model}.py'
        # parse the python source file into an abstract syntax tree
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in ast.iter_child_nodes(tree):
            if (isinstance(node, ast.Assign) and 
                hasattr(node.targets[0], 'id') and
                node.targets[0].id == 'ARGS_SPEC'):
                print(ast.dump(node))
                args_spec_node = node.value
                break

        wrap_strings2(args_spec_node, path)
