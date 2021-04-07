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

def wrap_str(key, val, indent=0):
    """Wrap a key-value pair from a dictionary.

    Args:
        key (str): the key of the key-value pair. e.g. "about" 
        val (str):
        indent (int): how many spaces to indent the first line by.
            wrapped lines are indented by one additional tab.

    Returns:
        list[str]: list of complete wrapped lines (without newline chars)
    """
    # see if it can all fit in one line
    one_line = f'{space * indent}"{key}": "{val}"'
    if len(one_line) <= max_width:
        return [one_line]

    wrapper = textwrap.TextWrapper()
    # allow room for a space and quote on the end (for the wrapped text)
    wrapper.width = max_width - 2
    # start the first line indented, with key: "...
    wrapper.initial_indent = f'{space * indent}"{key}": ("'
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



def wrap_strings(module, dict_name):
    # parse the python source file into an abstract syntax tree
    with open(module.__file__) as f:
        source = f.read()
        tree = ast.parse(source)

    with open(module.__file__, newline='') as f:
        lines = f.readlines()

    # identify the linebreak (newline or carriage return)
    # avoid changing the linebreak unnecessarily (minimize diff size)
    if lines[0][-2:] == '\r\n':
        linebreak = '\r\n'
    elif lines[0][-1] == '\n':
        linebreak = '\n'
    else:
        raise ValueError(f'Unknown linebreak: {repr(lines[0][-1])}')

    # find where the variable name is assigned to
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            if node.targets[0].id == dict_name:
                root_node = node
                start, end = node.lineno, node.end_lineno
                print(node.lineno, node.end_lineno)

    changes = []
    for node in ast.walk(root_node):
        if isinstance(node, ast.Dict):
            for i, (key, val) in enumerate(zip(node.keys, node.values)):
                if type(val) is ast.Constant and type(val.value) is str:
                    print('wrapping', val.value)
                    wrapped = wrap_str(key.value, val.value, indent=key.col_offset)
                    if wrapped[0] == f'{space * key.col_offset}"{key.value}": "{val.value}"':
                        print('no changes')
                        continue
                    if i < len(node.keys) - 1:
                        wrapped[-1] += ','
                    wrapped = [line + linebreak for line in wrapped]
                    print(key.lineno, key.end_lineno, val.lineno, val.end_lineno, wrapped)
                    # start inclusive, end exclusive
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
    for i, line in enumerate(wrapped):
        if len(line) > max_width + len(linebreak):
            print(len(line), line)


    # overwrite the module file with the modified lines
    with open(module.__file__, 'w', newline='') as f:
        print(module.__file__)
        for line in lines:
            f.write(line)



def wrap_dictionary(module, dict_name):
    """Modify source code to correctly wrap a dictionary in place.

    Args:
        module (module): imported module containing the dictionary
        dict_name (str): name of the dictionary variable to wrap

    Returns:
        None
    """
    # parse the python source file into an abstract syntax tree
    with open(module.__file__) as f:
        tree = ast.parse(f.read())

    # find where the variable name is assigned to
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            if node.targets[0].id == dict_name:
                start, end = node.lineno, node.end_lineno
                print(node.lineno, node.end_lineno)

    # read in the text of the file
    with open(module.__file__, newline='') as f:
        lines = f.readlines()
    # identify the linebreak (newline or carriage return)
    # avoid changing the linebreak unnecessarily (minimize diff size)
    if lines[0][-2:] == '\r\n':
        linebreak = '\r\n'
    elif lines[0][-1] == '\n':
        linebreak = '\n'
    else:
        raise ValueError(f'Unknown linebreak: {repr(lines[0][-1])}')

    # wrap each line to the max_width
    wrapped = wrap_dict(getattr(module, dict_name))
    wrapped = [line + linebreak for line in wrapped]

    # print out if any are longer than expected
    for i, line in enumerate(wrapped):
        if len(line) > max_width + len(linebreak):
            print(len(line), line)

    modified = lines[:start - 1]
    modified += [f'{dict_name} = {{{linebreak}']  # add the assignment line back in
    modified += wrapped
    modified += [f'}}{linebreak}']  # final closing bracket for the dict
    modified += lines[end:]

    # overwrite the module file with the modified lines
    with open(module.__file__, 'w', newline='') as f:
        for line in modified:
            f.write(line)


if __name__ == '__main__':
    module = importlib.import_module('src.natcap.invest.carbon')
    wrap_strings(module, 'ARGS_SPEC')
