import importlib

from docutils import frontend
from docutils import utils
from docutils.parsers import rst
from natcap.invest import set_locale
from natcap.invest import spec


def parse_rst(text):
    """Parse RST text into a list of docutils nodes.

    Args:
        text (str): RST-formatted text to parse. May only use standard
            docutils features (no Sphinx roles etc)

    Returns:
        list[docutils.Node]
    """
    doc = utils.new_document(
        '',
        settings=frontend.OptionParser(
            components=(rst.Parser,)
        ).get_default_values())
    parser = rst.Parser()
    parser.parse(text, doc)

    # Skip the all-encompassing document node
    first_node = doc.next_node()
    # This is a list of the node and its siblings
    return list(first_node.findall(descend=False, siblings=True))


def get_input_from_key(module_name, *arg_keys):
    """Get the `Input` that corresponds to a given chain of keys

    Args:
        module_name (str): invest model module containing the arg.
        *arg_keys: one or more strings that are nested arg keys.

    Returns:
        spec.Input
    """
    # import the specified module (that should have an MODEL_SPEC attribute)
    module = importlib.import_module(module_name)

    # start with the spec for all args
    # narrow down to the nested spec indicated by the sequence of arg keys
    spec = module.MODEL_SPEC.get_input(arg_keys[0])
    arg_keys = arg_keys[1:]
    for i, key in enumerate(arg_keys):
        try:
            # convert raster band numbers to ints
            if i > 0 and arg_keys[i - 1] == 'bands':
                key = int(key)
            elif i > 0 and arg_keys[i - 1] == 'fields':
                spec = spec.get_field(key)
            elif i > 0 and arg_keys[i - 1] == 'contents':
                spec = spec.get_contents(key)
            elif i > 0 and arg_keys[i - 1] == 'columns':
                spec = spec.get_column(key)
            elif i > 0 and arg_keys[i - 1] == 'rows':
                # the attibute is called columns regardless of table orientation
                spec = spec.get_column(key)
            elif key in {'bands', 'fields', 'contents', 'columns', 'rows'}:
                continue
            else:
                spec = spec.getattr(key)
        except KeyError:
            keys_so_far = '.'.join(arg_keys[:i + 1])
            raise ValueError(
                f"Could not find the key '{keys_so_far}' in the "
                f"{module_name} model's MODEL_SPEC")
    return spec


def describe_input(module_name, keys):
    """Create RST description for a given model input.

    Args:
        module_name (str): name of the model module
        keys (list[str]): series of keys identifying the input

    Returns
        RST string
    """
    _input = get_input_from_key(module_name, *keys)
    # anchor names cannot contain underscores. sphinx will replace them
    # automatically, but lets explicitly replace them here
    anchor_name = '-'.join(keys).replace('_', '-')
    rst = '\n\n'.join(_input.describe_rst())
    return f'.. _{anchor_name}:\n\n{rst}'


def invest_spec(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Custom docutils role to generate InVEST model input docs from spec.

    This is a custom Sphinx extension that generates documentation of InVEST
    model inputs from the model's `MODEL_SPEC`. Its purpose is to help us reduce
    duplicated information and provide consistent, user-friendly documentation.
    The `investspec` extension provides the `:investspec:` role, which can be
    used inline in RST files to insert generated documentation anywhere you want.

    To use in a sphinx project, add 'natcap.invest.rst_generator' to the list of
    extensions in the conf.py.

    Usage:

    The `investspec` role takes two arguments: \:investspec:`module key`

    `module` (or `f'{investspec_module_prefix}.{module}'` if
    `investspec_module_prefix` is defined) must be an importable python module.
    It must have an attribute `MODEL_SPEC` .

    The second argument specifies which (nested) arg to document. It is a
    period-separated series of dictionary keys accessed starting at
    `MODEL_SPEC.args`. For example:

    - \:investspec:`annual_water_yield biophysical_table_path`

    - \:investspec:`annual_water_yield biophysical_table_path.columns.kc`

    Note that this implementation can only generate output that uses standard
    docutils features, and no sphinx-specific features.
    See natcap/invest.users-guide#35 for details.

    Docutils expects a function that accepts all of these args:

    Args:
        name (str): the local name of the interpreted text role, the role name
            actually used in the document.
        rawtext (str): a string containing the entire interpreted text
            construct. Return it as a ``problematic`` node linked to a system
            message if there is a problem.
        text (str): the interpreted text content, with backslash escapes
            converted to nulls (``\x00``).
        lineno (int): the line number where the interpreted text begins.
        inliner (Inliner): the Inliner object that called the role function.
            It defines the following useful attributes: ``reporter``,
            ``problematic``, ``memo``, ``parent``, ``document``.
        options (dict): A dictionary of directive options for customization, to
            be interpreted by the role function.  Used for additional
            attributes for the generated elements and other functionality.
        content (list[str]): the directive content for customization
            ("role" directive).  To be interpreted by the role function.

    Interpreted role functions return a tuple of two values:

    Returns:
        a tuple of two values:
            - A list of nodes which will be inserted into the document tree at
                the point where the interpreted role was encountered
            - A list of system messages, which will be inserted into the
                document tree immediately after the end of the current
                inline block.
    """
    # expect one or two space-separated arguments
    # the first argument is a module name to import (that has an MODEL_SPEC)
    # the second argument is a period-separated series of dictionary keys
    # that says what layer in the nested MODEL_SPEC dictionary to document
    arguments = text.split(' ', maxsplit=1)
    module_name = f'natcap.invest.{arguments[0]}'
    keys = arguments[1].split('.')  # period-separated series of keys
    # access the 'language' setting
    language = inliner.document.settings.env.app.config.language or 'en'
    set_locale(language)
    importlib.reload(importlib.import_module(name=module_name))
    return parse_rst(describe_input(module_name, keys)), []


def setup(app):
    """Add the custom extension to Sphinx.

    Sphinx calls this when it runs conf.py which contains
    `extensions = ['natcap.invest.investspec']`

    Args:
        app (sphinx.application.Sphinx)

    Returns:
        empty dictionary
    """
    app.add_role("investspec", invest_spec)
    return {}
