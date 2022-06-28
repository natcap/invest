import importlib
import json
import os

import pint

from . import gettext

# the same unit registry instance should be shared across everything
# load from custom unit defintions file
# don't raise warnings when redefining units
u = pint.UnitRegistry(on_redefinition='ignore')
u.load_definitions(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'unit_definitions.txt'))

# Specs for common arg types ##################################################
WORKSPACE = {
    "name": gettext("workspace"),
    "about": gettext(
        "The folder where all the model's output files will be written. If "
        "this folder does not exist, it will be created. If data already "
        "exists in the folder, it will be overwritten."),
    "type": "directory",
    "contents": {},
    "must_exist": False,
    "permissions": "rwx",
}

SUFFIX = {
    "name": gettext("file suffix"),
    "about": gettext(
        "Suffix that will be appended to all output file names. Useful to "
        "differentiate between model runs."),
    "type": "freestyle_string",
    "required": False,
    "regexp": "[a-zA-Z0-9_-]*"
}

N_WORKERS = {
    "name": gettext("taskgraph n_workers parameter"),
    "about": gettext(
        "The n_workers parameter to provide to taskgraph. "
        "-1 will cause all jobs to run synchronously. "
        "0 will run all jobs in the same process, but scheduling will take "
        "place asynchronously. Any other positive integer will cause that "
        "many processes to be spawned to execute tasks."),
    "type": "number",
    "units": u.none,
    "required": False,
    "expression": "value >= -1"
}

METER_RASTER = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.meter
        }
    }
}
AOI = {
    "type": "vector",
    "fields": {},
    "geometries": {"POLYGON", "MULTIPOLYGON"},
    "name": gettext("area of interest"),
    "about": gettext(
        "A map of areas over which to aggregate and "
        "summarize the final results."),
}
LULC = {
    "type": "raster",
    "bands": {1: {"type": "integer"}},
    "about": gettext("Map of land use/land cover codes."),
    "name": gettext("land use/land cover")
}
DEM = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.meter
        }
    },
    "about": gettext("Map of elevation above sea level."),
    "name": gettext("digital elevation model")
}
PRECIP = {
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.millimeter/u.year
        }
    },
    "about": gettext("Map of average annual precipitation."),
    "name": gettext("precipitation")
}
ET0 = {
    "name": gettext("evapotranspiration"),
    "type": "raster",
    "bands": {
        1: {
            "type": "number",
            "units": u.millimeter
        }
    },
    "about": gettext("Map of evapotranspiration values.")
}
SOIL_GROUP = {
    "type": "raster",
    "bands": {1: {"type": "integer"}},
    "about": gettext(
        "Map of soil hydrologic groups. Pixels may have values 1, 2, 3, or 4, "
        "corresponding to soil hydrologic groups A, B, C, or D, respectively."),
    "name": gettext("soil hydrologic group")
}
THRESHOLD_FLOW_ACCUMULATION = {
    "expression": "value >= 0",
    "type": "number",
    "units": u.pixel,
    "about": gettext(
        "The number of upslope pixels that must flow into a pixel "
        "before it is classified as a stream."),
    "name": gettext("threshold flow accumulation")
}

# geometry types ##############################################################
# the full list of ogr geometry types is in an enum in
# https://github.com/OSGeo/gdal/blob/master/gdal/ogr/ogr_core.h

POINT = {'POINT'}
LINESTRING = {'LINESTRING'}
POLYGON = {'POLYGON'}
MULTIPOINT = {'MULTIPOINT'}
MULTILINESTRING = {'MULTILINESTRING'}
MULTIPOLYGON = {'MULTIPOLYGON'}

LINES = LINESTRING | MULTILINESTRING
POLYGONS = POLYGON | MULTIPOLYGON
POINTS = POINT | MULTIPOINT
ALL_GEOMS = LINES | POLYGONS | POINTS


def format_unit(unit):
    """Represent a pint Unit as user-friendly unicode text.

    This attempts to follow the style guidelines from the NIST
    Guide to the SI (https://www.nist.gov/pml/special-publication-811):
    - Use standard symbols rather than spelling out
    - Use '/' to represent division
    - Use the center dot ' · ' to represent multiplication
    - Combine denominators into one, surrounded by parentheses

    Args:
        unit (pint.Unit): the unit to format

    Raises:
        TypeError if unit is not an instance of pint.Unit.

    Returns:
        String describing the unit.
    """
    if not isinstance(unit, pint.Unit):
        raise TypeError(
            f'{unit} is of type {type(unit)}. '
            f'It should be an instance of pint.Unit')

    # Optionally use a pre-set format for a particular unit
    custom_formats = {
        u.pixel: 'number of pixels',
        u.year_AD: '',  # don't need to mention units for a year input
        u.other: '',    # for inputs that can have any or multiple units
        # For soil erodibility (t*h*ha/(ha*MJ*mm)), by convention the ha's
        # are left on top and bottom and don't cancel out
        # pint always cancels units where it can, so add them back in here
        # this isn't a perfect solution
        # see https://github.com/hgrecco/pint/issues/1364
        u.t * u.hr / (u.MJ * u.mm): 't · h · ha / (ha · MJ · mm)',
        u.none: 'unitless'
    }
    if unit in custom_formats:
        return custom_formats[unit]

    # look up the abbreviated symbol for each unit
    # `formatter` expects an iterable of (unit, exponent) pairs, which lives in
    # the pint.Unit's `_units` attribute.
    unit_items = [(u.get_symbol(key), val) for key, val in unit._units.items()]
    return pint.formatting.formatter(
        unit_items,
        as_ratio=True,
        single_denominator=True,
        product_fmt=" · ",
        division_fmt='/',
        power_fmt="{}{}",
        parentheses_fmt="({})",
        exp_call=pint.formatting._pretty_fmt_exponent)


def serialize_args_spec(spec):
    """Serialize an ARGS_SPEC dict to a JSON string.

    Args:
        spec (dict): An invest model's ARGS_SPEC.

    Raises:
        TypeError if any object type within the spec is not handled by
        json.dumps or by the fallback serializer.

    Returns:
        JSON String
    """

    def fallback_serializer(obj):
        """Serialize objects that are otherwise not JSON serializeable."""
        if isinstance(obj, pint.Unit):
            return format_unit(obj)
        # Sets are present in 'geometries' attributes of some args
        # We don't need to worry about deserializing back to a set/array
        # so casting to string is okay.
        elif isinstance(obj, set):
            return str(obj)
        raise TypeError(f'fallback serializer is missing for {type(obj)}')

    return json.dumps(spec, default=fallback_serializer)


# accepted geometries for a vector will be displayed in this order
GEOMETRY_ORDER = [
    gettext('POINT'),
    gettext('MULTIPOINT'),
    gettext('LINESTRING'),
    gettext('MULTILINESTRING'),
    gettext('POLYGON'),
    gettext('MULTIPOLYGON')]

INPUT_TYPES_HTML_FILE = 'input_types.html'


def format_required_string(required):
    """Represent an arg's required status as a user-friendly string.

    Args:
        required (bool | str | None): required property of an arg. May be
            `True`, `False`, `None`, or a conditional string.

    Returns:
        string
    """
    if required is None or required is True:
        return gettext('required')
    elif required is False:
        return gettext('optional')
    else:
        # assume that the about text will describe the conditional
        return gettext('conditionally required')


def format_geometries_string(geometries):
    """Represent a set of allowed vector geometries as user-friendly text.

    Args:
        geometries (set(str)): set of geometry names

    Returns:
        string
    """
    # sort the geometries so they always display in a consistent order
    sorted_geoms = sorted(
        geometries,
        key=lambda g: GEOMETRY_ORDER.index(g))
    return '/'.join(geom.lower() for geom in sorted_geoms)


def format_permissions_string(permissions):
    """Represent a rwx-style permissions string as user-friendly text.

    Args:
        permissions (str): rwx-style permissions string

    Returns:
        string
    """
    permissions_strings = []
    if 'r' in permissions:
        permissions_strings.append(gettext('read'))
    if 'w' in permissions:
        permissions_strings.append(gettext('write'))
    if 'x' in permissions:
        permissions_strings.append(gettext('execute'))
    return ', '.join(permissions_strings)


def format_options_string_from_dict(options):
    """Represent a dictionary of option: description pairs as a bulleted list.

    Args:
        options (dict): the dictionary of options to document, where keys are
            options and values are descriptions of the options

    Returns:
        list of RST-formatted strings, where each is a line in a bullet list
    """
    lines = []
    # casefold() is a more aggressive version of lower() that may work better
    # for some languages to remove all case distinctions
    sorted_options = sorted(
        list(options.keys()),
        key=lambda option: option.casefold()
    )
    for option in sorted_options:
        lines.append(f'- {option}: {options[option]}')
    return lines


def format_options_string_from_list(options):
    """Represent options as a comma-separated list.

    Args:
        options (list[str]): the set of options to document

    Returns:
        string of comma-separated options
    """
    return ', '.join(options)


def capitalize(title):
    """Capitalize a string into title case.

    Args:
        title (str): string to capitalize

    Returns:
        capitalized string (each word capitalized except linking words)
    """

    def capitalize_word(word):
        """Capitalize a word, if appropriate."""
        if word in {'of', 'the'}:
            return word
        else:
            return word[0].upper() + word[1:]

    title = ' '.join([capitalize_word(word) for word in title.split(' ')])
    title = '/'.join([capitalize_word(word) for word in title.split('/')])
    return title


def format_type_string(arg_type):
    """Represent an arg type as a user-friendly string.

    Args:
        arg_type (str|set(str)): the type to format. May be a single type or a
            set of types.

    Returns:
        formatted string that links to a description of the input type(s)
    """
    # some types need a more user-friendly name
    # all types are listed here so that they can be marked up for translation
    type_names = {
        'boolean': gettext('true/false'),
        'csv': gettext('CSV'),
        'directory': gettext('directory'),
        'file': gettext('file'),
        'freestyle_string': gettext('text'),
        'integer': gettext('integer'),
        'number': gettext('number'),
        'option_string': gettext('option'),
        'percent': gettext('percent'),
        'raster': gettext('raster'),
        'ratio': gettext('ratio'),
        'vector': gettext('vector')
    }

    def format_single_type(arg_type):
        """Represent a type as a link to the corresponding Input Types section.

        Args:
            arg_type (str): the type to format.

        Returns:
            formatted string that links to a description of the input type
        """
        # Represent the type as a string. Some need a more user-friendly name.
        # we can only use standard docutils features here, so no :ref:
        # this syntax works to link to a section in a different page, but it
        # isn't universally supported and depends on knowing the built page name.
        if arg_type == 'freestyle_string':
            section_name = 'text'
        elif arg_type == 'option_string':
            section_name = 'option'
        elif arg_type == 'boolean':
            section_name = 'truefalse'
        elif arg_type == 'csv':
            section_name = 'csv'
        else:
            section_name = arg_type

        return f'`{type_names[arg_type]} <{INPUT_TYPES_HTML_FILE}#{section_name}>`__'

    if isinstance(arg_type, set):
        return ' or '.join(format_single_type(t) for t in sorted(arg_type))
    else:
        return format_single_type(arg_type)


def describe_arg_from_spec(name, spec):
    """Generate RST documentation for an arg, given an arg spec.

    This is used for documenting:
        - a single top-level arg
        - a row or column in a CSV
        - a field in a vector
        - an item in a directory

    Args:
        name (str): Name to give the section. For top-level args this is
            arg['name']. For nested args it's typically their key in the
            dictionary one level up.
        spec (dict): A arg spec dictionary that conforms to the InVEST args
            spec specification. It must at least have the key `'type'`, and
            whatever other keys are expected for that type.
    Returns:
        list of strings, where each string is a line of RST-formatted text.
        The first line has the arg name, type, required state, description,
        and units if applicable. Depending on the type, there may be additional
        lines that are indented, that describe details of the arg such as
        vector fields and geometries, option_string options, etc.
    """
    type_string = format_type_string(spec['type'])
    in_parentheses = [type_string]

    # For numbers and rasters that have units, display the units
    units = None
    if spec['type'] == 'number':
        units = spec['units']
    elif spec['type'] == 'raster' and spec['bands'][1]['type'] == 'number':
        units = spec['bands'][1]['units']
    if units:
        units_string = format_unit(units)
        if units_string:
            in_parentheses.append(f'units: **{units_string}**')

    if spec['type'] == 'vector':
        in_parentheses.append(format_geometries_string(spec["geometries"]))

    # Represent the required state as a string, defaulting to required
    # It doesn't make sense to include this for boolean checkboxes
    if spec['type'] != 'boolean':
        # get() returns None if the key doesn't exist in the dictionary
        required_string = format_required_string(spec.get('required'))
        in_parentheses.append(f'*{required_string}*')

    # Nested args may not have an about section
    if 'about' in spec:
        about_string = f': {spec["about"]}'
    else:
        about_string = ''

    first_line = f"**{name}** ({', '.join(in_parentheses)}){about_string}"

    # Add details for the types that have them
    indented_block = []
    if spec['type'] == 'option_string':
        # may be either a dict or set. if it's empty, the options are
        # dynamically generated. don't try to document them.
        if spec['options']:
            if isinstance(spec['options'], dict):
                indented_block.append(gettext('Options:'))
                indented_block += format_options_string_from_dict(spec['options'])
            else:
                formatted_options = format_options_string_from_list(spec['options'])
                indented_block.append(gettext('Options:') + f' {formatted_options}')

    elif spec['type'] == 'csv':
        if 'columns' not in spec and 'rows' not in spec:
            first_line += gettext(
                ' Please see the sample data table for details on the format.')

    # prepend the indent to each line in the indented block
    return [first_line] + ['\t' + line for line in indented_block]


def describe_arg_from_name(module_name, *arg_keys):
    """Generate RST documentation for an arg, given its model and name.

    Args:
        module_name (str): invest model module containing the arg.
        *arg_keys: one or more strings that are nested arg keys.

    Returns:
        String describing the arg in RST format. Contains an anchor named
        <arg_keys[0]>-<arg_keys[1]>...-<arg_keys[n]>
        where underscores in arg keys are replaced with hyphens.
    """
    # import the specified module (that should have an ARGS_SPEC attribute)
    module = importlib.import_module(module_name)
    # start with the spec for all args
    # narrow down to the nested spec indicated by the sequence of arg keys
    spec = module.ARGS_SPEC['args']
    for i, key in enumerate(arg_keys):
        # convert raster band numbers to ints
        if arg_keys[i - 1] == 'bands':
            key = int(key)
        try:
            spec = spec[key]
        except KeyError:
            keys_so_far = '.'.join(arg_keys[:i + 1])
            raise ValueError(
                f"Could not find the key '{keys_so_far}' in the "
                f"{module_name} model's ARGS_SPEC")

    # format spec into an RST formatted description string
    if 'name' in spec:
        arg_name = capitalize(spec['name'])
    else:
        arg_name = arg_keys[-1]

    # anchor names cannot contain underscores. sphinx will replace them
    # automatically, but lets explicitly replace them here
    anchor_name = '-'.join(arg_keys).replace('_', '-')
    rst_description = '\n\n'.join(describe_arg_from_spec(arg_name, spec))
    return f'.. _{anchor_name}:\n\n{rst_description}'
