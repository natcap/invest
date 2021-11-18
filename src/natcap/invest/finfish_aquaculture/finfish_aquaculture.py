"""InVEST Finfish Aquaculture."""
import os
import csv
import logging

from natcap.invest.finfish_aquaculture import finfish_aquaculture_core
from .. import utils
from .. import spec_utils
from ..spec_utils import u
from .. import validation
from .. import MODEL_METADATA


LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": MODEL_METADATA["finfish_aquaculture"].model_title,
    "pyname": MODEL_METADATA["finfish_aquaculture"].pyname,
    "userguide_html": MODEL_METADATA["finfish_aquaculture"].userguide,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "ff_farm_loc": {
            "name": "finfish farm location",
            "about": "Map of finfish farm locations.",
            "type": "vector",
            "fields": {
                "[FARM_ID]": {  # may be anything, will be selected as the farm_ID
                    "type": "integer",
                    "about": (
                        "A unique identifier for each geometry. This field "
                        "name must be selected as the Farm Identifier Name.")
                }
            },
            "geometries": spec_utils.POLYGON | spec_utils.POINT,
        },
        "farm_ID": {
            "name": "farm identifier name",
            "about": (
                "Name of the field in the Finfish Farm Location map that "
                "contains a unique identifier for each farm geometry."),
            "type": "option_string",
            "options": {},
        },
        "g_param_a": {
            "name": "fish growth parameter α",
            "about": "Growth parameter α in the fish growth equation.",
            "type": "number",
            "units": u.gram/u.day,
        },
        "g_param_b": {
            "name": "fish growth parameter β",
            "about": "Growth parameter β in the fish growth equation.",
            "type": "number",
            "units": u.none,
        },
        "g_param_tau": {
            "name": "fish growth parameter τ",
            "about": (
                "Growth parameter τ in the fish growth equation. Specifies "
                "how sensitive finfish growth is to temperature."),
            "type": "number",
            "units": u.degree_Celsius**-1,
        },
        "use_uncertainty": {
            "name": "enable uncertainty analysis",
            "about": (
                "Run uncertainty analysis using a Monte Carlo simulation."),
            "type": "boolean",
        },
        "g_param_a_sd": {
            "name": "α standard deviation",
            "about": (
                "Standard deviation for fish growth parameter α. This "
                "indicates the level of uncertainty in the value of α. "
                "Required if Enable Uncertainty Analysis is selected."),
            "type": "number",
            "units": u.gram/u.day,
            "required": "use_uncertainty",
        },
        "g_param_b_sd": {
            "name": "β standard deviation",
            "about": (
                "Standard deviation for fish growth parameter β. This "
                "indicates the level of uncertainty in the value of β."
                "Required if Enable Uncertainty Analysis is selected."),
            "type": "number",
            "units": u.gram/u.day,
            "required": "use_uncertainty",
        },
        "num_monte_carlo_runs": {
            "name": "Monte Carlo simulation runs",
            "about": (
                "Number of times to run the model for the Monte Carlo "
                "simulation. "
                "Required if Enable Uncertainty Analysis is selected."),
            "type": "number",
            "units": u.count,
            "required": "use_uncertainty",
        },
        "water_temp_tbl": {
            "name": "daily water temperature table",
            "type": "csv",
            "about": (
                "Table of water temperatures in degrees Celsius for each farm "
                "on each day of the year. There are 365 rows (rows 6-370), "
                "each corresponding to a day of the year. The first two "
                "columns contain the number for that year (1-365) and the day "
                "and month. The following columns contain temperature data "
                "for each farm. Farm column headers must correspond to the "
                "farm's unique identifier in the Finfish Farm Location map.")
        },
        "farm_op_tbl": {
            "name": "farm operations table",
            "type": "csv",
            "about": (
                "A table of general and farm-specific operations parameters.")
        },
        "outplant_buffer": {
            "name": "Outplant Date Buffer",
            "type": "number",
            "units": u.day,
            "about": (
                "This value will allow the outplant start day to start plus "
                "or minus the number of days specified here."),
        },
        "do_valuation": {
            "name": "Run valuation model",
            "about": "Run valuation model.",
            "type": "boolean",
        },
        "p_per_kg": {
            "name": "price",
            "about": (
                "Market price of processed fish. "
                "Required if Run Valuation is selected."),
            "type": "number",
            "units": u.currency/u.kilogram,
            "required": "do_valuation",
        },
        "frac_p": {
            "name": "fraction of price accounting for costs",
            "about": (
                "Fraction of the market price that accounts for business "
                "expenses, rather than profit. "
                "Required if Run Valuation is selected."),
            "required": "do_valuation",
            "type": "ratio"
        },
        "discount": {
            "name": "daily market discount rate",
            "about": (
                "Daily market discount rate that reflects the preference for "
                "immediate benefits vs. future benefits. "
                "Required if Run Valuation is selected."),
            "required": "do_valuation",
            "type": "ratio"
        }
    }
}


def execute(args):
    """Finfish Aquaculture.

    This function will take care of preparing files passed into
    the finfish aquaculture model. It will handle all files/inputs associated
    with biophysical and valuation calculations and manipulations. It will
    create objects to be passed to the aquaculture_core.py module. It may
    write log, warning, or error messages to stdout.

    Args:
        workspace_dir (string): The directory in which to place all result
            files.
        results_suffix (string): (optional) string to append to any
            output file names
        ff_farm_loc (string): URI that points to a shape file of fishery
            locations
        farm_ID (string): column heading used to describe individual farms.
            Used to link GIS location data to later inputs.
        g_param_a (float): Growth parameter alpha, used in modeling fish
            growth, should be an int or float.
        g_param_b (float): Growth parameter beta, used in modeling fish growth,
            should be an int or float.
        g_param_tau (float): Growth parameter tau, used in modeling fish
            growth, should be an int or float
        use_uncertainty (boolean)
        g_param_a_sd (float): (description)
        g_param_b_sd (float): (description)
        num_monte_carlo_runs (int):
        water_temp_tbl (string): URI to a CSV table where daily water
            temperature values are stored from one year
        farm_op_tbl (string): URI to CSV table of static variables for
            calculations
        outplant_buffer (int): This value will allow the outplanting
            start day to be flexible plus or minus the number of days specified
            here.
        do_valuation (boolean): Boolean that indicates whether or not valuation
            should be performed on the aquaculture model
        p_per_kg (float): Market price per kilogram of processed fish
        frac_p (float): Fraction of market price that accounts for costs rather
            than profit
        discount (float): Daily market discount rate

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'results_suffix': 'test',
            'ff_farm_loc': 'path/to/shapefile',
            'farm_ID': 'FarmID'
            'g_param_a': 0.038,
            'g_param_b': 0.6667,
            'g_param_tau': 0.08,
            'use_uncertainty': True,
            'g_param_a_sd': 0.005,
            'g_param_b_sd': 0.05,
            'num_monte_carlo_runs': 1000,
            'water_temp_tbl': 'path/to/water_temp_tbl',
            'farm_op_tbl': 'path/to/farm_op_tbl',
            'outplant_buffer': 3,
            'do_valuation': True,
            'p_per_kg': 2.25,
            'frac_p': 0.3,
            'discount': 0.000192,
        }

    """
    # initialize new dictionary of purely biophysical/general arguments which
    # will be passed to the aquaculture core module. Then get desirable
    # arguments that are being passed in, and load them into the biophysical
    # dictionary.

    ff_aqua_args = {}

    workspace = args['workspace_dir']
    output_dir = workspace + os.sep + 'output'
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    if not (os.path.exists(output_dir)):
        LOGGER.debug('Creating output directory')
        os.makedirs(output_dir)

    ff_aqua_args['workspace_dir'] = args['workspace_dir']
    ff_aqua_args['results_suffix'] = file_suffix
    ff_aqua_args['ff_farm_file'] = args['ff_farm_loc']
    ff_aqua_args['farm_ID'] = args['farm_ID']
    ff_aqua_args['outplant_buffer'] = int(args['outplant_buffer'])
    ff_aqua_args['g_param_a'] = float(args['g_param_a'])
    ff_aqua_args['g_param_b'] = float(args['g_param_b'])
    ff_aqua_args['g_param_tau'] = float(args['g_param_tau'])

    if args['use_uncertainty']:
        LOGGER.debug('Adding uncertainty parameters')
        ff_aqua_args['num_monte_carlo_runs'] = int(
            args['num_monte_carlo_runs'])
        for key in ['g_param_a_sd', 'g_param_b_sd']:
            ff_aqua_args[key] = float(args[key])

    # Both CSVs are being pulled in, but need to do some maintenance to remove
    # undesirable information before they can be passed into core

    format_ops_table(args['farm_op_tbl'], "Farm #:", ff_aqua_args)
    format_temp_table(args['water_temp_tbl'], ff_aqua_args)

    ff_aqua_args['do_valuation'] = args['do_valuation']

    # Valuation arguments
    key = 'do_valuation'

    if ff_aqua_args['do_valuation'] is True:
        LOGGER.debug('Yes, we want to do valuation')

        ff_aqua_args['p_per_kg'] = float(args['p_per_kg'])
        ff_aqua_args['frac_p'] = float(args['frac_p'])
        ff_aqua_args['discount'] = float(args['discount'])

    # Fire up the biophysical function in finfish_aquaculture_core with the
    # gathered arguments
    LOGGER.debug('Starting finfish model')
    finfish_aquaculture_core.execute(ff_aqua_args)


def format_ops_table(op_path, farm_ID, ff_aqua_args):
    """Takes in the path to the operating parameters table as well as the
    keyword to look for to identify the farm number to go with the parameters,
    and outputs a 2D dictionary that contains all parameters by farm and
    description. The outer key is farm number, and the inner key is a string
    description of the parameter.

    Args:
        op_path: URI to CSV table of static variables for calculations
        farm_ID: The string to look for in order to identify the column in
            which the farm numbers are stored. That column data will become the
            keys for the dictionary output.
        ff_aqua_args: Dictionary of arguments being created in order to be
            passed to the aquaculture core function.
        ff_aqua_args['farm_op_dict']: A dictionary that is built up to store
            the static parameters for the aquaculture model run. This is a 2D
            dictionary, where the outer key is the farm ID number, and the
            inner keys are strings of parameter names.

    Returns:
        None
    """
    # NOTE: Have to do some explicit calls to strings here. This is BAD. Don't
    # do it if you don't have to. THESE EXPLICIT STRINGS COME FROM THE "Farm
    # Operations" table.

    new_dict_op = {}
    csv_file = open(op_path)

    # this will be separate arguments that are passed along straight into
    # biophysical_args
    general_ops = {}
    line = None

    dialect = csv.Sniffer().sniff(csv_file.read())
    csv_file.seek(0)

    delim = dialect.delimiter
    end_line = dialect.lineterminator

    while True:
        line = csv_file.readline().rstrip(end_line)

        if farm_ID in line:
            break

        split_line = line.split(delim)
        if 'Fraction of fish remaining after processing' in split_line[0]:
            general_ops['frac_post_process'] = float(split_line[1][:-1])/100

        if 'Natural mortality rate on the farm (daily)' in split_line[0]:
            general_ops['mort_rate_daily'] = split_line[1]

        if 'Duration of simulation (years)' in split_line[0]:
            general_ops['duration'] = split_line[1]

    # this is explicitly telling it the fields that I want to get data for
    # want to remove the 'Total Value' field, since there is not data inside
    # there, then tell the dictreader to set up a reader with dictionaries of
    # only those fields, where the overarching dictionary uses the Farm ID as
    # the key for each of the sub dictionaries
    fieldnames = line.split(delim)

    reader = csv.DictReader(
        csv_file,
        fieldnames=fieldnames,
        dialect=dialect,
        quoting=csv.QUOTE_NONE)

    for row in reader:

        sub_dict = {}

        for key in row:
            if (key != farm_ID):
                sub_dict[key] = row[key]

        if row[farm_ID] != '':
            new_dict_op[row[farm_ID]] = sub_dict

    ff_aqua_args['farm_op_dict'] = new_dict_op

    # add the gen args in
    for key in general_ops.keys():
        ff_aqua_args[key] = general_ops[key]


def format_temp_table(temp_path, ff_aqua_args):
    """ This function is doing much the same thing as format_ops_table- it
    takes in information from a temperature table, and is formatting it into a
    2D dictionary as an output.

    Input:
        temp_path: URI to a CSV file containing temperature data for 365 days
            for the farms on which we will look at growth cycles.
        ff_aqua_args: Dictionary of arguments that we are building up in order
            to pass it to the aquaculture core module.
    Output:
        ff_aqua_args['water_temp_dict']: A 2D dictionary containing temperature
            data for 365 days. The outer keys are days of the year from 0 to
            364 (we need to be able to check the day modulo 365) which we
            manually shift down by 1, and the inner keys are farm ID numbers.

    Returns nothing.
    """
    # EXPLICIT STRINGS FROM "Temp_Daily"
    water_temp_file = open(temp_path)

    new_dict_temp = {}
    line = None

    # This allows us to dynamically determine if the CSV file is comma
    # separated, or semicolon separated.
    dialect = csv.Sniffer().sniff(water_temp_file.read())
    water_temp_file.seek(0)
    delim = dialect.delimiter
    end_line = dialect.lineterminator

    # The farm ID numbers that fall under this column heading in the CSV will
    # be used as the keys in the second level of the dictionary.
    day_marker = 'Day #'

    while True:
        line = water_temp_file.readline().rstrip(end_line)
        if day_marker in line:
            break

    # this is explicitly telling it the fields that I want to get data for, and
    # am removing the Day/Month Field Since it's unnecessary
    fieldnames = line.split(delim)

    reader = csv.DictReader(
        water_temp_file,
        fieldnames,
        dialect=dialect)

    for row in reader:

        sub_dict = {}

        for key in row:
            if (key != day_marker and key != ''):
                sub_dict[key] = row[key]

        del sub_dict['Day/Month']

        # Subtract 1 here so that the day in the temp table allows for % 365
        new_dict_temp[str(int(row[day_marker]) - 1)] = sub_dict

    ff_aqua_args['water_temp_dict'] = new_dict_temp


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Finfish Aquaculture.

    Args:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    validation_warnings = validation.validate(args, ARGS_SPEC['args'])
    invalid_keys = validation.get_invalid_keys(validation_warnings)

    if 'ff_farm_loc' not in invalid_keys and 'farm_ID' not in invalid_keys:
        fieldnames = validation.load_fields_from_vector(
            args['ff_farm_loc'])
        error_msg = validation.check_option_string(args['farm_ID'],
                                                   fieldnames)
        if error_msg:
            validation_warnings.append((['farm_ID'], error_msg))

    return validation_warnings
