"""InVEST Finfish Aquaculture"""
import os
import csv
import logging

from natcap.invest.finfish_aquaculture import finfish_aquaculture_core
from .. import validation


LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "module_name": "Finfish Aquaculture",
    "module": __name__,
    "userguide_html": "marine_fish.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "ff_farm_loc": {
            "name": "Finfish Farm Location",
            "about": (
                "A GDAL-supported vector file containing polygon or "
                "point geometries, with a latitude and longitude value and a "
                "numerical identifier for each farm.  File can be "
                "named anything, but no spaces in the name."),
            "type": "vector",
            "required": True,
        },
        "farm_ID": {
            "name": "Farm Identifier Name",
            "about": (
                "The name of a column heading used to identify each "
                "farm and link the spatial information from the "
                "vector to subsequent table input data (farm "
                "operation and daily water temperature at farm "
                "tables). Additionally, the numbers underneath this "
                "farm identifier name must be unique integers for all "
                "the inputs."),
            "type": "freestyle_string",
            "required": True,
        },
        "g_param_a": {
            "name": "Fish Growth Parameter (a)",
            "about": (
                "Default a = (0.038 g/day). If the user chooses to "
                "adjust these parameters, we recommend using them in "
                "the simple growth model to determine if the time "
                "taken for a fish to reach a target harvest weight "
                "typical for the region of interest is accurate."),
            "type": "number",
            "required": True,
        },
        "g_param_b": {
            "name": "Fish Growth Parameter (b)",
            "about": (
                "Default b = (0.6667 g/day). If the user chooses to "
                "adjust these parameters, we recommend using them in "
                "the simple growth model to determine if the time "
                "taken for a fish to reach a target harvest weight "
                "typical for the region of interest is accurate."),
            "type": "number",
            "required": True,
        },
        "g_param_tau": {
            "name": "Fish Growth Parameter (tau)",
            "about": (
                "Default tau = (0.08 C^-1).  Specifies how sensitive "
                "finfish growth is to temperature.  If the user "
                "chooses to adjust these parameters, we recommend "
                "using them in the simple growth model to determine if "
                "the time taken for a fish to reach a target harvest "
                "weight typical for the region of interest is "
                "accurate."),
            "type": "number",
            "required": True,
        },
        "use_uncertainty": {
            "name": "Enable uncertainty analysis",
            "about": "Enable uncertainty analysis.",
            "type": "boolean",
            "required": True,
        },
        "g_param_a_sd": {
            "name": "Standard Deviation for Parameter (a)",
            "about": (
                "Standard deviation for fish growth parameter a. "
                "This indicates the level of uncertainty in the "
                "estimate for parameter a."),
            "type": "number",
            "required": "use_uncertainty",
        },
        "g_param_b_sd": {
            "name": "Standard Deviation for Parameter (b)",
            "about": (
                "Standard deviation for fish growth parameter b. "
                "This indicates the level of uncertainty in the "
                "estimate for parameter b."),
            "type": "number",
            "required": "use_uncertainty",
        },
        "num_monte_carlo_runs": {
            "name": "Number of Monte Carlo Simulation Runs",
            "about": (
                "Number of runs of the model to perform as part of a "
                "Monte Carlo simulation.  A larger number will tend to "
                "produce more consistent and reliable output, but will "
                "also take longer to run."),
            "type": "number",
            "required": "use_uncertainty",
        },
        "water_temp_tbl": {
            "name": "Table of Daily Water Temperature at Farm",
            "type": "csv",
            "required": True,
            "about": (
                "Users must provide a time series of daily water "
                "temperature (C) for each farm in the vector.  When "
                "daily temperatures are not available, users can "
                "interpolate seasonal or monthly temperatures to a "
                "daily resolution.  Water temperatures collected at "
                "existing aquaculture facilities are preferable, but "
                "if unavailable, users can consult online sources such "
                "as NOAAs 4 km AVHRR Pathfinder Data and Canadas "
                "Department of Fisheries and Oceans Oceanographic "
                "Database.  The most appropriate temperatures to use "
                "are those from the upper portion of the water column, "
                "which are the temperatures experienced by the fish in "
                "the netpens."),
        },
        "farm_op_tbl": {
            "name": "Farm Operations Table",
            "type": "csv",
            "required": True,
            "about": (
                "A table of general and farm-specific operations "
                "parameters.  Please refer to the sample data table "
                "for reference to ensure correct incorporation of data "
                "in the model. The values for 'farm operations' "
                "(applied to all farms) and 'add new farms' (beginning "
                "with row 32) may be modified according to the user's "
                "needs . However, the location of cells in this "
                "template must not be modified.  If for example, if "
                "the model is to run for three farms only, the farms "
                "should be listed in rows 10, 11 and 12 (farms 1, 2, "
                "and 3, respectively). Several default values that are "
                "applicable to Atlantic salmon farming in British "
                "Columbia are also included in the sample data table."),
        },
        "outplant_buffer": {
            "name": "Outplant Date Buffer",
            "type": "number",
            "required": True,
            "about": (
                "This value will allow the outplant start day to "
                "start plus or minus the number of days specified "
                "here."),
        },
        "do_valuation": {
            "name": "Run valuation model",
            "about": "Run valuation model",
            "type": "boolean",
            "required": True,
        },
        "p_per_kg": {
            "name": "Market Price per Kilogram of Processed Fish",
            "about": (
                "Default value comes from Urner-Berry monthly fresh "
                "sheet reports on price of farmed Atlantic salmon."),
            "type": "number",
            "required": "do_valuation",
        },
        "frac_p": {
            "name": "Fraction of Price that Accounts to Costs",
            "about": (
                "Fraction of market price that accounts for costs "
                "rather than profit.  Default value is 0.3 (30%)."),
            "required": "do_valuation",
            "type": "number",
            "validation_options": {
                "expression": "(value >= 0) & (value <= 1)",
            }
        },
        "discount": {
            "name": "Daily Market Discount Rate",
            "about": (
                "We use a 7% annual discount rate, adjusted to a "
                "daily rate of 0.000192 for 0.0192% (7%/365 days)."),
            "required": "do_valuation",
            "type": "number",
            "validation_options": {
                "expression": "(value >= 0) & (value <= 1)",
            }
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

    #initialize new dictionary of purely biophysical/general arguments which
    #will be passed to the aquaculture core module. Then get desirable
    #arguments that are being passed in, and load them into the biophysical
    #dictionary.

    ff_aqua_args = {}

    workspace = args['workspace_dir']
    output_dir = workspace + os.sep + 'output'

    if not (os.path.exists(output_dir)):
        LOGGER.debug('Creating output directory')
        os.makedirs(output_dir)

    ff_aqua_args['workspace_dir'] = args['workspace_dir']
    ff_aqua_args['ff_farm_file'] = args['ff_farm_loc']
    ff_aqua_args['farm_ID'] = args['farm_ID']
    ff_aqua_args['outplant_buffer'] = int(args['outplant_buffer'])
    ff_aqua_args['g_param_a'] = float(args['g_param_a'])
    ff_aqua_args['g_param_b'] = float(args['g_param_b'])
    ff_aqua_args['g_param_tau'] = float(args['g_param_tau'])

    if args['use_uncertainty']:
        LOGGER.debug('Adding uncertainty parameters')
        ff_aqua_args['num_monte_carlo_runs'] = int(args['num_monte_carlo_runs'])
        for key in ['g_param_a_sd', 'g_param_b_sd']:
            ff_aqua_args[key] = float(args[key])

    #Both CSVs are being pulled in, but need to do some maintenance to remove
    #undesirable information before they can be passed into core

    format_ops_table(args['farm_op_tbl'], "Farm #:", ff_aqua_args)
    format_temp_table(args['water_temp_tbl'], ff_aqua_args)

    ff_aqua_args['do_valuation'] = args['do_valuation']

    #Valuation arguments
    key = 'do_valuation'

    if ff_aqua_args['do_valuation'] is True:
        LOGGER.debug('Yes, we want to do valuation')

        ff_aqua_args['p_per_kg'] = float(args['p_per_kg'])
        ff_aqua_args['frac_p'] = float(args['frac_p'])
        ff_aqua_args['discount'] = float(args['discount'])

    #Fire up the biophysical function in finfish_aquaculture_core with the
    #gathered arguments
    LOGGER.debug('Starting finfish model')
    finfish_aquaculture_core.execute(ff_aqua_args)


def format_ops_table(op_path, farm_ID, ff_aqua_args):
    """Takes in the path to the operating parameters table as well as the
    keyword to look for to identify the farm number to go with the parameters,
    and outputs a 2D dictionary that contains all parameters by farm and
    description. The outer key is farm number, and the inner key is a string
    description of the parameter.

    Input:
        op_path: URI to CSV table of static variables for calculations
        farm_ID: The string to look for in order to identify the column in
            which the farm numbers are stored. That column data will become the
            keys for the dictionary output.
        ff_aqua_args: Dictionary of arguments being created in order to be
            passed to the aquaculture core function.
    Output:
        ff_aqua_args['farm_op_dict']: A dictionary that is built up to store
            the static parameters for the aquaculture model run. This is a 2D
            dictionary, where the outer key is the farm ID number, and the
            inner keys are strings of parameter names.

    Returns nothing.
    """

    #NOTE: Have to do some explicit calls to strings here. This is BAD. Don't
    #do it if you don't have to. THESE EXPLICIT STRINGS COME FROM THE "Farm
    #Operations" table.

    new_dict_op = {}
    csv_file = open(op_path)

    #this will be separate arguments that are passed along straight into
    #biophysical_args
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

    #this is explicitly telling it the fields that I want to get data for
    #want to remove the 'Total Value' field, since there is not data inside
    #there, then tell the dictreader to set up a reader with dictionaries of
    #only those fields, where the overarching dictionary uses the Farm ID as
    #the key for each of the sub dictionaries
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

    #add the gen args in
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

    #EXPLICIT STRINGS FROM "Temp_Daily"
    water_temp_file = open(temp_path)

    new_dict_temp = {}
    line = None

    #This allows us to dynamically determine if the CSV file is comma
    #separated, or semicolon separated.
    dialect = csv.Sniffer().sniff(water_temp_file.read())
    water_temp_file.seek(0)
    delim = dialect.delimiter
    end_line = dialect.lineterminator

    #The farm ID numbers that fall under this column heading in the CSV will
    #be used as the keys in the second level of the dictionary.
    day_marker = 'Day #'

    while True:
        line = water_temp_file.readline().rstrip(end_line)
        if day_marker in line:
            break

    #this is explicitly telling it the fields that I want to get data for, and
    #am removing the Day/Month Field Since it's unnecessary
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

        #Subtract 1 here so that the day in the temp table allows for % 365
        new_dict_temp[str(int(row[day_marker]) - 1)] = sub_dict

    ff_aqua_args['water_temp_dict'] = new_dict_temp


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Finfish Aquaculture.

    Parameters:
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

