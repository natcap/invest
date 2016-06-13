"""inVEST finfish aquaculture filehandler for biophysical and valuation data"""

import os
import csv
import logging

from natcap.invest.finfish_aquaculture import finfish_aquaculture_core

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('natcap.invest.finfish_aquaculture.finfish_aquaculture')


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
    ff_aqua_args['outplant_buffer'] = args['outplant_buffer']
    ff_aqua_args['g_param_a'] = args['g_param_a']
    ff_aqua_args['g_param_b'] = args['g_param_b']
    ff_aqua_args['g_param_tau'] = args['g_param_tau']

    if args['use_uncertainty']:
        LOGGER.debug('Adding uncertainty parameters')
        for key in ['g_param_a_sd', 'g_param_b_sd', 'num_monte_carlo_runs']:
            ff_aqua_args[key] = args[key]

    #Both CSVs are being pulled in, but need to do some maintenance to remove
    #undesirable information before they can be passed into core

    format_ops_table(args['farm_op_tbl'], "Farm #:", ff_aqua_args)
    format_temp_table(args['water_temp_tbl'], ff_aqua_args)

    ff_aqua_args['do_valuation'] = args['do_valuation']

    #Valuation arguments
    key = 'do_valuation'

    if ff_aqua_args['do_valuation'] is True:
        LOGGER.debug('Yes, we want to do valuation')

        ff_aqua_args['p_per_kg'] = args['p_per_kg']
        ff_aqua_args['frac_p'] = args['frac_p']
        ff_aqua_args['discount'] = args['discount']

    #Fire up the biophysical function in finfish_aquaculture_core with the
    #gathered arguments
    LOGGER.debug('Starting finfish model')
    finfish_aquaculture_core.execute(ff_aqua_args)


def format_ops_table(op_path, farm_ID, ff_aqua_args):
    '''Takes in the path to the operating parameters table as well as the
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
    '''

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
    ''' This function is doing much the same thing as format_ops_table- it
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
    '''

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
