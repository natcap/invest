'''
The Fisheries IO module contains functions for handling inputs and outputs
'''

import logging
import os
import csv

from osgeo import ogr
import numpy as np

import pygeoprocessing.geoprocessing
import pygeoprocessing.testing
from natcap.invest import reporting

LOGGER = logging.getLogger('natcap.invest.fisheries.io')


class MissingParameter(ValueError):
    '''
    An exception class that may be raised when a necessary parameter is not
    provided by the user.
    '''
    pass


# Fetch and Verify Arguments
def fetch_args(args, create_outputs=True):
    '''
    Fetches input arguments from the user, verifies for correctness and
    completeness, and returns a list of variables dictionaries

    Args:
        args (dictionary): arguments from the user

    Returns:
        model_list (list): set of variable dictionaries for each
            model

    Example Returns::

        model_list = [
            {
                'workspace_dir': 'path/to/workspace_dir',
                'results_suffix': 'scenario_name',
                'output_dir': 'path/to/output_dir',
                'aoi_uri': 'path/to/aoi_uri',
                'total_timesteps': 100,
                'population_type': 'Stage-Based',
                'sexsp': 2,
                'harvest_units': 'Individuals',
                'do_batch': False,
                'spawn_units': 'Weight',
                'total_init_recruits': 100.0,
                'recruitment_type': 'Ricker',
                'alpha': 32.4,
                'beta': 54.2,
                'total_recur_recruits': 92.1,
                'migr_cont': True,
                'val_cont': True,
                'frac_post_process': 0.5,
                'unit_price': 5.0,

                # Pop Params
                'population_csv_uri': 'path/to/csv_uri',
                'Survnaturalfrac': np.array(
                    [[[...], [...]], [[...], [...]], ...]),
                'Classes': np.array([...]),
                'Vulnfishing': np.array([...], [...]),
                'Maturity': np.array([...], [...]),
                'Duration': np.array([...], [...]),
                'Weight': np.array([...], [...]),
                'Fecundity': np.array([...], [...]),
                'Regions': np.array([...]),
                'Exploitationfraction': np.array([...]),
                'Larvaldispersal': np.array([...]),

                # Mig Params
                'migration_dir': 'path/to/mig_dir',
                'Migration': [np.matrix, np.matrix, ...]
            },
            {
                ...  # additional dictionary doesn't exist when 'do_batch'
                     # is false
            }
        ]

    Note:
        This function receives an unmodified 'args' dictionary from the user

    '''
    args['do_batch'] = bool(args['do_batch'])

    try:
        args['results_suffix']
    except:
        args['results_suffix'] = ''

    sexsp_dict = {
        'yes': 2,
        'no': 1,
    }
    args['sexsp'] = sexsp_dict[args['sexsp'].lower()]

    params_dict = _verify_single_params(args, create_outputs=create_outputs)

    # Implement Single / Batch Here
    pop_list = read_population_csvs(args)

    mig_dict = read_migration_tables(
        args, pop_list[0]['Classes'], pop_list[0]['Regions'])

    # Create model_list Here
    model_list = []
    for pop_dict in pop_list:
        vars_dict = dict(pop_dict.items() + mig_dict.items() +
                         params_dict.items())
        model_list.append(vars_dict)

    return model_list


def read_population_csvs(args):
    '''
    Parses and verifies the Population Parameters CSV files

    Args:
        args (dictionary): arguments provided by user

    Returns:
        pop_list (list): list of dictionaries containing verified population
            arguments

    Example Returns::

        pop_list = [
            {
                'Survnaturalfrac': np.array(
                    [[...], [...]], [[...], [...]], ...),

                # Class Vectors
                'Classes': np.array([...]),
                'Vulnfishing': np.array([...], [...]),
                'Maturity': np.array([...], [...]),
                'Duration': np.array([...], [...]),
                'Weight': np.array([...], [...]),
                'Fecundity': np.array([...], [...]),

                # Region Vectors
                'Regions': np.array([...]),
                'Exploitationfraction': np.array([...]),
                'Larvaldispersal': np.array([...]),
            },
            {
                ...
            }
        ]
    '''
    if args['do_batch'] is False:
        population_csv_uri_list = [args['population_csv_uri']]
    else:
        population_csv_uri_list = _listdir(
            args['population_csv_dir'])

    pop_list = []
    for uri in population_csv_uri_list:
        ext = os.path.splitext(uri)[1]
        if ext == '.csv':
            pop_dict = read_population_csv(args, uri)
            pop_list.append(pop_dict)

    return pop_list


def read_population_csv(args, uri):
    '''
    Parses and verifies a single Population Parameters CSV file

    Parses and verifies inputs from the Population Parameters CSV file.
    If not all necessary vectors are included, the function will raise a
    MissingParameter exception. Survival matrix will be arranged by
    class-elements, 2nd dim: sex, and 3rd dim: region. Class vectors will
    be arranged by class-elements, 2nd dim: sex (depending on whether model
    is sex-specific) Region vectors will be arraged by region-elements,
    sex-agnostic.

    Args:
        args (dictionary): arguments provided by user
        uri (string): the particular Population Parameters CSV file to
            parse and verifiy

    Returns:
        pop_dict (dictionary): dictionary containing verified population
            arguments

    Example Returns::

        pop_dict = {
            'population_csv_uri': 'path/to/csv',
            'Survnaturalfrac': np.array(
                [[...], [...]], [[...], [...]], ...),

            # Class Vectors
            'Classes': np.array([...]),
            'Vulnfishing': np.array([...], [...]),
            'Maturity': np.array([...], [...]),
            'Duration': np.array([...], [...]),
            'Weight': np.array([...], [...]),
            'Fecundity': np.array([...], [...]),

            # Region Vectors
            'Regions': np.array([...]),
            'Exploitationfraction': np.array([...]),
            'Larvaldispersal': np.array([...]),
        }
    '''
    pop_dict = _parse_population_csv(uri, args['sexsp'])
    pop_dict['population_csv_uri'] = uri

    # Check that required information exists
    Necessary_Params = ['Classes', 'Exploitationfraction', 'Regions',
                        'Survnaturalfrac', 'Vulnfishing']
    Matching_Params = [i for i in pop_dict.keys() if i in Necessary_Params]
    assert len(Matching_Params) == len(Necessary_Params), (
        "Population Parameters File does not contain all necessary parameters")

    if (args['recruitment_type'] != 'Fixed'):
        assert 'Maturity' in pop_dict.keys(), (
            "Population Parameters File must contain a 'Maturity' vector when "
            "running the given recruitment function. %s" % uri)

    if (args['population_type'] == 'Stage-Based'):
        assert 'Duration' in pop_dict.keys(), (
            "Population Parameters File must contain a 'Duration' vector when "
            "running Stage-Based models. %s" % uri)

    if (args['recruitment_type'] in ['Beverton-Holt', 'Ricker']) and (
            args['spawn_units'] == 'Weight'):
        assert 'Weight' in pop_dict.keys(), (
            "Population Parameters File must contain a 'Weight' vector when "
            "Spawners are calulated by weight using the Beverton-Holt or "
            "Ricker recruitment functions. %s" % uri)

    if (args['harvest_units'] == 'Weight'):
        assert 'Weight' in pop_dict.keys(), (
            "Population Parameters File must contain a 'Weight' vector when "
            "'Harvest by Weight' is selected. %s" % uri)

    if (args['recruitment_type'] == 'Fecundity'):
        assert 'Fecundity' in pop_dict.keys(), (
            "Population Parameters File must contain a 'Fecundity' vector "
            "when using the Fecundity recruitment function. %s" % uri)

    # Make sure parameters are initialized even when user does not enter data
    if 'Larvaldispersal' not in pop_dict.keys():
        num_regions = len(pop_dict['Regions'])
        pop_dict['Larvaldispersal'] = (np.array(np.ones(num_regions) /
                                       num_regions))

    # Check that similar vectors have same shapes (NOTE: checks region vectors)
    assert (pop_dict['Larvaldispersal'].shape ==
            pop_dict['Exploitationfraction'].shape), (
                "Region vector shapes do not match. %s" % uri)

    # Check that information is correct
    assert pygeoprocessing.testing.isclose(
        pop_dict['Larvaldispersal'].sum(), 1), (
            "The Larvaldisperal vector does not sum exactly to one.. %s" % uri)

    # Check that certain attributes have fraction elements
    Frac_Vectors = ['Survnaturalfrac', 'Vulnfishing',
                    'Exploitationfraction']
    if args['recruitment_type'] != 'Fixed':
        Frac_Vectors.append('Maturity')
    for attr in Frac_Vectors:
        a = pop_dict[attr]
        assert (a.min() >= 0.0 and a.max() <= 1.0), (
            "The %s vector has elements that are not decimal "
            "fractions. %s") % (attr, uri)

    # Make duration vector of type integer
    if args['population_type'] == 'Stage-Based':
        pop_dict['Duration'] = np.array(
            pop_dict['Duration'], dtype=int)

    # Fill in unused keys with null values
    All_Parameters = ['Classes', 'Duration', 'Exploitationfraction',
                      'Fecundity', 'Larvaldispersal', 'Maturity', 'Regions',
                      'Survnaturalfrac', 'Weight', 'Vulnfishing']
    for parameter in All_Parameters:
        if parameter not in pop_dict.keys():
            pop_dict[parameter] = None

    return pop_dict


def _parse_population_csv(uri, sexsp):
    '''
    Parses the given Population Parameters CSV file and returns a dictionary
    of lists, arrays, and matrices

    Dictionary items containing lists, arrays or matrices are capitalized,
    while single variables are lowercase.

    Keys: Survnaturalfrac, Vulnfishing, Maturity, Duration, Weight, Fecundity,
            Exploitationfraction, Larvaldispersal, Classes, Regions

    Args:
        uri (string): uri to population parameters csv file
        sexsp (int): indicates whether classes are distinguished by sex

    Returns:
        pop_dict (dictionary): verified population arguments

    Example Returns:

        pop_dict = {
            'Survnaturalfrac': np.array(
                [[...], [...]], [[...], [...]], ...),
            'Vulnfishing': np.array([...], [...]),
            ...
        }
    '''
    csv_data = []
    pop_dict = {}

    with open(uri, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_data.append(line)

    start_rows = _get_table_row_start_indexes(csv_data)
    start_cols = _get_table_col_start_indexes(csv_data, start_rows[0])
    end_rows = _get_table_row_end_indexes(csv_data)
    end_cols = _get_table_col_end_indexes(csv_data, start_rows[0])

    classes = _get_col(
        csv_data, start_rows[0])[0:end_rows[0]+1]

    pop_dict["Classes"] = map(lambda x: x.lower(), classes[1:])
    if sexsp == 2:
        pop_dict["Classes"] = pop_dict["Classes"][0:len(pop_dict["Classes"])/2]

    regions = _get_row(csv_data, start_cols[0])[0: end_cols[0]+1]

    pop_dict["Regions"] = regions[1:]

    surv_table = _get_table(csv_data, start_rows[0]+1, start_cols[0]+1)
    class_attributes_table = _get_table(
        csv_data, start_rows[0], start_cols[1])
    region_attributes_table = _get_table(
        csv_data, start_rows[1], start_cols[0])

    assert sexsp in (1, 2), 'Sex-specificity must be one of (1, 2)'
    if sexsp == 1:
        # Sex Neutral
        pop_dict['Survnaturalfrac'] = np.array(
            [surv_table], dtype=np.float_).swapaxes(1, 2).swapaxes(0, 1)
    elif sexsp == 2:
        # Sex Specific
        female = np.array(surv_table[0:len(surv_table)/sexsp], dtype=np.float_)
        male = np.array(surv_table[len(surv_table)/sexsp:], dtype=np.float_)
        pop_dict['Survnaturalfrac'] = np.array(
            [female, male]).swapaxes(1, 2).swapaxes(0, 1)

    for col in range(0, len(class_attributes_table[0])):
        pop_dict.update(_vectorize_attribute(
            _get_col(class_attributes_table, col), sexsp))

    for row in range(0, len(region_attributes_table)):
        pop_dict.update(_vectorize_reg_attribute(
            _get_row(region_attributes_table, row)))

    return pop_dict


def read_migration_tables(args, class_list, region_list):
    '''
    Parses, verifies and orders list of migration matrices necessary for
    program.

    Args:
        args (dictionary): same args as model entry point
        class_list (list): list of class names
        region_list (list): list of region names

    Returns:
        mig_dict (dictionary): see example below

    Example Returns::

        mig_dict = {
            'Migration': [np.matrix, np.matrix, ...]
        }

    Note:
        If migration matrices are not provided for all classes, the function will
        generate identity matrices for missing classes
    '''
    migration_dict = {}

    # If Migration:
    mig_dict = _parse_migration_tables(args, class_list)

    # Create indexed list
    matrix_list = map(lambda x: None, class_list)

    # Map np.matrices to indices in list
    for i in range(0, len(class_list)):
        if class_list[i] in mig_dict.keys():
            matrix_list[i] = mig_dict[class_list[i]]

    # Fill in rest with identity matrices
    for i in range(0, len(matrix_list)):
        if matrix_list[i] is None:
            matrix_list[i] = np.matrix(np.identity(len(region_list)))

    # Check migration regions are equal across matrices
    assert all((x.shape == matrix_list[0].shape for x in matrix_list)), (
        "Shape of migration matrices are not equal across lifecycle classes")

    # Check that all migration vectors approximately sum to one
    if not all((np.allclose(vector.sum(), 1)
                for matrix in matrix_list for vector in matrix)):
        LOGGER.warning("Elements in at least one migration matrices source "
                       "vector do not sum to one")

    migration_dict['Migration'] = matrix_list
    return migration_dict


def _parse_migration_tables(args, class_list):
    '''
    Parses the migration tables given by user

    Parses all files in the given directory as migration matrices and returns a
    dictionary of stages and their corresponding migration numpy matrix. If
    extra files are provided that do not match the class names, an exception
    will be thrown.

    Args:
        uri (string): filepath to the directory of migration tables

    Returns:
        mig_dict (dictionary)

    Example Returns::

        mig_dict = {
            {'stage1': np.matrix},
            {'stage2': np.matrix},
            # ...
        }
    '''
    mig_dict = {}

    if args['migr_cont']:
        uri = os.path.abspath(args['migration_dir'])
        for mig_csv in _listdir(uri):
            basename = os.path.splitext(os.path.basename(mig_csv))[0]
            class_name = basename.split('_').pop().lower()
            if class_name.lower() in class_list:
                LOGGER.info('Parsing csv %s for class %s', mig_csv,
                            class_name)
                with open(mig_csv, 'rU') as param_file:
                    csv_reader = csv.reader(param_file)
                    lines = []
                    for row in csv_reader:
                        lines.append(row)

                    matrix = []
                    for row in range(1, len(lines)):
                        array = []
                        for entry in range(1, len(lines[row])):
                            array.append(float(lines[row][entry]))
                        matrix.append(array)

                    Migration = np.matrix(matrix)

                mig_dict[class_name] = Migration

    return mig_dict


def _verify_single_params(args, create_outputs=True):
    '''
    Example Returned Parameters Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'population_csv_uri': 'path/to/csv_uri',
            'migration_dir': 'path/to/mig_dir',
            'aoi_uri': 'path/to/aoi_uri',
            'total_timesteps': 100,
            'population_type': 'Stage-Based',
            'sexsp': 2,
            'harvest_units': 'Individuals',
            'do_batch': False,
            'population_csv_uri': 'path/to/csv_uri',
            'population_csv_dir': ''
            'spawn_units': 'Weight',
            'total_init_recruits': 100.0,
            'recruitment_type': 'Ricker',
            'alpha': 32.4,
            'beta': 54.2,
            'total_recur_recruits': 92.1,
            'migr_cont': True,
            'val_cont': True,
            'frac_post_process': 0.5,
            'unit_price': 5.0,
            # ...
            'output_dir': 'path/to/output_dir',
            'intermediate_dir': 'path/to/intermediate_dir',
        }
    '''
    params_dict = args

    if create_outputs:
        # Create output directory
        output_dir = os.path.join(args['workspace_dir'], 'output')
        intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
        params_dict['output_dir'] = output_dir
        params_dict['intermediate_dir'] = intermediate_dir
        pygeoprocessing.create_directories([args['workspace_dir'],
                                            output_dir,
                                            intermediate_dir])

    # Check that timesteps is positive integer
    params_dict['total_timesteps'] = int(args['total_timesteps']) + 1

    return params_dict


# Helper function
def _listdir(path):
    '''
    A replacement for the standar os.listdir which, instead of returning
    only the filename, will include the entire path. This will use os as a
    base, then just lambda transform the whole list.

    Args:
        path (string): the location container from which we want to
            gather all files

    Returns:
        uris (list): A list of full URIs contained within 'path'
    '''
    file_names = os.listdir(path)
    uris = map(lambda x: os.path.join(path, x), file_names)

    return uris


# Helper functions for navigating CSV files
def _get_col(lsts, col):
    l = []
    for row in range(0, len(lsts)):
        if lsts[row][col] != '':
            l.append(lsts[row][col])
    return l


def _get_row(lsts, row):
    l = []
    for entry in range(0, len(lsts[row])):
        if lsts[row][entry] != '':
            l.append(lsts[row][entry])
    return l


def _get_table(lsts, row, col):
    table = []

    end_col = col
    while end_col + 1 < len(lsts[0]) and lsts[0][end_col + 1] != '':
        end_col += 1

    end_row = row
    while end_row + 1 < len(lsts) and lsts[end_row + 1][0] != '':
        end_row += 1

    for line in range(row, end_row + 1):
        table.append(lsts[line][col:end_col + 1])
    return table


def _get_table_row_start_indexes(lsts):
    indexes = []
    if lsts[0][0] != '':
        indexes.append(0)
    for line in range(1, len(lsts)):
        if lsts[line - 1][0] == '' and lsts[line][0] != '':
            indexes.append(line)
    return indexes


def _get_table_col_start_indexes(lsts, top):
    indexes = []
    if lsts[top][0] != '':
        indexes.append(0)
    for col in range(1, len(lsts[top])):
        if lsts[top][col - 1] == '' and lsts[top][col] != '':
            indexes.append(col)
    return indexes


def _get_table_row_end_indexes(lsts):
    indexes = []
    for i in range(1, len(lsts)):
        if (lsts[i][0] == '' and lsts[i-1][0] != ''):
            indexes.append(i-1)
    if lsts[-1] != '':
        indexes.append(len(lsts)-1)

    return indexes


def _get_table_col_end_indexes(lsts, top):
    indexes = []
    for i in range(1, len(lsts[top])):
        if (lsts[top][i] == '' and lsts[top][i-1] != ''):
            indexes.append(i-1)
    if lsts[top][-1] != '':
        indexes.append(len(lsts[top])-1)

    return indexes


def _vectorize_attribute(lst, rows):
    d = {}
    a = np.array(lst[1:], dtype=np.float_)
    a = np.reshape(a, (rows, a.shape[0] / rows))
    d[lst[0].strip().capitalize()] = a
    return d


def _vectorize_reg_attribute(lst):
    d = {}
    a = np.array(lst[1:], dtype=np.float_)
    d[lst[0].strip().capitalize()] = a
    return d


# Create Outputs
def create_outputs(vars_dict):
    '''
    Creates outputs from variables generated in the run_population_model()
    function in the fisheries_model module

    Creates the following:

        + Results CSV File
        + Results HTML Page
        + Results Shapefile (if provided)
        + Intermediate CSV File

    Args:
        vars_dict (dictionary): contains variables generated by model run

    '''
    # CSV results page
    _create_intermediate_csv(vars_dict)
    _create_results_csv(vars_dict)
    # HTML results page
    _create_results_html(vars_dict)
    # Append Results to Shapefile
    if vars_dict['aoi_uri']:
        _create_results_aoi(vars_dict)


def _create_intermediate_csv(vars_dict):
    '''
    Creates an intermediate output that gives the number of
    individuals within each area for each time step for each age/stage.
    '''
    do_batch = vars_dict['do_batch']
    if do_batch is True:
        basename = os.path.splitext(os.path.basename(
            vars_dict['population_csv_uri']))[0]
        filename = 'population_by_time_step_' + basename + '.csv'
    elif vars_dict['results_suffix'] is not '':
        filename = 'population_by_time_step_' + vars_dict[
            'results_suffix'] + '.csv'
    else:
        filename = 'population_by_time_step.csv'
    uri = os.path.join(
        vars_dict['intermediate_dir'], filename)

    Regions = vars_dict['Regions']
    Classes = vars_dict['Classes']
    N_tasx = vars_dict['N_tasx']
    N_txsa = N_tasx.swapaxes(1, 3)
    sexsp = vars_dict['sexsp']
    Sexes = ['Female', 'Male']

    with open(uri, 'wb') as c_file:
        # c_writer = csv.writer(c_file)
        if sexsp == 2:
            line = "Time Step, Region, Class, Sex, Numbers\n"
            c_file.write(line)
        else:
            line = "Time Step, Region, Class, Numbers\n"
            c_file.write(line)

        for t in range(0, len(N_txsa)):
            for x in range(0, len(Regions)):
                for a in range(0, len(Classes)):
                    if sexsp == 2:
                        for s in range(0, 2):
                            line = "%i, %s, %s, %s, %f\n" % (
                                t,
                                Regions[x],
                                Classes[a],
                                Sexes[s],
                                N_txsa[t, x, s, a])
                            c_file.write(line)
                    else:
                        line = "%i, %s, %s, %f\n" % (
                            t, Regions[x],
                            Classes[a],
                            N_txsa[t, x, 0, a])
                        c_file.write(line)


def _create_results_csv(vars_dict):
    '''
    Generates a CSV file that contains a summary of all harvest totals
    for each subregion.
    '''
    do_batch = vars_dict['do_batch']
    if do_batch is True:
        basename = os.path.splitext(os.path.basename(
            vars_dict['population_csv_uri']))[0]
        filename = 'results_table_' + basename + '.csv'
    elif vars_dict['results_suffix'] is not '':
        filename = 'results_table_' + vars_dict['results_suffix'] + '.csv'
    else:
        filename = 'results_table.csv'
    uri = os.path.join(vars_dict['output_dir'], filename)

    recruitment_type = vars_dict['recruitment_type']
    Spawners_t = vars_dict['Spawners_t']
    H_tx = vars_dict['H_tx']
    V_tx = vars_dict['V_tx']
    equilibrate_timestep = int(vars_dict['equilibrate_timestep'])
    Regions = vars_dict['Regions']

    with open(uri, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file)

        total_timesteps = vars_dict['total_timesteps']

        #Header for final results table
        csv_writer.writerow(
            ['Final Harvest by Subregion after ' + str(total_timesteps-1) +
                ' Time Steps'])
        csv_writer.writerow([])

        # Breakdown Harvest and Valuation for each Region of Final Cycle
        sum_headers_row = ['Subregion', 'Harvest']
        if vars_dict['val_cont']:
            sum_headers_row.append('Valuation')
        csv_writer.writerow(sum_headers_row)
        for i in range(0, len(H_tx[-1])):  # i is a cycle
            line = [Regions[i], "%.2f" % H_tx[-1, i]]
            if vars_dict['val_cont']:
                line.append("%.2f" % V_tx[-1, i])
            csv_writer.writerow(line)
        line = ['Total', "%.2f" % H_tx[-1].sum()]
        if vars_dict['val_cont']:
            line.append("%.2f" % V_tx[-1].sum())
        csv_writer.writerow(line)
        csv_writer.writerow([])

        # Give Total Harvest for Each Cycle
        csv_writer.writerow(['Time Step Breakdown'])
        csv_writer.writerow([])
        line = ['Time Step', 'Equilibrated?', 'Spawners', 'Harvest']
        csv_writer.writerow(line)

        for i in range(0, len(H_tx)):  # i is a cycle
            line = [i]
            if equilibrate_timestep and i >= equilibrate_timestep:
                line.append('Y')
            else:
                line.append('N')
            if i == 0:
                line.append("(none)")
            elif recruitment_type == 'Fixed':
                line.append("(fixed recruitment)")
            else:
                line.append("%.2f" % Spawners_t[i])
            line.append("%.2f" % H_tx[i].sum())
            csv_writer.writerow(line)


def _create_results_html(vars_dict):
    '''
    Creates an HTML file that contains a summary of all harvest totals
    for each subregion.
    '''
    do_batch = vars_dict['do_batch']
    if do_batch is True:
        basename = os.path.splitext(os.path.basename(
            vars_dict['population_csv_uri']))[0]
        filename = 'results_page_' + basename + '.html'
    elif vars_dict['results_suffix'] is not '':
        filename = 'results_page_' + vars_dict['results_suffix'] + '.html'
    else:
        filename = 'results_page.html'
    uri = os.path.join(vars_dict['output_dir'], filename)

    recruitment_type = vars_dict['recruitment_type']
    Spawners_t = vars_dict['Spawners_t']
    H_tx = vars_dict['H_tx']
    V_tx = vars_dict['V_tx']
    equilibrate_timestep = int(vars_dict['equilibrate_timestep'])
    Regions = vars_dict['Regions']

    # Set Reporting Arguments
    rep_args = {}
    rep_args['title'] = "Fisheries Results Page"
    rep_args['out_uri'] = uri
    rep_args['sortable'] = True  # JS Functionality
    rep_args['totals'] = True  # JS Functionality

    total_timesteps = len(H_tx)

    # Create Model Run Overview Table
    overview_columns = [
        {'name': 'Attribute', 'total': False},
        {'name': 'Value', 'total': False}]

    overview_body = [
        {'Attribute': 'Model Type', 'Value': vars_dict['population_type']},
        {'Attribute': 'Recruitment Type', 'Value':
            vars_dict['recruitment_type']},
        {'Attribute': 'Sex-Specific?', 'Value': (
            'Yes' if vars_dict['sexsp'] == 2 else 'No')},
        {'Attribute': 'Classes', 'Value': str(len(vars_dict['Classes']))},
        {'Attribute': 'Regions', 'Value': str(len(Regions))},
    ]

    # Create Final Cycle Harvest Summary Table
    final_cycle_columns = [
        {'name': 'Subregion', 'total': False},
        {'name': 'Harvest', 'total': True}]

    if vars_dict['val_cont']:
        final_cycle_columns.append(
            {'name': 'Valuation', 'total': True})

    final_timestep_body = []
    for i in range(0, len(H_tx[-1])):  # i is a cycle
        sub_dict = {}
        sub_dict['Subregion'] = Regions[i]
        sub_dict['Harvest'] = "%.2f" % H_tx[-1, i]
        if vars_dict['val_cont']:
            sub_dict['Valuation'] = "%.2f" % V_tx[-1, i]
        final_timestep_body.append(sub_dict)

    # Create Harvest Time Step Table
    timestep_breakdown_columns = [
        {'name': 'Time Step', 'total': False},
        {'name': 'Equilibrated?', 'total': False},
        {'name': 'Spawners', 'total': True},
        {'name': 'Harvest', 'total': True}]

    timestep_breakdown_body = []
    for i in range(0, total_timesteps):
        sub_dict = {}
        sub_dict['Time Step'] = str(i)
        if i == 0:
            sub_dict['Spawners'] = "(none)"
        elif recruitment_type == 'Fixed':
            sub_dict['Spawners'] = "(fixed recruitment)"
        else:
            sub_dict['Spawners'] = "%.2f" % Spawners_t[i]
        sub_dict['Harvest'] = "%.2f" % H_tx[i].sum()
        # This can be more rigorously checked
        if equilibrate_timestep and i >= equilibrate_timestep:
            sub_dict['Equilibrated?'] = 'Y'
        else:
            sub_dict['Equilibrated?'] = 'N'
        timestep_breakdown_body.append(sub_dict)

    # Generate Report
    css = """body { background-color: #EFECCA; color: #002F2F; }
         h1 { text-align: center }
         h1, h2, h3, h4, strong, th { color: #046380 }
         h2 { border-bottom: 1px solid #A7A37E }
         table { border: 5px solid #A7A37E; margin-bottom: 50px; \
         background-color: #E6E2AF; }
         table.sortable thead:hover { border: 5px solid #A7A37E; \
         margin-bottom: 50px; background-color: #E6E2AF; }
         td, th { margin-left: 0px; margin-right: 0px; padding-left: \
         8px; padding-right: 8px; padding-bottom: 2px; padding-top: \
         2px; text-align: left; }
         td { border-top: 5px solid #EFECCA }
         img { margin: 20px }"""

    elements = [
        {
            'type': 'text',
            'section': 'body',
            'text': '<h2>Model Run Overview</h2>'
        },
        {
            'type': 'table',
            'section': 'body',
            'sortable': True,
            'checkbox': False,
            'total': False,
            'data_type': 'dictionary',
            'columns': overview_columns,
            'data': overview_body
        },
        {
            'type': 'text',
            'section': 'body',
            'text': '<h2>Final Harvest by Subregion After ' +
                    str(total_timesteps-1) + ' Time Steps</h2>'},
        {
            'type': 'table',
            'section': 'body',
            'sortable': True,
            'checkbox': False,
            'total': True,
            'data_type': 'dictionary',
            'columns': final_cycle_columns,
            'data': final_timestep_body
        },
        {
            'type': 'text',
            'section': 'body',
            'text': '<h2>Time Step Breakdown</h2>'
        },
        {
            'type': 'table',
            'section': 'body',
            'sortable': True,
            'checkbox': False,
            'total': False,
            'data_type': 'dictionary',
            'columns': timestep_breakdown_columns,
            'data': timestep_breakdown_body
        },
        {
            'type': 'head',
            'section': 'head',
            'format': 'style',
            'data_src': css,
            'input_type': 'Text'
        }]

    equilibrium_warning = [{
        'type': 'text',
        'section': 'body',
        'text': '<h2 style="color:red">Warning: Population Did Not Reach Equilibrium State</h2>'
    }]

    if not bool(vars_dict['equilibrate_timestep']):
        elements = equilibrium_warning + elements

    rep_args['elements'] = elements

    reporting.generate_report(rep_args)


def _create_results_aoi(vars_dict):
    '''
    Appends the final harvest and valuation values for each region to an
    input shapefile.  The 'Name' attributes (case-sensitive) of each region
    in the input shapefile must exactly match the names of each region in the
    population parameters file.

    '''
    aoi_uri = vars_dict['aoi_uri']
    Regions = vars_dict['Regions']
    H_tx = vars_dict['H_tx']
    V_tx = vars_dict['V_tx']
    basename = os.path.splitext(os.path.basename(aoi_uri))[0]
    do_batch = vars_dict['do_batch']
    if do_batch is True:
        basename2 = os.path.splitext(os.path.basename(
            vars_dict['population_csv_uri']))[0]
        filename = basename + '_results_aoi_' + basename2 + '.shp'
    elif vars_dict['results_suffix'] is not '':
        filename = basename + '_results_aoi_' + vars_dict[
            'results_suffix'] + '.shp'
    else:
        filename = basename + '_results_aoi.shp'

    output_aoi_uri = os.path.join(vars_dict['output_dir'], filename)
    LOGGER.info('Copying AOI %s to %s', aoi_uri, output_aoi_uri)

    # Copy AOI file to outputs directory
    pygeoprocessing.geoprocessing.copy_datasource_uri(aoi_uri, output_aoi_uri)

    # Append attributes to Shapefile
    ds = ogr.Open(output_aoi_uri, update=1)
    layer = ds.GetLayer()

    # Set Harvest
    harvest_field = ogr.FieldDefn('Hrv_Total', ogr.OFTReal)
    layer.CreateField(harvest_field)

    harv_reg_dict = {}
    for i in range(0, len(Regions)):
        harv_reg_dict[Regions[i]] = H_tx[-1][i]

    # Set Valuation
    if vars_dict['val_cont']:
        val_field = ogr.FieldDefn('Val_Total', ogr.OFTReal)
        layer.CreateField(val_field)

    val_reg_dict = {}
    for i in range(0, len(Regions)):
        val_reg_dict[Regions[i]] = V_tx[-1][i]

    # Add Information to Shapefile
    for feature in layer:
        region_name = str(feature.items()['Name'])
        feature.SetField('Hrv_Total', "%.2f" % harv_reg_dict[region_name])
        if vars_dict['val_cont']:
            feature.SetField('Val_Total', "%.2f" % val_reg_dict[region_name])
        layer.SetFeature(feature)

    layer.ResetReading()
