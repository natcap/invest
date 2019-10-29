"""
The Fisheries Habitat Scenarios Tool IO module contains functions for handling
inputs and outputs
"""

import logging
import os
import csv
import copy
import io

import numpy as np

from .. import utils

LOGGER = logging.getLogger('natcap.invest.fisheries.hst_io')


# Fetch and Verify Arguments
def fetch_args(args):
    """
    Fetches input arguments from the user, verifies for correctness and
    completeness, and returns a list of variables dictionaries

    Args:
        args (dictionary): arguments from the user (same as Fisheries
            Preprocessor entry point)

    Returns:
        vars_dict (dictionary): dictionary containing necessary variables

    Raises:
        ValueError: parameter mismatch between Population and Habitat CSV files

    Example Returns::

        vars_dict = {
            'workspace_dir': 'path/to/workspace_dir/',
            'output_dir': 'path/to/output_dir/',
            'sexsp': 2,
            'gamma': 0.5,

            # Pop Vars
            'population_csv_path': 'path/to/csv_path',
            'Surv_nat_xsa': np.array(
                [[[...], [...]], [[...], [...]], ...]),
            'Classes': np.array([...]),
            'Class_vectors': {
                'Vulnfishing': np.array([...], [...]),
                'Maturity': np.array([...], [...]),
                'Duration': np.array([...], [...]),
                'Weight': np.array([...], [...]),
                'Fecundity': np.array([...], [...]),
            },
            'Regions': np.array([...]),
            'Region_vectors': {
                'Exploitationfraction': np.array([...]),
                'Larvaldispersal': np.array([...]),
            },

            # Habitat Vars
            'habitat_chg_csv_path': 'path/to/csv',
            'habitat_dep_csv_path': 'path/to/csv',
            'Habitats': ['habitat1', 'habitat2', ...],
            'Hab_classes': ['class1', 'class2', ...],
            'Hab_regions': ['region1', 'region2', ...],
            'Hab_chg_hx': np.array(
                [[[...], [...]], [[...], [...]], ...]),
            'Hab_dep_ha': np.array(
                [[[...], [...]], [[...], [...]], ...]),
            'Hab_class_mvmt_a': np.array([...]),
            'Hab_dep_num_a': np.array([...]),
        }

    """
    args = args.copy()
    sexsp_dict = {
        'no': 1,
        'yes': 2,
    }
    args['sexsp'] = sexsp_dict[args['sexsp'].lower()]
    args['gamma'] = float(args['gamma'])

    # Fetch Data
    args['output_dir'] = os.path.join(args['workspace_dir'], 'output')
    utils.make_directories([args['workspace_dir'], args['output_dir']])
    pop_dict = read_population_csv(args)
    habitat_chg_dict = read_habitat_chg_csv(args)
    habitat_dep_dict = read_habitat_dep_csv(args)

    # Check that habitat names match between habitat parameter files
    assert habitat_dep_dict['Habitats'] == habitat_chg_dict['Habitats'], (
        "Mismatch between Habitat names in Habitat Paramater CSV files.")

    del habitat_dep_dict['Habitats']
    habitat_dict = dict(list(habitat_chg_dict.items()) +
                        list(habitat_dep_dict.items()))

    # Check that classes and regions match
    P_Classes = [x.lower() for x in pop_dict['Classes']]
    H_Classes = [x.lower() for x in habitat_dict['Hab_classes']]
    assert P_Classes == H_Classes, (
        "Mismatch between class names in Population and Habitat CSV files "
        "(%s vs %s)") % (P_Classes, H_Classes)

    P_Regions = [x.lower() for x in pop_dict['Regions']]
    H_Regions = [x.lower() for x in habitat_dict['Hab_regions']]
    assert P_Regions == H_Regions, (
        "Mismatch between region names in Population and Habitat CSV files")

    # Combine Data
    vars_dict = dict(list(args.items()) + list(pop_dict.items()) +
                     list(habitat_dict.items()))

    return vars_dict


def read_population_csv(args):
    """
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

    Returns:
        pop_dict (dictionary): dictionary containing verified population
            arguments

    Raises:
        MissingParameter: required parameter not included
        ValueError: values are out of bounds or of wrong type

    Example Returns::

        pop_dict = {
            'population_csv_path': 'path/to/csv',
            'Surv_nat_xsa': np.array(
                [[...], [...]], [[...], [...]], ...),

            # Class Vectors
            'Classes': np.array([...]),
            'Class_vector_names': [...],
            'Class_vectors': {
                'Vulnfishing': np.array([...], [...]),
                'Maturity': np.array([...], [...]),
                'Duration': np.array([...], [...]),
                'Weight': np.array([...], [...]),
                'Fecundity': np.array([...], [...]),
            },

            # Region Vectors
            'Regions': np.array([...]),
            'Region_vector_names': [...],
            'Region_vectors': {
                'Exploitationfraction': np.array([...]),
                'Larvaldispersal': np.array([...]),
            },
        }
    """
    path = args['population_csv_path']
    pop_dict = _parse_population_csv(path, args['sexsp'])

    # Check that required information exists
    assert 'Surv_nat_xsa' in pop_dict, (
        "Population Parameters File does not contain a Survival Matrix")

    Necessary_Params = ['Classes', 'Regions', 'Surv_nat_xsa']
    Matching_Params = [i for i in pop_dict.keys() if i in Necessary_Params]

    assert len(Matching_Params) == len(Necessary_Params), (
        "Population Parameters File does not contain all necessary "
        "parameters. %s") % path

    # Checks that all Survival Matrix elements exist between [0, 1]
    A = pop_dict['Surv_nat_xsa']
    assert all((A.min() >= 0.0, A.max() <= 1.0)), (
        "Surivial Matrix contains values outside [0, 1]")

    return pop_dict


def _parse_population_csv(path, sexsp):
    """
    Parses the given Population Parameters CSV file and returns a dictionary
    of lists, arrays, and matrices

    Dictionary items containing lists, arrays or matrices are capitalized,
    while single variables are lowercase.

    Keys: Surv_nat_xsa, Vulnfishing, Maturity, Duration, Weight, Fecundity,
            Exploitationfraction, Larvaldispersal, Classes, Regions

    Returns:
        pop_dict (dictionary): verified population arguments

    Example Returns::

        pop_dict = {
            'Surv_nat_xsa': np.array(
                [[...], [...]], [[...], [...]], ...),

            # Class Vectors
            'Classes': np.array([...]),
            'Class_vector_names': [...],
            'Class_vectors': {
                'Vulnfishing': np.array([...], [...]),
                'Maturity': np.array([...], [...]),
                'Duration': np.array([...], [...]),
                'Weight': np.array([...], [...]),
                'Fecundity': np.array([...], [...]),
            }

            # Region Vectors
            'Regions': np.array([...]),
            'Region_vector_names': [...],
            'Region_vectors': {
                'Exploitationfraction': np.array([...]),
                'Larvaldispersal': np.array([...]),
            }
        }
    """
    assert sexsp in (1, 2), 'Sexsp value %s unknown' % sexsp
    csv_data = []
    pop_dict = {}

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_data.append(line)

    start_rows = _get_table_row_start_indexes(csv_data)
    start_cols = _get_table_col_start_indexes(csv_data, start_rows[0])
    end_rows = _get_table_row_end_indexes(csv_data)
    end_cols = _get_table_col_end_indexes(csv_data, start_rows[0])

    classes = _get_col(
        csv_data, start_rows[0])[0:end_rows[0]+1]

    pop_dict["Classes"] = classes[1:]
    if sexsp == 2:
        pop_dict["Classes"] = pop_dict["Classes"][0:len(pop_dict["Classes"])//2]

    regions = _get_row(csv_data, start_cols[0])[0: end_cols[0]+1]

    pop_dict["Regions"] = regions[1:]

    surv_table = _get_table(csv_data, start_rows[0]+1, start_cols[0]+1)
    class_attributes_table = _get_table(
        csv_data, start_rows[0], start_cols[1])
    region_attributes_table = _get_table(
        csv_data, start_rows[1], start_cols[0])

    if sexsp == 1:
        # Sex Neutral
        pop_dict['Surv_nat_xsa'] = np.array(
            [surv_table], dtype=np.float_).swapaxes(1, 2).swapaxes(0, 1)

    elif sexsp == 2:
        # Sex Specific
        female = np.array(surv_table[0:len(surv_table)//sexsp], dtype=np.float_)
        male = np.array(surv_table[len(surv_table)//sexsp:], dtype=np.float_)
        pop_dict['Surv_nat_xsa'] = np.array(
            [female, male]).swapaxes(1, 2).swapaxes(0, 1)

    Class_vector_names = class_attributes_table[0]
    for i in range(0, len(Class_vector_names)):
        Class_vector_names[i] = Class_vector_names[i].capitalize()
    pop_dict['Class_vector_names'] = Class_vector_names
    pop_dict['Class_vectors'] = {}
    for col in range(0, len(class_attributes_table[0])):
        pop_dict['Class_vectors'].update(_vectorize_attribute(
            _get_col(class_attributes_table, col), sexsp))

    Region_vector_names = []
    for attribute in region_attributes_table:
        Region_vector_names.append(attribute[0].capitalize())
    pop_dict['Region_vector_names'] = Region_vector_names
    pop_dict['Region_vectors'] = {}
    for row in range(0, len(region_attributes_table)):
        pop_dict['Region_vectors'].update(_vectorize_reg_attribute(
            _get_row(region_attributes_table, row)))

    return pop_dict


def read_habitat_dep_csv(args):
    """
    Parses and verifies a Habitat Dependency Parameters CSV file and returns a
    dictionary of information related to the interaction between a species and
    the given habitats.

    Parses the Habitat Parameters CSV file for the following vectors:

        + Names of Habitats and Classes
        + Habitat-Class Dependency

    The following vectors are derived from the information given in the file:

        + Classes where movement between habitats occurs
        + Number of habitats that a particular class depends upon

    Args:
        args (dictionary): arguments from the user (same as Fisheries
            HST entry point)

    Returns:
        habitat_dep_dict (dictionary): dictionary containing necessary
            variables

    Raises:
        MissingParameter - required parameter not included
        ValueError - values are out of bounds or of wrong type
        IndexError - likely a file formatting issue

    Example Returns::

        habitat_dep_dict = {
            'Habitats': ['habitat1', 'habitat2', ...],
            'Hab_classes': ['class1', 'class2', ...],
            'Hab_dep_ha': np.array(
                [[[...], [...]], [[...], [...]], ...]),
            'Hab_class_mvmt_a': np.array([...]),
            'Hab_dep_num_a': np.array([...]),
        }
    """
    habitat_dep_dict = _parse_habitat_dep_csv(args)

    # Verify provided information
    A = habitat_dep_dict['Hab_dep_ha']
    assert (A.min() >= 0.0 and A.max() <= 1.0), (
        "At least one element of the Habitat Dependency by Class vectors is "
        "out of bounds. Values must be between [0, 1] inclusive.")

    # Derive additional information
    # TRANSITION BITMAP NEEDS CLEARER DEFINITION
    A_ah = copy.copy(habitat_dep_dict['Hab_dep_ha'].swapaxes(0, 1))
    A_ah[A_ah != 0] = 1
    Hab_class_mvmt_a = [0]
    for i in range(1, len(A_ah)):
        if np.all(A_ah[i] == A_ah[i-1]):
            Hab_class_mvmt_a.append(0)
        else:
            Hab_class_mvmt_a.append(1)
    habitat_dep_dict['Hab_class_mvmt_a'] = np.array(Hab_class_mvmt_a)

    Hab_dep_num_a = []
    for A_h in A_ah:
        Hab_dep_num_a.append(int(len(np.nonzero(A_h)[0])))
    habitat_dep_dict['Hab_dep_num_a'] = np.array(Hab_dep_num_a)

    return habitat_dep_dict


def _parse_habitat_dep_csv(args):
    """
    Parses the Habitat Dependency Parameters CSV file for the following vectors
        + Names of Habitats and Classes
        + Habitat-Class Dependency

    Args:
        args (dictionary): arguments from the user (same as Fisheries
            HST entry point)

    Returns:
        habitat_dep_dict (dictionary): dictionary containing necessary
            variables

    Raises:
        MissingParameter: required parameter not included
        IndexError: likely a file formatting issue

    Example Returns::

        habitat_dep_dict = {
            'Habitats': ['habitat1', 'habitat2', ...],
            'Hab_classes': ['class1', 'class2', ...],
            'Hab_dep_ha': np.array(
                [[[...], [...]], [[...], [...]], ...]),
            'Hab_class_mvmt_a': np.array([...]),
            'Hab_dep_num_a': np.array([...]),
        }
    """
    path = args['habitat_dep_csv_path']
    csv_data = []
    habitat_dep_dict = {}

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_data.append(line)

    # Find Boundary Information
    start_rows = _get_table_row_start_indexes(csv_data)
    start_cols = _get_table_col_start_indexes(csv_data, start_rows[0])
    end_rows = _get_table_row_end_indexes(csv_data)
    end_cols = _get_table_col_end_indexes(csv_data, start_rows[0])

    # Start Parsing Data
    Habitats = _get_col(
        csv_data, start_rows[0])[start_rows[0]+1:end_rows[0]+1]
    Hab_classes = _get_row(
        csv_data, start_cols[0])[start_cols[0]+1:end_cols[0]+1]
    Hab_dep_ha = _get_table(
        csv_data, start_rows[0]+1, start_cols[0]+1)

    # Standardize capitalization
    Habitats = [x.capitalize() for x in Habitats]
    Hab_classes = [x.capitalize() for x in Hab_classes]

    # Reformat and add data to dictionary
    habitat_dep_dict['Habitats'] = Habitats
    habitat_dep_dict['Hab_classes'] = Hab_classes
    habitat_dep_dict['Hab_dep_ha'] = np.array(Hab_dep_ha, dtype=float)

    return habitat_dep_dict


def read_habitat_chg_csv(args):
    """
    Parses and verifies a Habitat Change Parameters CSV file and returns a
    dictionary of information related to the interaction between a species
    and the given habitats.

    Parses the Habitat Change Parameters CSV file for the following vectors:

        + Names of Habitats and Regions
        + Habitat Area Change

    Args:
        args (dictionary): arguments from the user (same as Fisheries
            HST entry point)

    Returns:
        habitat_chg_dict (dictionary): dictionary containing necessary
            variables

    Raises:
        MissingParameter: required parameter not included
        ValueError: values are out of bounds or of wrong type
        IndexError: likely a file formatting issue

    Example Returns::

        habitat_chg_dict = {
            'Habitats': ['habitat1', 'habitat2', ...],
            'Hab_regions': ['region1', 'region2', ...],
            'Hab_chg_hx': np.array(
                [[[...], [...]], [[...], [...]], ...]),
        }
    """
    habitat_chg_dict = _parse_habitat_chg_csv(args)

    # Verify provided information
    A = habitat_chg_dict['Hab_chg_hx']
    assert A.min() >= -1.0, (
        "At least one element of the Habitat Area Change vectors is out of "
        "bounds. Values must be between [-1.0, +inf).")

    return habitat_chg_dict


def _parse_habitat_chg_csv(args):
    """
    Parses the Habitat Change Parameters CSV file for the following vectors
        + Names of Habitats and Regions
        + Habitat Area Change

    Args:
        args (dictionary): arguments from the user (same as Fisheries
            HST entry point)

    Returns:
        habitat_chg_dict (dictionary): dictionary containing necessary
            variables

    Raises:
        MissingParameter: required parameter not included
        IndexError: likely a file formatting issue

    Example Returns::

        habitat_chg_dict = {
            'Habitats': ['habitat1', 'habitat2', ...],
            'Hab_regions': ['region1', 'region2', ...],
            'Hab_chg_hx': np.array(
                [[[...], [...]], [[...], [...]], ...]),
        }
    """
    path = args['habitat_chg_csv_path']
    csv_data = []
    habitat_chg_dict = {}

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_data.append(line)

    # Find Boundary Information
    start_rows = _get_table_row_start_indexes(csv_data)
    start_cols = _get_table_col_start_indexes(csv_data, start_rows[0])
    end_rows = _get_table_row_end_indexes(csv_data)
    end_cols = _get_table_col_end_indexes(csv_data, start_rows[0])

    # Start Parsing Data
    Habitats = _get_col(
        csv_data, start_rows[0])[start_rows[0]+1:end_rows[0]+1]
    Hab_regions = _get_row(
        csv_data, start_cols[0])[start_cols[0]+1:end_cols[0]+1]
    Hab_chg_hx = _get_table(
        csv_data, start_rows[0]+1, start_cols[0]+1)

    # Standardize capitalization
    Habitats = [x.capitalize() for x in Habitats]
    Hab_regions = [x.capitalize() for x in Hab_regions]

    # Reformat and add data to dictionary
    habitat_chg_dict['Habitats'] = Habitats
    habitat_chg_dict['Hab_regions'] = Hab_regions
    habitat_chg_dict['Hab_chg_hx'] = np.array(Hab_chg_hx, dtype=float)

    return habitat_chg_dict


# Helper functions for navigating CSV files
def _get_col(lsts, col):
    l = []
    for row in range(0, len(lsts)):
        l.append(lsts[row][col])
    return l


def _get_row(lsts, row):
    l = []
    for entry in range(0, len(lsts[row])):
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
    a = np.reshape(a, (rows, a.shape[0] // rows))
    d[lst[0].strip().capitalize()] = a
    return d


def _vectorize_reg_attribute(lst):
    d = {}
    a = np.array(lst[1:], dtype=np.float_)
    d[lst[0].strip().capitalize()] = a
    return d


# Generate Outputs
def save_population_csv(vars_dict):
    """
    Creates a new Population Parameters CSV file based the provided inputs.

    Args:
        vars_dict (dictionary): variables generated by preprocessor arguments
            and run.

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'output_dir': 'path/to/output_dir/',
            'sexsp': 2,
            'population_csv_path': 'path/to/csv',  # original csv file
            'Surv_nat_xsa': np.ndarray([...]),
            'Surv_nat_xsa_mod': np.ndarray([...]),

            # Class Vectors
            'Classes': np.array([...]),
            'Class_vector_names': [...],
            'Class_vectors': {
                'Vulnfishing': np.array([...], [...]),
                'Maturity': np.array([...], [...]),
                'Duration': np.array([...], [...]),
                'Weight': np.array([...], [...]),
                'Fecundity': np.array([...], [...]),
            },

            # Region Vectors
            'Regions': np.array([...]),
            'Region_vector_names': [...],
            'Region_vectors': {
                'Exploitationfraction': np.array([...]),
                'Larvaldispersal': np.array([...]),
            },

            # other arguments are ignored ...
        }

    Note:
        + Creates a modified Population Parameters CSV file located in the 'workspace/output/' folder
        + Currently appends '_modified' to original filename for new filename
    """
    Surv_nat_asx_mod = vars_dict['Surv_nat_xsa_mod'].swapaxes(0, 2)
    num_classes = len(vars_dict['Classes'])
    l = []
    # Create header
    l.append('Class')
    for region in vars_dict['Regions']:
        l.append(region)
    l.append('')
    l = [l]

    # Add survival matrix
    for c in range(1, num_classes + 1):
        l.append([])
        l[c].append(vars_dict['Classes'][c-1])
        for i in Surv_nat_asx_mod[c-1][0].tolist():
            l[c].append(i)
        l[c].append('')
    # l[c].append('')
    if vars_dict['sexsp'] == 2:
        for c in range(1, num_classes + 1):
            l.append([])
            l[c + num_classes].append(vars_dict['Classes'][c-1])
            for i in Surv_nat_asx_mod[c-1][1].tolist():
                l[c + num_classes].append(i)
            l[c + num_classes].append('')

    # Add class vectors
    for key in vars_dict['Class_vector_names']:
        l[0].append(key)
        vector = vars_dict['Class_vectors'][key].tolist()
        if vars_dict['sexsp'] == 2:
            vector = vector[0] + vector[1]
        else:
            vector = vector[0]
        i = 1  # skip the first list in l, it's a header
        for v in vector:
            l[i].append(v)
            i += 1

    # Add row of spaces
    l.append([])
    while (len(l[-1]) < len(l[0])):
        l[-1].append('')

    # Add region vectors
    for key in vars_dict['Region_vector_names']:
        l.append([])
        l[-1].append(key)
        vector = vars_dict['Region_vectors'][key].tolist()
        for i in vector:
            l[-1].append(i)
        while (len(l[-1]) < len(l[0])):
            l[-1].append('')

    # Write List to File
    basename, ext = os.path.splitext(os.path.basename(
        vars_dict['population_csv_path']))
    filename = basename + '_modified' + ext
    output_path = os.path.join(vars_dict['output_dir'], filename)
    with open(output_path, 'w') as f:
        wr = csv.writer(f)
        for row in l:
            wr.writerow(row)
