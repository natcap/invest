"""
The Fisheries Habitat Scenario Tool module contains the high-level code for
generating a new Population Parameters CSV File based on habitat area
change and the dependencies that particular classes of the given species
have on particular habitats.
"""
from __future__ import absolute_import
import logging
import csv

import numpy as np

from . import fisheries_hst_io as io
from .. import validation

LOGGER = logging.getLogger('natcap.invest.fisheries.hst')


def execute(args):
    """Fisheries: Habitat Scenario Tool.

    The Fisheries Habitat Scenario Tool generates a new Population Parameters
    CSV File with modified survival attributes across classes and regions
    based on habitat area changes and class-level dependencies on those
    habitats.

    args['workspace_dir'] (str): location into which the resultant
        modified Population Parameters CSV file should be placed.

    args['sexsp'] (str): specifies whether or not the age and stage
        classes are distinguished by sex. Options: 'Yes' or 'No'

    args['population_csv_path'] (str): location of the population
        parameters csv file. This file contains all age and stage specific
        parameters.

    args['habitat_chg_csv_path'] (str): location of the habitat change
        parameters csv file. This file contains habitat area change
        information.

    args['habitat_dep_csv_path'] (str): location of the habitat dependency
        parameters csv file. This file contains habitat-class dependency
        information.

    args['gamma'] (float): describes the relationship between a change
        in habitat area and a change in survival of life stages dependent on
        that habitat

    Returns:
        None

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'sexsp': 'Yes',
            'population_csv_path': 'path/to/csv',
            'habitat_chg_csv_path': 'path/to/csv',
            'habitat_dep_csv_path': 'path/to/csv',
            'gamma': 0.5,
        }

    Note:

        + Modified Population Parameters CSV File saved to 'workspace_dir/output/'
    """

    # Parse, Verify Inputs
    vars_dict = io.fetch_args(args)

    # Convert Data
    vars_dict = convert_survival_matrix(vars_dict)

    # Generate Modified Population Parameters CSV File
    io.save_population_csv(vars_dict)


def convert_survival_matrix(vars_dict):
    """
    Creates a new survival matrix based on the information provided by
    the user related to habitat area changes and class-level dependencies
    on those habitats.

    Args:
        vars_dict (dictionary): see fisheries_preprocessor_io.fetch_args for
            example

    Returns:
        vars_dict (dictionary): modified vars_dict with new Survival matrix
            accessible using the key 'Surv_nat_xsa_mod' with element values
            that exist between [0,1]

    Example Returns::

        ret = {
            # Other Variables...

            'Surv_nat_xsa_mod': np.ndarray([...])
        }
    """
    # Fetch original survival matrix
    S_sxa = vars_dict['Surv_nat_xsa'].swapaxes(0, 1)

    # Fetch conversion parameters
    gamma = vars_dict['gamma']
    H_chg_hx = vars_dict['Hab_chg_hx']      # H_hx
    D_ha = vars_dict['Hab_dep_ha']          # D_ah
    t_a = vars_dict['Hab_class_mvmt_a']     # T_a
    n_a = vars_dict['Hab_dep_num_a']        # n_h
    n_a[n_a == 0] = 0
    num_habitats = len(vars_dict['Habitats'])
    num_classes = len(vars_dict['Classes'])
    num_regions = len(vars_dict['Regions'])

    # Apply function
    Mod_elements_xha = np.ones([num_regions, num_habitats, num_classes])
    A = Mod_elements_xha * D_ha
    A[A != 0] = 1
    Mod_elements_xha = A

    # Create element-wise exponents
    Exp_xha = Mod_elements_xha * D_ha * gamma

    # Swap Axes in Arrays showing modified elements
    Mod_elements_ahx = Mod_elements_xha.swapaxes(0, 2)

    # Absolute percent change in habitat size across all elements
    H_chg_all_ahx = (Mod_elements_ahx * H_chg_hx)
    nonzero_elements = (H_chg_all_ahx != 0)
    H_chg_all_ahx[nonzero_elements] += 1

    # Swap Axes
    H_chg_all_xha = H_chg_all_ahx.swapaxes(0, 2)

    # Apply sensitivity exponent to habitat area change matrix
    H_xha = (H_chg_all_xha ** Exp_xha)
    ones_elements = (H_xha == 1)
    H_xha[ones_elements] = 0

    # Sum across habitats
    H_xa = H_xha.sum(axis=1)

    # Divide by number of habitats and cancel non-class-transition elements
    H_xa_weighted = np.where(n_a == 0, 0, (H_xa * t_a) / n_a)

    # Add unchanged elements back in to matrix
    nan_elements = np.isnan(H_xa_weighted)
    H_xa_weighted[nan_elements] = 1
    zero_elements = (H_xa_weighted == 0)
    H_xa_weighted[zero_elements] = 1
    H_coefficient_xa = H_xa_weighted

    # Multiply coefficients by original Survival matrix
    nan_idx = np.isnan(H_coefficient_xa)
    H_coefficient_xa[nan_idx] = 1
    S_mod_sxa = S_sxa * H_coefficient_xa

    # Filter and correct for elements outside [0, 1]
    S_mod_sxa[S_mod_sxa > 1.0] = 1
    S_mod_sxa[S_mod_sxa < 0.0] = 0

    # Return
    vars_dict['Surv_nat_xsa_mod'] = S_mod_sxa.swapaxes(0, 1)

    return vars_dict


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Fisheries HST.

    Parameters:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    warnings = []
    keys_with_empty_values = set([])
    missing_keys = set([])
    for key in ('workspace_dir',
                'results_suffix',
                'population_csv_path',
                'sexsp',
                'habitat_dep_csv_path',
                'habitat_chg_csv_path',
                'gamma'):
        if key in (None, limit_to):
            try:
                if args[key] in ('', None):
                    keys_with_empty_values.add(key)
            except KeyError:
                missing_keys.add(key)

    if len(missing_keys) > 0:
        raise KeyError(
            'Args is missing required keys: %s' % ', '.join(
                sorted(missing_keys)))

    if len(keys_with_empty_values) > 0:
        warnings.append((keys_with_empty_values,
                         'Argument must have a value'))

    for csv_key in ('population_csv_path',
                    'habitat_dep_csv_path',
                    'habitat_chg_csv_path'):
        if limit_to in (csv_key, None):
            try:
                csv.reader(open(args[csv_key], 'r'))
            except (csv.Error, IOError):
                warnings.append(([csv_key],
                                 'Parameter must be a valid CSV file'))

    if limit_to in ('sexsp', None):
        if args['sexsp'] not in ('Yes', 'No'):
            warnings.append((['sexsp'],
                             'Parameter must be either "Yes" or "No"'))

    if limit_to in ('gamma', None):
        try:
            float(args['gamma'])
        except ValueError:
            warnings.append((['gamma'],
                             'Parameter must be a number'))

    return warnings
