'''
The Fisheries Habitat Scenario Tool module contains the high-level code for
generating a new Population Parameters CSV File based on habitat area
change and the dependencies that particular classes of the given species
have on particular habitats.
'''
import logging
import pprint

import numpy as np

from . import fisheries_hst_io as io

pp = pprint.PrettyPrinter(indent=4)

LOGGER = logging.getLogger('natcap.invest.fisheries.hst')


def execute(args):
    """Fisheries: Habitat Scenario Tool.

    The Fisheries Habitat Scenario Tool generates a new Population Parameters
    CSV File with modified survival attributes across classes and regions
    based on habitat area changes and class-level dependencies on those
    habitats.

    :param str args['workspace_dir']: location into which the resultant
        modified Population Parameters CSV file should be placed.

    :param str args['sexsp']: specifies whether or not the age and stage
        classes are distinguished by sex. Options: 'Yes' or 'No'

    :param str args['population_csv_uri']: location of the population
        parameters csv file. This file contains all age and stage specific
        parameters.

    :param str args['habitat_chg_csv_uri']: location of the habitat change
        parameters csv file. This file contains habitat area change
        information.

    :param str args['habitat_dep_csv_uri']: location of the habitat dependency
        parameters csv file. This file contains habitat-class dependency
        information.

    :param float args['gamma']: describes the relationship between a change
        in habitat area and a change in survival of life stages dependent on
        that habitat

    Returns:
        None

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'sexsp': 'Yes',
            'population_csv_uri': 'path/to/csv',
            'habitat_chg_csv_uri': 'path/to/csv',
            'habitat_dep_csv_uri': 'path/to/csv',
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
