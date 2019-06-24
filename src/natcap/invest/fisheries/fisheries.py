"""
The Fisheries module contains the high-level code for executing the fisheries
model
"""
from __future__ import absolute_import
import logging
import csv
import os

from osgeo import gdal

from . import fisheries_io as io
from . import fisheries_model as model
from .. import utils
from .. import validation

LOGGER = logging.getLogger('natcap.invest.fisheries.fisheries')
LABEL = 'Fisheries'


def execute(args, create_outputs=True):
    """Fisheries.

    args['workspace_dir'] (str): location into which all intermediate
        and output files should be placed.

    args['results_suffix'] (str): a string to append to output filenames

    args['aoi_vector_path'] (str): location of shapefile which will be
        used as subregions for calculation. Each region must contain a 'Name'
        attribute (case-sensitive) matching the given name in the population
        parameters csv file.

    args['timesteps'] (int): represents the number of time steps that
        the user desires the model to run.

    args['population_type'] (str): specifies whether the model
        is age-specific or stage-specific. Options will be either "Age
        Specific" or "Stage Specific" and will change which equation is
        used in modeling growth.

    args['sexsp'] (str): specifies whether or not the age and stage
        classes are distinguished by sex.

    args['harvest_units'] (str): specifies how the user wants to get
        the harvest data. Options are either "Individuals" or "Weight", and
        will change the harvest equation used in core. (Required if
        args['val_cont'] is True)

    args['do_batch'] (bool): specifies whether program will perform a
        single model run or a batch (set) of model runs.

    args['population_csv_path'] (str): location of the population
        parameters csv. This will contain all age and stage specific
        parameters. (Required if args['do_batch'] is False)

    args['population_csv_dir'] (str): location of the directory that
        contains the Population Parameters CSV files for batch processing
        (Required if args['do_batch'] is True)

    args['spawn_units'] (str): specifies whether the spawner abundance used in
        the recruitment function should be calculated in terms of number of
        individuals ('Individuals') or in terms of biomass ('Weight'). If
        'Weight' is selected, the user must provide a 'Weight' vector alongside
        the survival matrix in the Population Parameters CSV File. The 'alpha'
        and 'beta' parameters provided by the user should correspond to the
        selected choice.

    args['total_init_recruits'] (float): represents the initial number of
        recruits that will be used in calculation of population on a per area
        basis.

    args['recruitment_type'] (str): name corresponding to one of the built-in
        recruitment functions {'Beverton-Holt', 'Ricker', 'Fecundity', Fixed},
        or 'Other', meaning that the user is passing in their own recruitment
        function as an anonymous python function via the optional dictionary
        argument 'recruitment_func'.

    args['recruitment_func'] (function): Required if args['recruitment_type']
        is set to 'Other'.  See below for instructions on how to create a user-
        defined recruitment function.

    args['alpha'] (float): must exist within args for BH or Ricker Recruitment.
        Parameter that will be used in calculation of recruitment.

    args['beta'] (float): must exist within args for BH or Ricker Recruitment.
        Parameter that will be used in calculation of recruitment.

    args['total_recur_recruits'] (float): must exist within args for Fixed
        Recruitment. Parameter that will be used in calculation of recruitment.

    args['migr_cont'] (bool): if True, model uses migration.

    args['migration_dir'] (str): if this parameter exists, it means migration
        is desired. This is  the location of the parameters folder containing
        files for migration. There should be one file for every age class which
        migrates. (Required if args['migr_cont'] is True)

    args['val_cont'] (bool): if True, model computes valuation.

    args['frac_post_process'] (float): represents the fraction of the species
        remaining after processing of the whole carcass is complete. This will
        exist only if valuation is desired for the particular species.
        (Required if args['val_cont'] is True)

    args['unit_price'] (float): represents the price for a single unit of
        harvest. Exists only if valuation is desired. (Required if
        args['val_cont'] is True)

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'results_suffix': 'scenario_name',
            'aoi_vector_path': 'path/to/aoi_vector_path',
            'total_timesteps': 100,
            'population_type': 'Stage-Based',
            'sexsp': 'Yes',
            'harvest_units': 'Individuals',
            'do_batch': False,
            'population_csv_path': 'path/to/csv_path',
            'population_csv_dir': '',
            'spawn_units': 'Weight',
            'total_init_recruits': 100000.0,
            'recruitment_type': 'Ricker',
            'alpha': 32.4,
            'beta': 54.2,
            'total_recur_recruits': 92.1,
            'migr_cont': True,
            'migration_dir': 'path/to/mig_dir/',
            'val_cont': True,
            'frac_post_process': 0.5,
            'unit_price': 5.0,
        }

    **Creating a User-Defined Recruitment Function**

    An optional argument has been created in the Fisheries Model to allow users
    proficient in Python to pass their own recruitment function into the
    program via the args dictionary.

    Using the Beverton-Holt recruitment function as an example, here's how a
    user might create and pass in their own recruitment function::

        import natcap.invest
        import numpy as np

        # define input data
        Matu = np.array([...])  # the Maturity vector in the Population
            Parameters File
        Weight = np.array([...])  # the Weight vector in the Population
            Parameters File
        LarvDisp = np.array([...])  # the LarvalDispersal vector in the
            Population Parameters File
        alpha = 2.0  # scalar value
        beta = 10.0  # scalar value
        sexsp = 2   # 1 = not sex-specific, 2 = sex-specific

        # create recruitment function
        def spawners(N_prev):
            return (N_prev * Matu * Weight).sum()

        def rec_func_BH(N_prev):
            N_0 = (LarvDisp * ((alpha * spawners(
                N_prev) / (beta + spawners(N_prev)))) / sexsp)
            return (N_0, spawners(N_prev))

        # fill out args dictionary
        args = {}
        # ... define other arguments ...
        args['recruitment_type'] = 'Other'  # lets program know to use user-
            defined function
        args['recruitment_func'] = rec_func_BH  # pass recruitment function as
            'anonymous' Python function

        # run model
        natcap.invest.fisheries.fisheries.execute(args)

    Conditions that a new recruitment function must meet to run properly:

    + **The function must accept as an argument:** a single numpy three-
        dimensional array (N_prev) representing the state of the population at
        the previous time step. N_prev has three dimensions: the indices of the
        first dimension correspond to the region (must be in same order as
        provided in the Population Parameters File), the indices of the second
        dimension represent the sex if it is specific (i.e. two indices
        representing female, then male if the model is 'sex-specific', else
        just a single zero index representing the female and male populations
        aggregated together), and the indicies of the third dimension represent
        age/stage in ascending order.

    + **The function must return:** a tuple of two values. The first value
        (N_0) being a single numpy one-dimensional array representing the
        youngest age of the population for the next time step. The indices of
        the array correspond to the regions of the population (outputted in
        same order as provided). If the model is sex-specific, it is currently
        assumed that males and females are produced in equal number and that
        the returned array has been already been divided by 2 in the
        recruitment function. The second value (spawners) is the number or
        weight of the spawners created by the population from the previous time
        step, provided as a scalar float value (non-negative).

    Example of How Recruitment Function Operates within Fisheries Model::

        # input data
        N_prev_xsa = [[[region0-female-age0, region0-female-age1],
                       [region0-male-age0, region1-male-age1]],
                      [[region1-female-age0, region1-female-age1],
                       [region1-male-age0], [region1-male-age1]]]

        # execute function
        N_0_x, spawners = rec_func(N_prev_xsa)

        # output data - where N_0 contains information about the youngest
        #     age/stage of the population for the next time step:
        N_0_x = [region0-age0, region1-age0] # if sex-specific, rec_func should
            divide by two before returning type(spawners) is float

    """
    args = args.copy()

    # Parse Inputs
    model_list = io.fetch_args(args, create_outputs=create_outputs)

    # For Each Model in Set:
    vars_all_models = []
    for model_args_dict in model_list:

        # Setup Model
        model_vars_dict = model.initialize_vars(model_args_dict)

        recru_func = model.set_recru_func(model_vars_dict)
        init_cond_func = model.set_init_cond_func(model_vars_dict)
        cycle_func = model.set_cycle_func(model_vars_dict, recru_func)
        harvest_func = model.set_harvest_func(model_vars_dict)

        # Run Model
        model_vars_dict = model.run_population_model(
            model_vars_dict, init_cond_func, cycle_func, harvest_func)

        vars_all_models.append(model_vars_dict)

        if create_outputs:
            # Create Model Outputs
            io.create_outputs(model_vars_dict)

    LOGGER.warning(vars_all_models[0]['results_suffix'])
    return vars_all_models


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Fisheries.

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

    missing_keys = set([])
    keys_with_empty_values = set([])
    required_key_list = [
        ('workspace_dir', True),
        ('results_suffix', False),
        ('aoi_vector_path', False),
        ('total_timesteps', True),
        ('population_type', True),
        ('sexsp', True),
        ('harvest_units', True),
        ('total_init_recruits', True),
        ('recruitment_type', True),
        ('spawn_units', True),
        ('alpha', False),
        ('beta', False),
    ]

    if 'do_batch' in args:
        if bool(args['do_batch']):
            # If we're doing batch processing, require the batch-processing
            # directory.
            required_key_list.append(('population_csv_dir', True))
        else:
            # If we're not doing batch processing, just require the one CSV.
            required_key_list.append(('population_csv_path', True))

    if 'val_cont' in args and bool(args['val_cont']):
        required_key_list += [
            ('frac_post_process', True),
            ('unit_price', True),
        ]

    if 'migr_cont' in args and bool(args['migr_cont']):
        required_key_list += [
            ('migration_dir', True),
        ]

    if 'recruitment_type' in args and args['recruitment_type'] == 'Fixed':
        required_key_list += [
            ('total_recur_recruits', True),
        ]

    for key, required in required_key_list:
        if limit_to in (None, key):
            try:
                if args[key] in ('', None) and required:
                    keys_with_empty_values.add(key)
            except KeyError:
                if required:
                    missing_keys.add(key)

    if len(missing_keys) > 0:
        raise KeyError(
            'Args is missing required keys: %s' % ', '.join(
                sorted(missing_keys)))

    if len(keys_with_empty_values) > 0:
        warnings.append((keys_with_empty_values,
                         'Argument must have a value.'))

    if (limit_to in ('aoi_vector_path', None) and
            'aoi_vector_path' in args and args['aoi_vector_path'] != ''):
        with utils.capture_gdal_logging():
            dataset = gdal.OpenEx(args['aoi_vector_path'], gdal.OF_VECTOR)
        if dataset is None:
            warnings.append(
                (['aoi_vector_path'],
                 'AOI vector must be an OGR-compatible vector.'))
        else:
            layer = dataset.GetLayer()
            column_names = [defn.GetName() for defn in layer.schema]
            if 'Name' not in column_names:
                warnings.append(
                    (['aoi_vector_path'],
                     'Case-sensitive column name "Name" is missing'))

    if limit_to in ('do_batch', None):
        if args['do_batch'] not in (True, False):
            warnings.append((['do_batch'],
                             'Parameter must be either True or False"'))

    if limit_to in ('population_csv_path', None):
        # Only validate the CSV if it's provided.
        # Either the CSV or the batch-processing dir must be valid.
        if ('population_csv_path' in args and
                args['population_csv_path'] not in ('', None)):
            try:
                csv.reader(open(args['population_csv_path'], 'r'))
            except (csv.Error, IOError):
                warnings.append((['population_csv_path'],
                                 'Parameter must be a valid CSV file.'))

    for directory_key in ('population_csv_dir', 'migration_dir'):
        try:
            if all((limit_to in (directory_key, None),
                    args['directory_key'] != '',
                    not os.path.isdir(args[directory_key]))):
                warnings.append(([directory_key],
                                'Directory could not be found.'))
        except KeyError:
            # These parameters are not necessarily required, and may not be in
            # args.
            pass

    for float_key, max_value in (('total_init_recruits', None),
                                 ('alpha', None),
                                 ('beta', None),
                                 ('total_timesteps', None),
                                 ('total_recur_recruits', None),
                                 ('unit_price', None),
                                 ('frac_post_process', 1.0)):
        if limit_to in (float_key, None) and (float_key, True) in required_key_list:
            try:
                if float(args[float_key]) < 0:
                    warnings.append(([float_key],
                                     'Value must be positive'))

                if (max_value is not None and
                        float(args[float_key]) > max_value):
                    warnings.append(
                        ([float_key],
                         'Value cannot be greater than %s' % max_value))
            except ValueError, TypeError:
                warnings.append(([float_key],
                                 'Value must be a number.'))
            except KeyError:
                # Parameter is not necessarily required.
                pass

    for options_key, options in (
            ('recruitment_type', ('Beverton-Holt', 'Ricker', 'Fecundity',
                                  'Fixed', 'Other')),
            ('harvest_units', ('Individuals', 'Weight')),
            ('spawn_units', ('Individuals', 'Weight')),
            ('sexsp', ('Yes', 'No')),
            ('population_type', ("Age-Based", "Stage-Based"))):
        if (limit_to in (options_key, None) and
                args[options_key] not in options):
            warnings.append(
                ([options_key],
                    'Parameter must be one of %s' % ', '.join(options)))

    return warnings
