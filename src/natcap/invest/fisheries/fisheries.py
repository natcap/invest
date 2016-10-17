'''
The Fisheries module contains the high-level code for excuting the fisheries
model
'''

import logging

import fisheries_io as io
import fisheries_model as model

LOGGER = logging.getLogger('natcap.invest.fisheries.fisheries')


def execute(args, create_outputs=True):
    """Fisheries.

    :param str args['workspace_dir']: location into which all intermediate
        and output files should be placed.

    :param str args['results_suffix']: a string to append to output filenames

    :param str args['aoi_uri']: location of shapefile which will be used as
        subregions for calculation. Each region must conatin a 'Name'
        attribute (case-sensitive) matching the given name in the population
        parameters csv file.

    :param int args['timesteps']: represents the number of time steps that
        the user desires the model to run.

    :param str args['population_type']: specifies whether the model
        is age-specific or stage-specific. Options will be either "Age
        Specific" or "Stage Specific" and will change which equation is
        used in modeling growth.

    :param str args['sexsp']: specifies whether or not the age and stage
        classes are distinguished by sex.

    :param str args['harvest_units']: specifies how the user wants to get
        the harvest data. Options are either "Individuals" or "Weight", and
        will change the harvest equation used in core. (Required if
        args['val_cont'] is True)

    :param bool args['do_batch']: specifies whether program will perform a
        single model run or a batch (set) of model runs.

    :param str args['population_csv_uri']: location of the population
        parameters csv. This will contain all age and stage specific
        parameters. (Required if args['do_batch'] is False)

    :param str args['population_csv_dir']: location of the directory that
        contains the Population Parameters CSV files for batch processing
        (Required if args['do_batch'] is True)

    :param str args['spawn_units']: (description)

    :param float args['total_init_recruits']: represents the initial number
        of recruits that will be used in calculation of population on a per
        area basis.

    :param str args['recruitment_type']: Name corresponding to one of the
        built-in recruitment functions {'Beverton-Holt', 'Ricker', 'Fecundity',
        Fixed}, or 'Other', meaning that the user is passing in their own
        recruitment function as an anonymous python function via the
        optional dictionary argument 'recruitment_func'.

    :param function args['recruitment_func']: Required if
        args['recruitment_type'] is set to 'Other'.  See below for
        instructions on how to create a user-defined recruitment function.

    :param float args['alpha']: must exist within args for BH or Ricker
        Recruitment. Parameter that will be used in calculation of recruitment.

    :param float args['beta']: must exist within args for BH or Ricker
        Recruitment. Parameter that will be used in calculation of recruitment.

    :param float args['total_recur_recruits']: must exist within args for
        Fixed Recruitment. Parameter that will be used in calculation of
        recruitment.

    :param bool args['migr_cont']: if True, model uses migration

    :param str args['migration_dir']: if this parameter exists, it means
        migration is desired. This is  the location of the parameters
        folder containing files for migration. There should be one file for
        every age class which migrates. (Required if args['migr_cont'] is
        True)

    :param bool args['val_cont']: if True, model computes valuation

    :param float args['frac_post_process']: represents the fraction of the
        species remaining after processing of the whole carcass is
        complete. This will exist only if valuation is desired for the
        particular species. (Required if args['val_cont'] is True)

    :param float args['unit_price']: represents the price for a single unit
        of harvest. Exists only if valuation is desired. (Required if
        args['val_cont'] is True)

    Example Args::

        args = {
            'workspace_dir': 'path/to/workspace_dir/',
            'results_suffix': 'scenario_name',
            'aoi_uri': 'path/to/aoi_uri',
            'total_timesteps': 100,
            'population_type': 'Stage-Based',
            'sexsp': 'Yes',
            'harvest_units': 'Individuals',
            'do_batch': False,
            'population_csv_uri': 'path/to/csv_uri',
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

    An optional argument has been created in the Fisheries Model to allow users proficient in Python to pass their own recruitment function into the program via the args dictionary.

    Using the Beverton-Holt recruitment function as an example, here's how a user might create and pass in their own recruitment function::

        import natcap.invest
        import numpy as np

        # define input data
        Matu = np.array([...])  # the Maturity vector in the Population Parameters File
        Weight = np.array([...])  # the Weight vector in the Population Parameters File
        LarvDisp = np.array([...])  # the LarvalDispersal vector in the Population Parameters File
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
        args['recruitment_type'] = 'Other'  # lets program know to use user-defined function
        args['recruitment_func'] = rec_func_BH  # pass recruitment function as 'anonymous' Python function

        # run model
        natcap.invest.fisheries.fisheries.execute(args)

    Conditions that a new recruitment function must meet to run properly:

    + **The function must accept as an argument:** a single numpy three-dimensional array (N_prev) representing the state of the population at the previous time step. N_prev has three dimensions: the indices of the first dimension correspond to the region (must be in same order as provided in the Population Parameters File), the indices of the second dimension represent the sex if it is specific (i.e. two indices representing female, then male if the model is 'sex-specific', else just a single zero index representing the female and male populations aggregated together), and  the indicies of the third dimension represent age/stage in ascending order.

    + **The function must return:** a tuple of two values. The first value (N_0) being a single numpy one-dimensional array representing the youngest age of the population for the next time step. The indices of the array correspond to the regions of the population (outputted in same order as provided). If the model is sex-specific, it is currently assumed that males and females are produced in equal number and that the returned array has been already been divided by 2 in the recruitment function.   The second value (spawners) is the number or weight of the spawners created by the population from the previous time step, provided as a scalar float value (non-negative).

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
        N_0_x = [region0-age0, region1-age0] # if sex-specific, rec_func should divide by two before returning
        type(spawners) is float

    """

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
