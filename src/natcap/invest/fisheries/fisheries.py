"""Fisheries."""
import logging
import csv
import os

from osgeo import gdal

from . import fisheries_io as io
from . import fisheries_model as model
from .. import utils
from .. import validation

LOGGER = logging.getLogger(__name__)
LABEL = 'Fisheries'

ARGS_SPEC = {
    "model_name": "Fisheries",
    "module": __name__,
    "userguide_html": "fisheries.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "aoi_vector_path": {
            "validation_options": {
                "required_fields": ["NAME"],
            },
            "type": "vector",
            "required": False,
            "about": (
                "A GDAL-supported vector file used to display outputs within "
                "the region(s) of interest. The layer should contain "
                "one feature for every region of interest, each feature of "
                "which should have a 'NAME' attribute.  The 'NAME' "
                "attribute can be numeric or alphabetic, but must be unique "
                "within the given file."),
            "name": "Area of Interest"
        },
        "total_timesteps": {
            "validation_options": {
                "expression": "value > 0",
            },
            "type": "number",
            "required": True,
            "about": (
                "The number of time steps the simulation shall execute "
                "before completion. Must be a positive integer."),
            "name": "Number of Time Steps for Model Run"
        },
        "population_type": {
            "validation_options": {
                "options": ["Age-Based", "Stage-Based"],
            },
            "type": "option_string",
            "required": True,
            "about": (
                "Specifies whether the lifecycle classes provided in the "
                "Population Parameters CSV file represent ages (uniform "
                "duration) or stages. Age-based models (e.g. "
                "Lobster, Dungeness Crab) are separated by uniform, "
                "fixed-length time steps (usually representing a year). "
                "Stage-based models (e.g. White Shrimp) allow "
                "lifecycle-classes to have nonuniform durations based on the "
                "assumed resolution of the provided time step. If the "
                "stage-based model is selected, the Population Parameters "
                "CSV file must include a 'Duration' vector "
                "alongside the survival matrix that contains the number of "
                "time steps that each stage lasts."),
            "name": "Population Model Type"
        },
        "sexsp": {
            "validation_options": {
                "options": ["No", "Yes"]
            },
            "type": "option_string",
            "required": True,
            "about": (
                "Specifies whether or not the lifecycle classes provided in "
                "the Population Parameters CSV file are distinguished by "
                "sex."),
            "name": "Population Classes are Sex-Specific"
        },
        "harvest_units": {
            "validation_options": {
                "options": ["Individuals", "Weight"],
            },
            "type": "option_string",
            "required": True,
            "about": (
                "Specifies whether the harvest output values are calculated "
                "in terms of number of individuals or in terms of biomass "
                "(weight). If 'Weight' is selected, the Population "
                "Parameters CSV file must include a 'Weight' vector "
                "alongside the survival matrix that contains the weight of "
                "each lifecycle class and sex if model is sex-specific."),
            "name": "Harvest by Individuals or Weight"
        },
        "do_batch": {
            "type": "boolean",
            "required": False,
            "about": (
                "Specifies whether program will perform a single model run "
                "or a batch (set) of model runs. For single model "
                "runs, users submit a filepath pointing to a single "
                "Population Parameters CSV file.  For batch model runs, "
                "users submit a directory path pointing to a set of "
                "Population Parameters CSV files."),
            "name": "Batch Processing"
        },
        "population_csv_path": {
            "type": "csv",
            "required": "not do_batch",
            "about": (
                "The provided CSV file should contain all necessary "
                "attributes for the sub-populations based on lifecycle "
                "class, sex, and area - excluding possible migration "
                "information. Please consult the documentation to learn "
                "more about what content should be provided and how the "
                "CSV file should be structured."),
            "name": "Population Parameters File"
        },
        "population_csv_dir": {
            "type": "directory",
            "required": "do_batch",
            "validation_options": {
                "exists": True,
            },
            "about": (
                "The provided CSV folder should contain a set of Population "
                "Parameters CSV files with all necessary attributes for "
                "sub-populations based on lifecycle class, sex, and area - "
                "excluding possible migration information. The name "
                "of each file will serve as the prefix of the outputs "
                "created by the model run. Please consult the "
                "documentation to learn more about what content should be "
                "provided and how the CSV file should be structured."),
            "name": "Population Parameters CSV Folder"
        },
        "spawn_units": {
            "validation_options": {
                "options": ["Weight", "Individuals"],
            },
            "type": "option_string",
            "required": True,
            "about": (
                "Specifies whether the spawner abundance used in the "
                "recruitment function should be calculated in terms of "
                "number of individuals or in terms of biomass (weight). "
                "If 'Weight' is selected, the user must provide a 'Weight' "
                "vector alongside the survival matrix in the Population "
                "Parameters CSV file.  The 'Alpha' and 'Beta' parameters "
                "provided by the user should correspond to the selected "
                "choice. Used only for the Beverton-Holt and Ricker "
                "recruitment functions."),
            "name": "Spawners by Individuals or Weight (Beverton-Holt / Ricker)"
        },
        "total_init_recruits": {
            "validation_options": {
                "expression": "value > 0",
            },
            "type": "number",
            "required": True,
            "about": (
                "The initial number of recruits in the population model at "
                "time equal to zero.<br><br>If the model contains multiple "
                "regions of interest or is distinguished by sex, this value "
                "will be evenly divided and distributed into each "
                "sub-population."),
            "name": "Total Initial Recruits"
        },
        "recruitment_type": {
            "validation_options": {
                "options": ["Beverton-Holt", "Ricker", "Fecundity", "Fixed"],
            },
            "type": "option_string",
            "required": True,
            "about": (
                "The selected equation is used to calculate recruitment into "
                "the subregions at the beginning of each time step.  "
                "Corresponding parameters must be specified with each "
                "function: The Beverton- Holt and Ricker functions both "
                "require arguments for the 'Alpha' and 'Beta' parameters. "
                "The Fecundity function requires a 'Fecundity' vector "
                "alongside the survival matrix in the Population Parameters "
                "CSV file indicating the per-capita offspring for each "
                "lifecycle class. The Fixed function requires an argument "
                "for the 'Total Recruits per Time Step' parameter that "
                "represents a single total recruitment value to be "
                "distributed into the population model at the beginning of "
                "each time step."),
            "name": "Recruitment Function Type"
        },
        "alpha": {
            "type": "number",
            "required": False,
            "about": (
                "Specifies the shape of the stock-recruit curve. Used only "
                "for the Beverton-Holt and Ricker recruitment functions. "
                "Used only for the Beverton-Holt and Ricker recruitment "
                "functions."),
            "name": "Alpha (Beverton-Holt / Ricker)"
        },
        "beta": {
            "type": "number",
            "required": False,
            "about": (
                "Specifies the shape of the stock-recruit curve. Used only "
                "for the Beverton-Holt and Ricker recruitment functions."),
            "name": "Beta (Beverton-Holt / Ricker)"
        },
        "total_recur_recruits": {
            "type": "number",
            "required": False,
            "about": (
                "Specifies the total number of recruits that come into the "
                "population at each time step (a fixed number). Used only "
                "for the Fixed recruitment function."),
            "name": "Total Recruits per Time Step (Fixed)"
        },
        "migr_cont": {
            "type": "boolean",
            "required": True,
            "about": "if True, model uses migration.",
            "name": "Migration Parameters"
        },
        "migration_dir": {
            "validation_options": {
                "exists": True,
            },
            "type": "directory",
            "required": "migr_cont",
            "about": (
                "The selected folder contain CSV migration matrices to be "
                "used in the simulation.  Each CSV file contains a single "
                "migration matrix corresponding to an lifecycle class that "
                "migrates. The folder should contain one CSV file for each "
                "lifecycle class that migrates. The files may be "
                "named anything, but must end with an underscore followed by "
                "the name of the age or stage.  The name of the age or stage "
                "must correspond to an age or stage within the Population "
                "Parameters CSV file.  For example, a migration file might "
                "be named 'migration_adult.csv'. Each matrix cell "
                "should contain a decimal fraction indicating the percentage "
                "of the population that will move from one area to another. "
                "Each column should sum to one."),
            "name": "Migration Matrix CSV Folder (Optional)"
        },
        "val_cont": {
            "type": "boolean",
            "required": True,
            "about": "if True, model computes valuation.",
            "name": "Valuation Parameters"
        },
        "frac_post_process": {
            "validation_options": {},
            "type": "number",
            "required": "val_cont",
            "about": (
                "Decimal fraction indicating the percentage of harvested "
                "catch remaining after post-harvest processing is complete."),
            "name": "Fraction of Harvest Kept After Processing"
        },
        "unit_price": {
            "type": "number",
            "required": "val_cont",
            "about": (
                "Specifies the price per harvest unit. If 'Harvest by "
                "Individuals or Weight' was set to 'Individuals', this should "
                "be the price per individual. If set to 'Weight', this "
                "should be the price per unit weight."),
            "name": "Unit Price"
        }
    }
}


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
    return validation.validate(args, ARGS_SPEC['args'])
