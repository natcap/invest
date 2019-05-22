# coding=UTF-8
"""Single entry point for all InVEST applications."""
from __future__ import absolute_import

import argparse
import os
import importlib
import logging
import sys
import collections
import pprint
import multiprocessing

try:
    from . import utils
except ValueError:
    # When we're in a pyinstaller build, this isn't a module.
    from natcap.invest import utils


DEFAULT_EXIT_CODE = 1
LOGGER = logging.getLogger(__name__)
_UIMETA = collections.namedtuple('UIMeta', 'pyname gui aliases')

_MODEL_UIS = {
    'carbon': _UIMETA(
        pyname='natcap.invest.carbon',
        gui='carbon.Carbon',
        aliases=()),
    'coastal_blue_carbon': _UIMETA(
        pyname='natcap.invest.coastal_blue_carbon.coastal_blue_carbon',
        gui='cbc.CoastalBlueCarbon',
        aliases=('cbc',)),
    'coastal_blue_carbon_preprocessor': _UIMETA(
        pyname='natcap.invest.coastal_blue_carbon.preprocessor',
        gui='cbc.CoastalBlueCarbonPreprocessor',
        aliases=('cbc_pre',)),
    'coastal_vulnerability': _UIMETA(
        pyname='natcap.invest.coastal_vulnerability',
        gui='coastal_vulnerability.CoastalVulnerability',
        aliases=('cv',)),
    'crop_production_percentile': _UIMETA(
        pyname='natcap.invest.crop_production_percentile',
        gui='crop_production.CropProductionPercentile',
        aliases=('cpp',)),
    'crop_production_regression': _UIMETA(
        pyname='natcap.invest.crop_production_regression',
        gui='crop_production.CropProductionRegression',
        aliases=('cpr',)),
    'delineateit': _UIMETA(
        pyname='natcap.invest.routing.delineateit',
        gui='routing.Delineateit',
        aliases=()),
    'finfish_aquaculture': _UIMETA(
        pyname='natcap.invest.finfish_aquaculture.finfish_aquaculture',
        gui='finfish.FinfishAquaculture',
        aliases=()),
    'fisheries': _UIMETA(
        pyname='natcap.invest.fisheries.fisheries',
        gui='fisheries.Fisheries',
        aliases=()),
    'fisheries_hst': _UIMETA(
        pyname='natcap.invest.fisheries.fisheries_hst',
        gui='fisheries.FisheriesHST',
        aliases=()),
    'forest_carbon_edge_effect': _UIMETA(
        pyname='natcap.invest.forest_carbon_edge_effect',
        gui='forest_carbon.ForestCarbonEdgeEffect',
        aliases=('fc',)),
    'globio': _UIMETA(
        pyname='natcap.invest.globio',
        gui='globio.GLOBIO',
        aliases=()),
    'habitat_quality': _UIMETA(
        pyname='natcap.invest.habitat_quality',
        gui='habitat_quality.HabitatQuality',
        aliases=('hq',)),
    'habitat_risk_assessment': _UIMETA(
        pyname='natcap.invest.hra',
        gui='hra.HabitatRiskAssessment',
        aliases=('hra',)),
    'hydropower_water_yield': _UIMETA(
        pyname='natcap.invest.hydropower.hydropower_water_yield',
        gui='hydropower.HydropowerWaterYield',
        aliases=('hwy',)),
    'ndr': _UIMETA(
        pyname='natcap.invest.ndr.ndr',
        gui='ndr.Nutrient',
        aliases=()),
    'pollination': _UIMETA(
        pyname='natcap.invest.pollination',
        gui='pollination.Pollination',
        aliases=()),
    'recreation': _UIMETA(
        pyname='natcap.invest.recreation.recmodel_client',
        gui='recreation.Recreation',
        aliases=()),
    'routedem': _UIMETA(
        pyname='natcap.invest.routedem',
        gui='routedem.RouteDEM',
        aliases=()),
    'scenario_generator_proximity': _UIMETA(
        pyname='natcap.invest.scenario_gen_proximity',
        gui='scenario_gen.ScenarioGenProximity',
        aliases=('sgp',)),
    'scenic_quality': _UIMETA(
        pyname='natcap.invest.scenic_quality.scenic_quality',
        gui='scenic_quality.ScenicQuality',
        aliases=('sq',)),
    'sdr': _UIMETA(
        pyname='natcap.invest.sdr',
        gui='sdr.SDR',
        aliases=()),
    'seasonal_water_yield': _UIMETA(
        pyname='natcap.invest.seasonal_water_yield.seasonal_water_yield',
        gui='seasonal_water_yield.SeasonalWaterYield',
        aliases=('swy',)),
    'wind_energy': _UIMETA(
        pyname='natcap.invest.wind_energy',
        gui='wind_energy.WindEnergy',
        aliases=()),
    'wave_energy': _UIMETA(
        pyname='natcap.invest.wave_energy',
        gui='wave_energy.WaveEnergy',
        aliases=()),
    'habitat_suitability': _UIMETA(
        pyname='natcap.invest.habitat_suitability',
        gui=None,
        aliases=('hs',)),
    'urban_flood_risk_mitigation': _UIMETA(
        pyname='natcap.invest.urban_flood_risk_mitigation',
        gui='urban_flood_risk_mitigation.UrbanFloodRiskMitigation',
        aliases=('ufrm',)),
    'urban_cooling_model': _UIMETA(
        pyname='natcap.invest.urban_cooling_model',
        gui='urban_cooling_model.UrbanCoolingModel',
        aliases=('ucm',)),
    'habitat_suitability': _UIMETA(
        pyname='natcap.invest.habitat_suitability',
        gui=None,
        aliases=('hs',)),
}

# Build up an index mapping aliase to modelname.
# ``modelname`` is the key to the _MODEL_UIS dict, above.
_MODEL_ALIASES = {}
for _modelname, _meta in _MODEL_UIS.iteritems():
    for _alias in _meta.aliases:
        assert _alias not in _MODEL_ALIASES, (
            'Alias %s already defined for model %s') % (
                _alias, _MODEL_ALIASES[_alias])
        _MODEL_ALIASES[_alias] = _modelname


# metadata for models: full modelname, first released, full citation,
# local documentation name.

# Goal: allow InVEST models to be run at the command-line, without a UI.
#   problem: how to identify which models have Qt UIs available?
#       1.  If we can't import the ui infrastructure, we don't have any qt uis.
#       2.  We could iterate through all the model UI files and Identify the
#           model from its name and attributes.
#       3.  We could access a pre-processed list of models available, perhaps
#           written to a file during the setuptools build step.
#   problem: how to identify which models are available to the API?
#       1.  Recursively import natcap.invest and look for modules with execute
#           functions available.
#       2.  Import all of the execute functions to a known place (an __init__
#           file?).
#   problem: how to provide parameters?
#       1.  If execute is parseable, just extract parameters from the docstring
#           and allow each param to be provided as a CLI flag.
#       2.  Allow parameters to be passed as a JSON file
#       3.  Allow model to run with a datastack file.
#       PS: Optionally, don't validate inputs, but do validate by default.


def build_model_list_table():
    """Build a table of model names, aliases and other details.

    This table is a table only in the sense that its contents are aligned
    into columns, but are not separated by a delimited.  This table
    is intended to be printed to stdout.

    Returns:
        A string representation of the formatted table.
    """
    model_names = sorted(_MODEL_UIS.keys())
    max_model_name_length = max(len(name) for name in model_names)
    max_alias_name_length = max(len(', '.join(meta.aliases))
                                for meta in _MODEL_UIS.values())
    template_string = '    {modelname} {aliases}   {usage}'
    strings = ['Available models:']
    for model_name in sorted(_MODEL_UIS.keys()):
        usage_string = '(No GUI available)'
        if _MODEL_UIS[model_name].gui is not None:
            usage_string = ''

        alias_string = ', '.join(_MODEL_UIS[model_name].aliases)
        if alias_string:
            alias_string = '(%s)' % alias_string

        strings.append(template_string.format(
            modelname=model_name.ljust(max_model_name_length),
            aliases=alias_string.ljust(max_alias_name_length),
            usage=usage_string))
    return '\n'.join(strings) + '\n'


class ListModelsAction(argparse.Action):
    """An argparse action to list the available models."""
    def __call__(self, parser, namespace, values, option_string=None):
        """Print the available models and quit the argparse parser.

        See https://docs.python.org/2.7/library/argparse.html#action-classes
        for the full documentation for argparse classes.

        Overridden from argparse.Action.__call__"""
        setattr(namespace, self.dest, self.const)
        parser.exit(message=build_model_list_table())


class SelectModelAction(argparse.Action):
    """Given a possily-ambiguous model string, identify the model to run.

    This is a subclass of ``argparse.Action`` and is executed when the argparse
    interface detects that the user has attempted to select a model by name.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        """Given the user's input, determine which model they're referring to.

        When the user didn't provide a model name, we print the help and exit
        with a nonzero exit code.

        Identifiable model names are:

            * the model name (verbatim) as identified in the keys of _MODEL_UIS
            * a uniquely identifiable prefix for the model name (e.g. "d"
              matches "delineateit", but "fi" matches both "fisheries" and
              "finfish"
            * a known model alias, as registered in _MODEL_UIS

        If no single model can be identified based on these rules, an error
        message is printed and the parser exits with a nonzero exit code.

        See https://docs.python.org/2.7/library/argparse.html#action-classes
        for the full documentation for argparse classes and this __call__
        method.

        Overridden from argparse.Action.__call__"""
        if values in ['', None]:
            parser.print_help()
            parser.exit(1, message=build_model_list_table())
        else:
            known_models = sorted(_MODEL_UIS.keys() + ['launcher'])

            matching_models = [model for model in known_models if
                               model.startswith(values)]

            exact_matches = [model for model in known_models if
                             model == values]

            if len(matching_models) == 1:  # match an identifying substring
                modelname = matching_models[0]
            elif len(exact_matches) == 1:  # match an exact modelname
                modelname = exact_matches[0]
            elif values in _MODEL_ALIASES:  # match an alias
                modelname = _MODEL_ALIASES[values]
            elif len(matching_models) == 0:
                parser.exit("Error: '%s' not a known model" % values)
            else:
                parser.exit((
                    "Model string '{model}' is ambiguous:\n"
                    "    {matching_models}").format(
                        model=values,
                        matching_models=' '.join(matching_models)))
        setattr(namespace, self.dest, modelname)


def main():
    """CLI entry point for launching InVEST runs.

    This command-line interface supports two methods of launching InVEST models
    from the command-line:

        * through its GUI
        * in headless mode, without its GUI.

    Running in headless mode allows us to bypass all GUI functionality,
    so models may be run in this way wthout having GUI packages
    installed.
    """

    parser = argparse.ArgumentParser(description=(
        'Integrated Valuation of Ecosystem Services and Tradeoffs.  '
        'InVEST (Integrated Valuation of Ecosystem Services and Tradeoffs) is '
        'a family of tools for quantifying the values of natural capital in '
        'clear, credible, and practical ways. In promising a return (of '
        'societal benefits) on investments in nature, the scientific community '
        'needs to deliver knowledge and tools to quantify and forecast this '
        'return. InVEST enables decision-makers to quantify the importance of '
        'natural capital, to assess the tradeoffs associated with alternative '
        'choices, and to integrate conservation and human development.  \n\n'
        'Older versions of InVEST ran as script tools in the ArcGIS ArcToolBox '
        'environment, but have almost all been ported over to a purely '
        'open-source python environment.'),
        prog='invest'
    )
    list_group = parser.add_mutually_exclusive_group()
    verbosity_group = parser.add_mutually_exclusive_group()
    import natcap.invest

    parser.add_argument('--version', action='version',
                        version=natcap.invest.__version__)
    verbosity_group.add_argument('-v', '--verbose', dest='verbosity', default=0,
                                 action='count', help=(
                                     'Increase verbosity. Affects how much is '
                                     'printed to the console and (if running '
                                     'in headless mode) how much is written '
                                     'to the logfile.'))
    verbosity_group.add_argument('--debug', dest='log_level',
                                 default=logging.CRITICAL,
                                 action='store_const', const=logging.DEBUG,
                                 help='Enable debug logging. Alias for -vvvvv')
    list_group.add_argument('--list', action=ListModelsAction,
                            nargs=0, const=True,
                            help='List available models')
    parser.add_argument('-l', '--headless', action='store_true',
                        dest='headless',
                        help=('Attempt to run InVEST without its GUI.'))
    parser.add_argument('-d', '--datastack', default=None, nargs='?',
                        help='Run the specified model with this datastack')
    parser.add_argument('-w', '--workspace', default=None, nargs='?',
                        help='The workspace in which outputs will be saved')

    gui_options_group = parser.add_argument_group(
        'gui options', 'These options are ignored if running in headless mode')
    gui_options_group.add_argument('-q', '--quickrun', action='store_true',
                                   help=('Run the target model without '
                                         'validating and quit with a nonzero '
                                         'exit status if an exception is '
                                         'encountered'))

    cli_options_group = parser.add_argument_group('headless options')
    cli_options_group.add_argument('-y', '--overwrite', action='store_true',
                                   default=False,
                                   help=('Overwrite the workspace without '
                                         'prompting for confirmation'))
    cli_options_group.add_argument('-n', '--no-validate', action='store_true',
                                   dest='validate', default=True,
                                   help=('Do not validate inputs before '
                                         'running the model.'))

    list_group.add_argument('model', action=SelectModelAction, nargs='?',
                            help=('The model/tool to run. Use --list to show '
                                  'available models/tools. Identifiable model '
                                  'prefixes may also be used. Alternatively,'
                                  'specify "launcher" to reveal a model '
                                  'launcher window.'))

    args = parser.parse_args()

    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S ')
    handler.setFormatter(formatter)

    # Set the log level based on what the user provides in the available
    # arguments.  Verbosity: the more v's the lower the logging threshold.
    # If --debug is used, the logging threshold is 10.
    # If the user goes lower than logging.DEBUG, default to logging.DEBUG.
    log_level = min(args.log_level, logging.CRITICAL - (args.verbosity*10))
    handler.setLevel(max(log_level, logging.DEBUG))  # don't go lower than DEBUG
    root_logger.addHandler(handler)
    LOGGER.info('Setting handler log level to %s', log_level)

    # FYI: Root logger by default has a level of logging.WARNING.
    # To capture ALL logging produced in this system at runtime, use this:
    # logging.getLogger().setLevel(logging.DEBUG)
    # Also FYI: using logging.DEBUG means that the logger will defer to
    # the setting of the parent logger.
    logging.getLogger('natcap').setLevel(logging.DEBUG)

    # Now that we've set up logging based on args, we can start logging.
    LOGGER.debug(args)

    try:
        # Importing model UI files here will usually import qtpy before we can
        # set the sip API in natcap.invest.ui.inputs.
        # Set it here, before we can do the actual importing.
        import sip
        # 2 indicates SIP/Qt API version 2
        sip.setapi('QString', 2)

        from natcap.invest.ui import inputs
    except ImportError as error:
        # Can't import UI, exit with nonzero exit code
        LOGGER.exception('Unable to import the UI')
        parser.error(('Unable to import the UI (failed with "%s")\n'
                      'Is the UI installed?\n'
                      '    pip install natcap.invest[ui]') % error)

    if args.model == 'launcher':
        from natcap.invest.ui import launcher
        launcher.main()

    elif args.headless:
        from natcap.invest import datastack
        target_mod = _MODEL_UIS[args.model].pyname
        model_module = importlib.import_module(name=target_mod)
        LOGGER.info('imported target %s from %s',
                    model_module.__name__, model_module)

        paramset = datastack.extract_parameter_set(args.datastack)

        # prefer CLI option for workspace dir, but use paramset workspace if
        # the CLI options do not define a workspace.
        if args.workspace:
            workspace = os.path.abspath(args.workspace)
            paramset.args['workspace_dir'] = workspace
        else:
            if 'workspace_dir' in paramset.args:
                workspace = paramset.args['workspace_dir']
            else:
                parser.exit(DEFAULT_EXIT_CODE, (
                    'Workspace not defined. \n'
                    'Use --workspace to specify or add a '
                    '"workspace_dir" parameter to your datastack.'))

        with utils.prepare_workspace(workspace,
                                     name=paramset.model_name,
                                     logging_level=log_level):
            LOGGER.log(datastack.ARGS_LOG_LEVEL,
                       datastack.format_args_dict(paramset.args,
                                                  paramset.model_name))
            if not args.validate:
                LOGGER.info('Skipping validation by user request')
            else:
                model_warnings = []
                try:
                    model_warnings = getattr(
                        model_module, 'validate')(paramset.args)
                except AttributeError:
                    LOGGER.warn(
                        '%s does not have a defined validation function.',
                        paramset.model_name)
                finally:
                    if model_warnings:
                        LOGGER.warn('Warnings found: \n%s',
                                    pprint.pformat(model_warnings))

            if not args.workspace:
                args.workspace = os.getcwd()

            # If the workspace exists and we don't have up-front permission to
            # overwrite the workspace, prompt for permission.
            if (os.path.exists(args.workspace) and
                    len(os.listdir(args.workspace)) > 0 and
                    not args.overwrite):
                overwrite_denied = False
                if not sys.stdout.isatty():
                    overwrite_denied = True
                else:
                    user_response = raw_input(
                        'Workspace exists: %s\n    Overwrite? (y/n) ' % (
                            os.path.abspath(args.workspace)))
                    while user_response not in ('y', 'n'):
                        user_response = raw_input(
                            "Response must be either 'y' or 'n': ")
                    if user_response == 'n':
                        overwrite_denied = True

                if overwrite_denied:
                    # Exit the parser with an error message.
                    parser.exit(DEFAULT_EXIT_CODE,
                                ('Use --workspace to define an '
                                 'alternate workspace.  Aborting.'))
                else:
                    LOGGER.warning(
                        'Overwriting the workspace per user input %s',
                        os.path.abspath(args.workspace))

            if 'workspace_dir' not in paramset.args:
                paramset.args['workspace_dir'] = args.workspace

            # execute the model's execute function with the loaded args
            getattr(model_module, 'execute')(paramset.args)
    else:
        # import the GUI from the known class
        gui_class = _MODEL_UIS[args.model].gui
        module_name, classname = gui_class.split('.')
        module = importlib.import_module(
            name='.ui.%s' % module_name,
            package='natcap.invest')

        # Instantiate the form
        model_form = getattr(module, classname)()

        # load the datastack if one was provided
        try:
            if args.datastack:
                model_form.load_datastack(args.datastack)
        except Exception as error:
            # If we encounter an exception while loading the datastack, log the
            # exception (so it can be seen if we're running with appropriate
            # verbosity) and exit the argparse application with exit code 1 and
            # a helpful error message.
            LOGGER.exception('Could not load datastack')
            parser.exit(DEFAULT_EXIT_CODE,
                        'Could not load datastack: %s\n' % str(error))

        if args.workspace:
            model_form.workspace.set_value(args.workspace)

        # Run the UI's event loop
        model_form.run(quickrun=args.quickrun)
        app_exitcode = inputs.QT_APP.exec_()

        # Handle a graceful exit
        if model_form.form.run_dialog.messageArea.error:
            parser.exit(DEFAULT_EXIT_CODE,
                        'Model %s: run failed\n' % args.model)

        if app_exitcode != 0:
            parser.exit(app_exitcode,
                        'App terminated with exit code %s\n' % app_exitcode)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
