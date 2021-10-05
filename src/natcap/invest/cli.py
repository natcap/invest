# coding=UTF-8
"""Single entry point for all InVEST applications."""
import argparse
import codecs
import collections
import datetime
import importlib
import json
import logging
import multiprocessing
import os
import platform
import pprint
import sys
import textwrap
import warnings

try:
    from . import __version__
    from . import utils
    from . import datastack
except (ValueError, ImportError):
    # When we're in a PyInstaller build, this isn't a module.
    from natcap.invest import __version__
    from natcap.invest import utils
    from natcap.invest import datastack


DEFAULT_EXIT_CODE = 1
LOGGER = logging.getLogger(__name__)
_UIMETA = collections.namedtuple('UIMeta', 'humanname pyname gui aliases')

# this is used only to build up the MODEL_UIS dictionary.
# use MODEL_UIS for everything else.
_MODELS = {
    # model name: python module name, gui, aliases
    'carbon': ('carbon', 'carbon.Carbon', ()),
    'coastal_blue_carbon': ('coastal_blue_carbon.coastal_blue_carbon',
        'cbc.CoastalBlueCarbon', ('cbc',)),
    'coastal_blue_carbon_preprocessor': ('coastal_blue_carbon.preprocessor', 'cbc.CoastalBlueCarbonPreprocessor', ('cbc_pre',)),
    'coastal_vulnerability': ('coastal_vulnerability', 'coastal_vulnerability.CoastalVulnerability', ('cv',)),
    'crop_production_percentile': ('crop_production_percentile', 'crop_production.CropProductionPercentile', ('cpp',)),
    'crop_production_regression': ('crop_production_regression', 'crop_production.CropProductionRegression', ('cpr',)),
    'delineateit': ('delineateit.delineateit', 'delineateit.Delineateit', ()),
    'finfish_aquaculture': ('finfish_aquaculture.finfish_aquaculture', 'finfish.FinfishAquaculture', ()),
    'fisheries': ('fisheries.fisheries', 'fisheries.Fisheries', ()),
    'fisheries_hst': ('fisheries.fisheries_hst', 'fisheries.FisheriesHST', ()),
    'forest_carbon_edge_effect': ('forest_carbon_edge_effect', 'forest_carbon.ForestCarbonEdgeEffect', ('fc',)),
    'globio': ('globio', 'globio.GLOBIO', ()),
    'habitat_quality': ('habitat_quality', 'habitat_quality.HabitatQuality', ('hq',)),
    'habitat_risk_assessment': ('hra', 'hra.HabitatRiskAssessment', ('hra',)),
    'hydropower_water_yield': ('hydropower.hydropower_water_yield', 'hydropower.HydropowerWaterYield', ('hwy',)),
    'ndr': ('ndr.ndr', 'ndr.Nutrient', ()),
    'pollination': ('pollination', 'pollination.Pollination', ()),
    'recreation': ('recreation.recmodel_client', 'recreation.Recreation', ()),
    'routedem': ('routedem', 'routedem.RouteDEM', ()),
    'scenario_generator_proximity': ('scenario_gen_proximity', 'scenario_gen.ScenarioGenProximity', ('sgp',)),
    'scenic_quality': ('scenic_quality.scenic_quality', 'scenic_quality.ScenicQuality', ('sq',)),
    'sdr': ('sdr.sdr', 'sdr.SDR', ()),
    'seasonal_water_yield': ('seasonal_water_yield.seasonal_water_yield', 'seasonal_water_yield.SeasonalWaterYield', ('swy',)),
    'wind_energy': ('wind_energy', 'wind_energy.WindEnergy', ()),
    'wave_energy': ('wave_energy', 'wave_energy.WaveEnergy', ()),
    'urban_flood_risk_mitigation': ('urban_flood_risk_mitigation', 'urban_flood_risk_mitigation.UrbanFloodRiskMitigation', ('ufrm',)),
    'urban_cooling_model': ('urban_cooling_model', 'urban_cooling_model.UrbanCoolingModel', ('ucm',)),

}

MODEL_UIS = {
    model_name: _UIMETA(
        humanname=importlib.import_module(
            f'natcap.invest.{pyname}').ARGS_SPEC['model_name'],
        pyname=f'natcap.invest.{pyname}',
        gui=gui,
        aliases=aliases
    ) for (model_name, (pyname, gui, aliases)) in _MODELS.items()}

# Build up an index mapping aliases to modelname.
# ``modelname`` is the key to the MODEL_UIS dict, above.
_MODEL_ALIASES = {}
for _modelname, _meta in MODEL_UIS.items():
    for _alias in _meta.aliases:
        assert _alias not in _MODEL_ALIASES, (
            'Alias %s already defined for model %s') % (
                _alias, _MODEL_ALIASES[_alias])
        _MODEL_ALIASES[_alias] = _modelname


def build_model_list_table():
    """Build a table of model names, aliases and other details.

    This table is a table only in the sense that its contents are aligned
    into columns, but are not separated by a delimited.  This table
    is intended to be printed to stdout.

    Returns:
        A string representation of the formatted table.
    """
    model_names = sorted(MODEL_UIS.keys())
    max_model_name_length = max(len(name) for name in model_names)

    # Adding 3 to max alias name length for the parentheses plus some padding.
    max_alias_name_length = max(len(', '.join(meta.aliases))
                                for meta in MODEL_UIS.values()) + 3
    template_string = '    {modelname} {aliases} {humanname} {usage}'
    strings = ['Available models:']
    for model_name in sorted(MODEL_UIS.keys()):
        usage_string = '(No GUI available)'
        if MODEL_UIS[model_name].gui is not None:
            usage_string = ''

        alias_string = ', '.join(MODEL_UIS[model_name].aliases)
        if alias_string:
            alias_string = '(%s)' % alias_string

        strings.append(template_string.format(
            modelname=model_name.ljust(max_model_name_length),
            aliases=alias_string.ljust(max_alias_name_length),
            humanname=MODEL_UIS[model_name].humanname,
            usage=usage_string))
    return '\n'.join(strings) + '\n'


def build_model_list_json():
    """Build a json object of relevant information for the CLI.

    The json object returned uses the human-readable model names for keys
    and the values are another dict containing the internal python name
    of the model and the aliases recognized by the CLI.

    Returns:
        A string representation of the JSON object.

    """
    json_object = {}
    for internal_model_name, model_data in MODEL_UIS.items():
        json_object[model_data.humanname] = {
            'internal_name': internal_model_name,
            'aliases': model_data.aliases
        }

    return json.dumps(json_object)


def export_to_python(target_filepath, model, args_dict=None):
    script_template = textwrap.dedent("""\
    # coding=UTF-8
    # -----------------------------------------------
    # Generated by InVEST {invest_version} on {today}
    # Model: {modelname}

    import logging
    import sys

    import {py_model}
    import natcap.invest.utils

    LOGGER = logging.getLogger(__name__)
    root_logger = logging.getLogger()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt=natcap.invest.utils.LOG_FMT,
        datefmt='%m/%d/%Y %H:%M:%S ')
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    args = {model_args}

    if __name__ == '__main__':
        {py_model}.execute(args)
    """)

    target_model = MODEL_UIS[model].pyname
    if args_dict is None:
        model_module = importlib.import_module(name=target_model)
        spec = model_module.ARGS_SPEC
        cast_args = {key: '' for key in spec['args'].keys()}
    else:
        cast_args = dict((str(key), value) for (key, value)
                         in args_dict.items())

    with codecs.open(target_filepath, 'w', encoding='utf-8') as py_file:
        args = pprint.pformat(cast_args, indent=4)  # 4 spaces

        # Tweak formatting from pprint:
        # * Bump parameter inline with starting { to next line
        # * add trailing comma to last item item pair
        # * add extra space to spacing before first item
        args = args.replace('{', '{\n ')
        args = args.replace('}', ',\n}')
        py_file.write(script_template.format(
            invest_version=__version__,
            today=datetime.datetime.now().strftime('%c'),
            modelname=MODEL_UIS[model].humanname,
            py_model=target_model,
            model_args=args))


class SelectModelAction(argparse.Action):
    """Given a possibly-ambiguous model string, identify the model to run.

    This is a subclass of ``argparse.Action`` and is executed when the argparse
    interface detects that the user has attempted to select a model by name.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Given the user's input, determine which model they're referring to.

        When the user didn't provide a model name, we print the help and exit
        with a nonzero exit code.

        Identifiable model names are:

            * the model name (verbatim) as identified in the keys of MODEL_UIS
            * a uniquely identifiable prefix for the model name (e.g. "d"
              matches "delineateit", but "fi" matches both "fisheries" and
              "finfish"
            * a known model alias, as registered in MODEL_UIS

        If no single model can be identified based on these rules, an error
        message is printed and the parser exits with a nonzero exit code.

        See https://docs.python.org/3.7/library/argparse.html#action-classes
        for the full documentation for argparse classes and this __call__
        method.

        Overridden from argparse.Action.__call__.
        """
        known_models = sorted(list(MODEL_UIS.keys()))

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
            parser.exit(status=1, message=(
                "Error: '%s' not a known model" % values))
        else:
            parser.exit(
                status=1,
                message=(
                    "Model string '{model}' is ambiguous:\n"
                    "    {matching_models}").format(
                        model=values,
                        matching_models=' '.join(matching_models)))
        setattr(namespace, self.dest, modelname)


def main(user_args=None):
    """CLI entry point for launching InVEST runs and other useful utilities.

    This command-line interface supports two methods of launching InVEST models
    from the command-line:

        * through its GUI
        * in headless mode, without its GUI.

    Running in headless mode allows us to bypass all GUI functionality,
    so models may be run in this way without having GUI packages
    installed.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Integrated Valuation of Ecosystem Services and Tradeoffs. '
            'InVEST (Integrated Valuation of Ecosystem Services and '
            'Tradeoffs) is a family of tools for quantifying the values of '
            'natural capital in clear, credible, and practical ways. In '
            'promising a return (of societal benefits) on investments in '
            'nature, the scientific community needs to deliver knowledge and '
            'tools to quantify and forecast this return. InVEST enables '
            'decision-makers to quantify the importance of natural capital, '
            'to assess the tradeoffs associated with alternative choices, '
            'and to integrate conservation and human development.  \n\n'
            'Older versions of InVEST ran as script tools in the ArcGIS '
            'ArcToolBox environment, but have almost all been ported over to '
            'a purely open-source python environment.'),
        prog='invest'
    )
    parser.add_argument('--version', action='version',
                        version=__version__)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '-v', '--verbose', dest='verbosity', default=0, action='count',
        help=('Increase verbosity.  Affects how much logging is printed to '
              'the console and (if running in headless mode) how much is '
              'written to the logfile.'))
    verbosity_group.add_argument(
        '--debug', dest='log_level', default=logging.ERROR,
        action='store_const', const=logging.DEBUG,
        help='Enable debug logging. Alias for -vvvv')

    subparsers = parser.add_subparsers(dest='subcommand')

    listmodels_subparser = subparsers.add_parser(
        'list', help='List the available InVEST models')
    listmodels_subparser.add_argument(
        '--json', action='store_true', help='Write output as a JSON object')

    subparsers.add_parser(
        'launch', help='Start the InVEST launcher window')

    run_subparser = subparsers.add_parser(
        'run', help='Run an InVEST model')
    run_subparser.add_argument(
        '-l', '--headless', action='store_true',
        help=('Run an InVEST model without its GUI. '
              'Requires a datastack and a workspace.'))
    run_subparser.add_argument(
        '-d', '--datastack', default=None, nargs='?',
        help=('Run the specified model with this JSON datastack. '
              'Required if using --headless'))
    run_subparser.add_argument(
        '-w', '--workspace', default=None, nargs='?',
        help=('The workspace in which outputs will be saved. '
              'Required if using --headless'))
    run_subparser.add_argument(
        'model', action=SelectModelAction,  # Assert valid model name
        help=('The model to run.  Use "invest list" to list the available '
              'models.'))

    quickrun_subparser = subparsers.add_parser(
        'quickrun', help=(
            'Run through a model with a specific datastack, exiting '
            'immediately upon completion. This subcommand is only intended '
            'to be used by automated testing scripts.'))
    quickrun_subparser.add_argument(
        'model', action=SelectModelAction,  # Assert valid model name
        help=('The model to run.  Use "invest list" to list the available '
              'models.'))
    quickrun_subparser.add_argument(
        'datastack', help=('Run the model with this JSON datastack.'))
    quickrun_subparser.add_argument(
        '-w', '--workspace', default=None, nargs='?',
        help=('The workspace in which outputs will be saved.'))

    validate_subparser = subparsers.add_parser(
        'validate', help=(
            'Validate the parameters of a datastack'))
    validate_subparser.add_argument(
        '--json', action='store_true', help='Write output as a JSON object')
    validate_subparser.add_argument(
        'datastack', help=('Path to a JSON datastack.'))

    getspec_subparser = subparsers.add_parser(
        'getspec', help=('Get the specification of a model.'))
    getspec_subparser.add_argument(
        '--json', action='store_true', help='Write output as a JSON object')
    getspec_subparser.add_argument(
        'model', action=SelectModelAction,  # Assert valid model name
        help=('The model for which the spec should be fetched.  Use "invest '
              'list" to list the available models.'))

    serve_subparser = subparsers.add_parser(
        'serve', help=('Start the flask app on the localhost.'))
    serve_subparser.add_argument(
        '--port', type=int, default=56789,
        help='Port number for the Flask server')

    export_py_subparser = subparsers.add_parser(
        'export-py', help=('Save a python script that executes a model.'))
    export_py_subparser.add_argument(
        'model', action=SelectModelAction,  # Assert valid model name
        help=('The model that the python script will execute.  Use "invest '
              'list" to list the available models.'))
    export_py_subparser.add_argument(
        '-f', '--filepath', default=None,
        help='Define a location for the saved .py file')

    args = parser.parse_args(user_args)

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
    log_level = min(args.log_level, logging.ERROR - (args.verbosity*10))
    handler.setLevel(max(log_level, logging.DEBUG))  # don't go below DEBUG
    root_logger.addHandler(handler)
    LOGGER.info('Setting handler log level to %s', log_level)

    # FYI: Root logger by default has a level of logging.WARNING.
    # To capture ALL logging produced in this system at runtime, use this:
    # logging.getLogger().setLevel(logging.DEBUG)
    # Also FYI: using logging.DEBUG means that the logger will defer to
    # the setting of the parent logger.
    logging.getLogger('natcap').setLevel(logging.DEBUG)

    if args.subcommand == 'list':
        if args.json:
            message = build_model_list_json()
        else:
            message = build_model_list_table()

        sys.stdout.write(message)
        parser.exit()

    if args.subcommand == 'launch':
        from natcap.invest.ui import launcher
        parser.exit(launcher.main())

    if args.subcommand == 'validate':
        try:
            parsed_datastack = datastack.extract_parameter_set(args.datastack)
        except Exception as error:
            parser.exit(
                1, "Error when parsing JSON datastack:\n    " + str(error))

        model_module = importlib.import_module(
            name=parsed_datastack.model_name)

        try:
            validation_result = model_module.validate(parsed_datastack.args)
        except KeyError as missing_keys_error:
            if args.json:
                message = json.dumps(
                    {'validation_results': {
                        str(list(missing_keys_error.args)): 'Key is missing'}})
            else:
                message = ('Datastack is missing keys:\n    ' +
                           str(missing_keys_error.args))

            # Missing keys have an exit code of 1 because that would indicate
            # probably programmer error.
            sys.stdout.write(message)
            parser.exit(1)
        except Exception as error:
            parser.exit(
                1, ('Datastack could not be validated:\n    ' +
                    str(error)))

        # Even validation errors will have an exit code of 0
        if args.json:
            message = json.dumps({
                'validation_results': validation_result})
        else:
            message = pprint.pformat(validation_result)

        sys.stdout.write(message)
        parser.exit(0)

    if args.subcommand == 'getspec':
        target_model = MODEL_UIS[args.model].pyname
        model_module = importlib.import_module(name=target_model)
        spec = model_module.ARGS_SPEC

        if args.json:
            message = json.dumps(spec)
        else:
            message = pprint.pformat(spec)
        sys.stdout.write(message)
        parser.exit(0)

    if args.subcommand == 'run' and args.headless:
        if not args.datastack:
            parser.exit(1, 'Datastack required for headless execution.')

        try:
            parsed_datastack = datastack.extract_parameter_set(args.datastack)
        except Exception as error:
            parser.exit(
                1, "Error when parsing JSON datastack:\n    " + str(error))

        if not args.workspace:
            if ('workspace_dir' not in parsed_datastack.args or
                    parsed_datastack.args['workspace_dir'] in ['', None]):
                parser.exit(
                    1, ('Workspace must be defined at the command line '
                        'or in the datastack file'))
        else:
            parsed_datastack.args['workspace_dir'] = args.workspace

        target_model = MODEL_UIS[args.model].pyname
        model_module = importlib.import_module(name=target_model)
        LOGGER.info('Imported target %s from %s',
                    model_module.__name__, model_module)

        with utils.prepare_workspace(parsed_datastack.args['workspace_dir'],
                                     name=parsed_datastack.model_name,
                                     logging_level=log_level):
            LOGGER.log(datastack.ARGS_LOG_LEVEL,
                       'Starting model with parameters: \n%s',
                       datastack.format_args_dict(parsed_datastack.args,
                                                  parsed_datastack.model_name))

            # We're deliberately not validating here because the user
            # can just call ``invest validate <datastack>`` to validate.
            #
            # Exceptions will already be logged to the logfile but will ALSO be
            # written to stdout if this exception is uncaught.  This is by
            # design.
            model_module.execute(parsed_datastack.args)

    # If we're running in a GUI (either through ``invest run`` or
    # ``invest quickrun``), we'll need to load the Model's GUI class,
    # populate parameters and then (if in a quickrun) exit when the model
    # completes.  Quickrun functionality is primarily useful for automated
    # testing of the model interfaces.
    if (args.subcommand == 'run' and not args.headless or
            args.subcommand == 'quickrun'):

        # Creating this warning for future us to alert us to potential issues
        # if/when we forget to define QT_MAC_WANTS_LAYER at runtime.
        if (platform.system() == "Darwin" and
                "QT_MAC_WANTS_LAYER" not in os.environ):
            warnings.warn(
                "Mac OS X Big Sur may require the 'QT_MAC_WANTS_LAYER' "
                "environment variable to be defined in order to run.  If "
                "the application hangs on startup, set 'QT_MAC_WANTS_LAYER=1' "
                "in the shell running this CLI.", RuntimeWarning)

        from natcap.invest.ui import inputs

        gui_class = MODEL_UIS[args.model].gui
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
        quickrun = False
        if args.subcommand == 'quickrun':
            quickrun = True
        model_form.run(quickrun=quickrun)
        app_exitcode = inputs.QT_APP.exec_()

        # Handle a graceful exit
        if model_form.form.run_dialog.messageArea.error:
            parser.exit(DEFAULT_EXIT_CODE,
                        'Model %s: run failed\n' % args.model)

        if app_exitcode != 0:
            parser.exit(app_exitcode,
                        'App terminated with exit code %s\n' % app_exitcode)

    if args.subcommand == 'serve':
        import natcap.invest.ui_server
        natcap.invest.ui_server.app.run(port=args.port)
        parser.exit(0)

    if args.subcommand == 'export-py':
        target_filepath = args.filepath
        if not args.filepath:
            target_filepath = f'{args.model}_execute.py'
        export_to_python(target_filepath, args.model)
        parser.exit()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
