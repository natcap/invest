"""
Single entry point for all InVEST applications.
"""

import argparse
import glob
import os
import sys
import json

import pkg_resources
import natcap.versioner


_CLI_CONFIG_FILENAME = 'cli_config'


def iui_dir():
    """
    Return the path to the IUI folder.
    """
    if getattr(sys, 'frozen', False):
        # we are running in a |PyInstaller| bundle
        basedir = os.path.join(
            os.path.dirname(sys.executable), 'natcap', 'invest', 'iui')
    else:
        # we are running in a normal Python environment
        if os.path.exists(__file__):
            basedir = os.path.dirname(__file__)
        else:
            # Assume we're in an egg or other binary installation format that
            # doesn't use the plain directory structure.
            basedir = pkg_resources.resource_filename('natcap.invest', 'iui')
    return basedir


def load_config():
    """
    Load configuration options from a config file and assume defaults if they
    aren't there.
    """

    try:
        config_file = os.path.join(iui_dir(), _CLI_CONFIG_FILENAME + '.json')
        user_config = json.load(open(config_file))
    except IOError:
        # Raised when the cli config file hasn't been defined or can't be
        # opened.  Assume that the user has not defined configuration in this
        # case.  Don't fail loudly because there are cases where we want to
        # assume this default behavior
        user_config = {}

    base_config = {
        'prompt_on_empty_input': False
    }

    base_config.update(user_config)
    return base_config


def list_models():
    """
    List all models that have .json files defined in the iui dir.

    Returns:
        A sorted list of model names.
    """
    model_names = []
    json_files = os.path.join(iui_dir(), '*.json')
    for json_file in glob.glob(json_files):
        json_name, _ = os.path.splitext(json_file)
        json_name = os.path.basename(json_name)

        if json_name == _CLI_CONFIG_FILENAME:
            continue

        model_names.append(json_name)
    return sorted(model_names)


def print_models():
    """
    Pretty-print available models.
    """
    print "Checking what's available in %s" % iui_dir()
    print 'Available models:'
    for model_name in list_models():
        print '    %-30s' % model_name


def write_console_files(out_dir, extension):
    """
    Write out console files for each of the target models to the output dir.

    Parameters:
        out_dir: The directory in which to save the console files.
        extension: The extension of the output files (e.g. 'bat', 'sh')

    Returns:
        Nothing.  Writes files to out_dir, though.
    """
    content_template = "invest %(model)s\n"
    filename_template = os.path.join(out_dir, "invest_%(modelname)s_.%(ext)s")
    for model_name in list_models():
        console_filepath = filename_template % {
            'modelname': model_name, 'ext': extension}
        console_file = open(console_filepath)
        console_file.write(content_template % {'model': model_name})
        console_file.close()


def main():
    """
    Single entry point for all InVEST model user interfaces.

    This function provides a CLI for calling InVEST models, though it it very
    primitive.  Apart from displaying a help messsage and the version, this
    function will also (optionally) list the known models (based on the found
    json filenames) and will fire up an IUI interface based on the model name
    provided.
    """

    parser = argparse.ArgumentParser(description=(
        'Integrated Valuation of Ecosystem Services and Tradeoffs.'
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
    import natcap.invest
    parser.add_argument('--version', action='version',
                        version=natcap.invest.__version__)
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--test', action='store_false',
                         help='Run in headless mode with default args.')
    parser.add_argument('model', nargs='?', help=(
        'The model/tool to run. Use --list to show available models/tools. '
        'Identifiable model prefixes may also be used.'))

    args = parser.parse_args()
    user_config = load_config()

    if args.list is True:
        print_models()
        return 0

    # args.model is '' or None when the user provided no input.
    if args.model in ['', None]:
        parser.print_help()
        print ''
        print_models()

        if user_config['prompt_on_empty_input'] is False:
            return 1

        args.model = raw_input("Choose a model: ")
        if args.model not in list_models():
            print "Error: '%s' not a known model" % args.model
            return 1

    else:
        known_models = list_models()
        matching_models = [model for model in known_models if
                           model.startswith(args.model)]

        exact_matches = [model for model in known_models if
                         model == args.model]

        if len(matching_models) == 1:
            modelname = matching_models[0]
        elif len(exact_matches) == 1:
            modelname = exact_matches[0]
        elif len(matching_models) == 0:
            print "Error: '%s' not a known model" % args.model
            return 1
        else:
            print "Model string '%s' is ambiguous:" % args.model
            print '    %s' % ' '.join(matching_models)
            return 2

        import natcap.invest.iui.modelui
        natcap.invest.iui.modelui.main(modelname + '.json', args.test)

if __name__ == '__main__':
    main()


