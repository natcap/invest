"""
Single entry point for all InVEST applications.
"""

import argparse
import os

import pkg_resources
import natcap.versioner


# Goal: allow InVEST models to be run at the command-line, without a UI.
#   problem: how to identify which models have Qt UIs available?
#       1.  If we can't import natcap.ui, we don't have any qt uis.
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
#       3.  Allow model to run with a scenario file.
#       PS: Optionally, don't validate inputs, but do validate by default.

_CLI_CONFIG_FILENAME = 'cli_config'


def list_models():
   from .ui import MODELS

   return sorted(MODELS.keys())


def print_models():
    """
    Pretty-print available models.
    """
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

    if args.list is True:
        print_models()
        return 0

    # args.model is '' or None when the user provided no input.
    if args.model in ['', None]:
        parser.print_help()
        print ''
        print_models()

        if not args.model:
            return 1

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


