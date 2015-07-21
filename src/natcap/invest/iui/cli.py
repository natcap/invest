"""
Single entry point for all InVEST applications.
"""

import argparse
import glob
import os
import sys

import natcap.versioner
import natcap.invest
import natcap.invest.iui.modelui

TOOLS_IN_DEVELOPMENT = set([
    'seasonal_water_yield',
    'ndr',
    'globio',
    'seasonal_water_yield',
    'scenic_quality',
    'crop_production',
    'scenic_quality',
])

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
        basedir = os.path.dirname(__file__)
    #print 'BASEDIR: %s' % basedir
    return basedir


def list_models():
    """
    List all models that have .json files defined in the iui dir.
    """
    model_names = []
    json_files = os.path.join(iui_dir(), '*.json')
    for json_file in glob.glob(json_files):
        json_name, _ = os.path.splitext(json_file)
        json_name = os.path.basename(json_name)

        model_names.append(json_name)
    return sorted(model_names)

def print_models():
    """
    Pretty-print available models.
    """
    print 'Available models:'
    for model_name in list_models():
        if model_name in TOOLS_IN_DEVELOPMENT:
            unstable = '    UNSTABLE'
        else:
            unstable = ''
        print '    %-30s %s' % (model_name, unstable)

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
    parser.add_argument('--version', action='version', version=natcap.invest.__version__)
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('model', nargs='?', help='The model/tool to run.  Use --list to show available models/tools.')

    args = parser.parse_args()

    if args.list is True:
        print_models()

    if args.model not in list_models():
        parser.print_help()
        print_models()
        args.model = raw_input("Choose a model: ")
        if args.model not in list_models():
            print "Error: %s not a known model" % args.model
            return 1

    natcap.invest.iui.modelui.main(args.model + '.json')

if __name__ == '__main__':
    main()


