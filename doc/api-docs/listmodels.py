import itertools
import pkgutil
import importlib
import logging
import os
import argparse
import sys

import natcap.invest

LOGGER = logging.getLogger('listmodels.py')


def main(args=None):
    """List out all InVEST model entrypoints in RST.

    This is a main function and is intended to be used as a CLI.  For the
    full list of options, use ``listmodels --help``.  Parameters may also
    be provided as a list of strings.

    Writes a file (defaults to ``models.rst``) with the list of models and
    their automodule documentation directives for processing by sphinx.

    Arguments:
        args (list): Optional. A list of string command-line parameters to
            parse.  If not provided, ``sys.argv[1:]`` will be used.

    Returns:
        None

    """
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description=(
        'List all models that have an execute function in RST'))
    parser.add_argument('outfile', type=str, nargs='?', default='models.rst')

    parsed_args = parser.parse_args(args)

    all_modules = {}
    iteration_args = {
        'path': natcap.invest.__path__,
        'prefix': 'natcap.invest.',
    }
    for _loader, name, _is_pkg in itertools.chain(
            pkgutil.walk_packages(**iteration_args),  # catch packages
            pkgutil.iter_modules(**iteration_args)):  # catch modules
        try:
            module = importlib.import_module(name)
        except Exception:
            # If we encounter an exception when importing a module, log it
            # but continue.
            LOGGER.exception('Error importing %s', name)
            continue

        if hasattr(module, 'execute'):
            all_modules[name] = module.execute

    excluded_modules = [
        '_core',  # anything ending in '_core'
        '_example_model',
        'carbon_biophysical',
        'carbon_valuation',
        'coastal_vulnerability_post_processing',
        'usage_logger',
        'recmodel_server',
        'recmodel_workspace_fetcher',
        'nearshore_wave_and_erosion', '3p0', 'WaveModel'
    ]

    print '\n\n'

    if os.path.isabs(parsed_args.outfile):
        filename = parsed_args.outfile
    else:
        filename = os.path.join(os.path.dirname(__file__), parsed_args.outfile)

    with open(filename, 'w') as models_rst:
        models_rst.write(
            '=========================\n'
            'InVEST Model Entry Points\n'
            '=========================\n'
            '\n'
            '.. contents::\n'
            '\n'
        )

        for name, module in sorted(all_modules.iteritems(),
                                   key=lambda x: x[0]):
            if any([name.endswith(x) for x in excluded_modules]):
                continue
            try:
                module_title = module.__doc__.strip().split('\n')[0]
                if module_title.endswith('.'):
                    module_title = module_title[:-1]
            except AttributeError:
                module_title = 'NONE'

            models_rst.write((
                '{module_title}\n'
                '{underline}\n'
                '.. autofunction:: {modname}.execute\n\n').format(
                    module_title=module_title,
                    underline=''.join(['=']*len(module_title)),
                    modname=name
                )
            )


if __name__ == '__main__':
    main()
