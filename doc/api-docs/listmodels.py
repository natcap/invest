import itertools
import pkgutil
import importlib
import logging
import os
import argparse
import sys
import warnings

import natcap.invest

logging.basicConfig()
LOGGER = logging.getLogger('listmodels')

MODEL_RST_TEMPLATE = """
.. _models:

=========================
InVEST Model Entry Points
=========================

All InVEST models share a consistent python API:

    1) The model has a function called ``execute`` that takes a single python
       dict (``"args"``) as its argument.
    2) This arguments dict contains an entry, ``'workspace_dir'``, which
       points to the folder on disk where all files created by the model
       should be saved.

Calling a model requires importing the model's execute function and then
calling the model with the correct parameters.  For example, if you were
to call the Carbon Storage and Sequestration model, your script might
include

.. code-block:: python

    import natcap.invest.carbon.carbon_combined
    args = {
        'workspace_dir': 'path/to/workspace'
        # Other arguments, as needed for Carbon.
    }

    natcap.invest.carbon.carbon_combined.execute(args)

For examples of scripts that could be created around a model run,
or multiple successive model runs, see :ref:`CreatingSamplePythonScripts`.


.. contents:: Available Models and Tools:
    :local:

"""

EXCLUDED_MODULES = [
    '_core',  # anything ending in '_core'
    '_example_model',
    'carbon_biophysical',
    'carbon_valuation',
    'coastal_vulnerability_post_processing',
    'usage_logger',
    'recmodel_server',
    'recmodel_workspace_fetcher',
]


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
    if args is None:
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

        if any([name.endswith(x) for x in EXCLUDED_MODULES]):
            continue

        # Skip anything within the UI.
        if name.startswith('natcap.invest.ui'):
            continue

        try:
            module = importlib.import_module(name)
        except Exception:
            # If we encounter an exception when importing a module, log it
            # but continue.
            LOGGER.exception('Error importing %s', name)
            continue

        if not hasattr(module, 'execute'):
            continue

        try:
            module_title = module.execute.__doc__.strip().split('\n')[0]
            if module_title.endswith('.'):
                module_title = module_title[:-1]
        except AttributeError:
            module_title = None
        all_modules[name] = module_title

    print('\n\n')

    if os.path.isabs(parsed_args.outfile):
        filename = parsed_args.outfile
    else:
        filename = os.path.join(os.path.dirname(__file__), parsed_args.outfile)

    LOGGER.debug('Writing models to file %s', filename)

    with open(filename, 'w') as models_rst:
        models_rst.write(MODEL_RST_TEMPLATE)

        for name, module_title in sorted(all_modules.items(),
                                         key=lambda x: x[1]):
            if module_title is None:
                warnings.warn('%s has no title' % name)
                module_title = 'unknown'

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
