==============
The InVEST CLI
==============

Installing
==========

The ``invest`` cli application is installed with the ``natcap.invest`` python
package.  See `Installing InVEST <installing.html>`_

Usage
=====

To run an InVEST model from the command-line, use the ``invest`` cli single
entry point::

    $ invest --help
    usage: invest [-h] [-v | --debug] {list,launch,run,quickrun,validate} ...

    Integrated Valuation of Ecosystem Services and Tradeoffs. InVEST (Integrated
    Valuation of Ecosystem Services and Tradeoffs) is a family of tools for
    quantifying the values of natural capital in clear, credible, and practical
    ways. In promising a return (of societal benefits) on investments in nature,
    the scientific community needs to deliver knowledge and tools to quantify and
    forecast this return. InVEST enables decision-makers to quantify the
    importance of natural capital, to assess the tradeoffs associated with
    alternative choices, and to integrate conservation and human development.
    Older versions of InVEST ran as script tools in the ArcGIS ArcToolBox
    environment, but have almost all been ported over to a purely open-source
    python environment.

    positional arguments:
      {list,launch,run,quickrun,validate}
        list                List the available InVEST models
        launch              Start the InVEST launcher window
        run                 Run an InVEST model
        quickrun            Run through a model with a specific datastack, exiting
                            immediately upon completion
        validate            Validate the parameters of a datastack

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase verbosity. Affects how much logging is
                            printed to the console and (if running in headless
                            mode) how much is written to the logfile.
      --debug               Enable debug logging. Alias for -vvvvv


To list the available models::

    $ invest list

To launch a model::

    $ invest run <modelname>

