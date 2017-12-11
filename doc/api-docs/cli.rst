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
    usage: invest [-h] [--version] [-v | --debug] [--list] [-l] [-d [DATASTACK]]
                  [-w [WORKSPACE]] [-q] [-y] [-n]
                  [model]

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
      model                 The model/tool to run. Use --list to show available
                            models/tools. Identifiable model prefixes may also be
                            used. Alternatively,specify "launcher" to reveal a
                            model launcher window.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -v, --verbose         Increase verbosity. Affects how much is printed to the
                            console and (if running in headless mode) how much is
                            written to the logfile.
      --debug               Enable debug logging. Alias for -vvvvv
      --list                List available models
      -l, --headless        Attempt to run InVEST without its GUI.
      -d [DATASTACK], --datastack [DATASTACK]
                            Run the specified model with this datastack
      -w [WORKSPACE], --workspace [WORKSPACE]
                            The workspace in which outputs will be saved
    
    gui options:
      These options are ignored if running in headless mode
    
      -q, --quickrun        Run the target model without validating and quit with
                            a nonzero exit status if an exception is encountered
    
    headless options:
      -y, --overwrite       Overwrite the workspace without prompting for
                            confirmation
      -n, --no-validate     Do not validate inputs before running the model.


To list the available models::

    $ invest --list

To launch a model::

    $ invest <modelname>

