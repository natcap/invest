.. _cli:

==============
The InVEST CLI
==============

Installing
==========

The ``invest`` cli application is installed with the ``natcap.invest`` python
package.  See `Installing InVEST <installing.html>`_

It is also available to Windows users who installed the InVEST Workbench. 
Just replace ``invest`` in the examples below with the full path to 
``invest.exe`` (e.g. ``C:\Program Files\InVEST 3.13.0 Workbench\resources\invest\invest.exe``)

Usage
=====

To run an InVEST model from the command-line, use the ``invest`` cli single
entry point::

    $ invest --help
    usage: invest [-h] [--version] [-v | --debug]
                  [--taskgraph-log-level {DEBUG,INFO,WARNING,ERROR}] [-L {en,es,zh}]
                  {list,run,validate,getspec,serve,export-py} ...

    Integrated Valuation of Ecosystem Services and Tradeoffs. InVEST (Integrated Valuation
    of Ecosystem Services and Tradeoffs) is a family of tools for quantifying the values of
    natural capital in clear, credible, and practical ways. In promising a return (of
    societal benefits) on investments in nature, the scientific community needs to deliver
    knowledge and tools to quantify and forecast this return. InVEST enables decision-makers
    to quantify the importance of natural capital, to assess the tradeoffs associated with
    alternative choices, and to integrate conservation and human development. Older versions
    of InVEST ran as script tools in the ArcGIS ArcToolBox environment, but have almost all
    been ported over to a purely open-source python environment.

    positional arguments:
      {list,run,validate,getspec,serve,export-py}
        list                List the available InVEST models
        run                 Run an InVEST model
        validate            Validate the parameters of a datastack
        getspec             Get the specification of a model.
        serve               Start the flask app on the localhost.
        export-py           Save a python script that executes a model.

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -v, --verbose         Increase verbosity. Affects how much logging is printed to the
                            console and (if running in headless mode) how much is written to
                            the logfile.
      --debug               Enable debug logging. Alias for -vvv
      --taskgraph-log-level {DEBUG,INFO,WARNING,ERROR}
                            Set the logging level for Taskgraph. Affects how much logging
                            Taskgraph prints to the console and (if running in headless
                            mode) how much is written to the logfile.
      -L {en,es,zh}, --language {en,es,zh}
                            Choose a language. Model specs, names, and validation messages
                            will be translated. Log messages are not translated. Value
                            should be an ISO 639-1 language code. Supported options are: en
                            (English), es (español), zh (中文).

To list the available models::

    $ invest list

To run a model directly from the command-line::

    $ invest -vvv run <modelname> -d <datastack json file> -w <output_workspace>

For more detailed instructions, get the help for each command::

    $ invest run --help
