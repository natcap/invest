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
    usage: invest [-h] [--version] [--list] [model]

    Integrated Valuation of Ecosystem Services and Tradeoffs.InVEST (Integrated
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
      model       The model/tool to run. Use --list to show available
                  models/tools. Identifiable model prefixes may also be used.

    optional arguments:
      -h, --help  show this help message and exit
      --version   show program's version number and exit
      --list      List available models

To list the available models::

    $ invest --list

To launch a model::

    $ invest <modelname>

