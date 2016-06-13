================================================================
InVEST: Integrated Valuation of Ecosystem Services and Tradeoffs 
================================================================

.. image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=invest-nightly-develop
  :target: http://builds.naturalcapitalproject.org/job/invest-nightly-develop

About  InVEST
=============

InVEST (Integrated Valuation of Ecosystem Services and Tradeoffs) is a family
of tools for quantifying the values of natural capital in clear, credible, and
practical ways. In promising a return (of societal benefits) on investments in
nature, the scientific community needs to deliver knowledge and tools to
quantify and forecast this return. InVEST enables decision-makers to quantify
the importance of natural capital, to assess the tradeoffs associated with
alternative choices, and to integrate conservation and human development.

Older versions of InVEST ran as script tools in the ArcGIS ArcToolBox environment,
but have almost all been ported over to a purely open-source python environment.

InVEST is licensed under a permissive, modified BSD license.

For more information, see:
  * `InVEST API documentation <http://invest.readthedocs.io/>`_
  * InVEST on `bitbucket <https://bitbucket.org/natcap/invest>`__
  * The `Natural Capital Project website <http://naturalcapitalproject.org>`__.


.. Everything after this comment will be included in the API docs.
.. START API

Installing InVEST
=================

Python Dependencies
-------------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt``:

.. These dependencies are listed here statically because when I push the
   readme page to PyPI, they won't render if I use the .. include::
   directive.  Annoying, but oh well.  It just means that we'll need to
   periodically check that this list is accurate.

.. code-block::

{requirements}

Additionally, ``PyQt4`` is required to use the ``invest`` cli, but is not
required for development against ``natcap.invest``.

Installing from Source
----------------------

If you have a compiler installed and configured for your system, and
dependencies installed, the easiest way to install InVEST as a python package 
is:

.. code-block:: console

    $ pip install natcap.invest

If you are working within virtual environments, there is a `documented issue
with namespaces 
<https://bitbucket.org/pypa/setuptools/issues/250/develop-and-install-single-version>`__
in setuptools that may cause problems when importing packages within the
``natcap`` namespace.  The current workaround is to use these extra pip flags:

.. code-block:: console

    $ pip install natcap.invest --egg --no-binary :all:

Installing the latest development version
-----------------------------------------

The latest development version of InVEST can be installed from our source tree:

.. code-block:: console

    $ pip install hg+https://bitbucket.org/natcap/invest@develop


Usage
=====

To run an InVEST model from the command-line, use the ``invest`` cli single
entry point:

.. code-block:: console

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

To list the available models:

.. code-block:: console 

    $ invest --list


Development
===========

Dependencies for developing InVEST are listed in ``requirements.txt`` and in
``requirements-dev.txt``.

Support
=======

Participate in the NatCap forums here:
http://forums.naturalcapitalproject.org

Bugs may be reported at http://bitbucket.org/natcap/invest
