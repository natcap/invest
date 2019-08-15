================================================================
InVEST: Integrated Valuation of Ecosystem Services and Tradeoffs
================================================================

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

Additionally, a python binding for Qt is needed to use the InVEST GUI, but is
not required for development against ``natcap.invest``.  InVEST uses the
interface library ``qtpy`` to support ``PyQt4``, ``PyQt5``, ``PySide``, and
``PySide2``.  In our experience, ``PyQt4`` and ``PySide2`` have been easiest
to work with.  One of these bindings for Qt must be installed in order to use
the GUI.


Installing from Source
----------------------

If you have a compiler installed and configured for your system, and
dependencies installed, the easiest way to install InVEST as a python package
is:

.. code-block:: console

    $ pip install natcap.invest


Installing the latest development version
-----------------------------------------

The latest development version of InVEST can be installed from our mercurial
source tree:

.. code-block:: console

    $ pip install hg+https://bitbucket.org/natcap/invest@develop


Usage
=====

To run an InVEST model from the command-line, use the ``invest`` cli single
entry point:

.. code-block:: console

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


Development
===========

Dependencies for developing InVEST are listed in ``requirements.txt`` and in
``requirements-dev.txt``.  If you're running a GUI, you'll need a Qt binding
(see above) and the packages installed in ``requirements-gui.txt``.

Support
=======

Participate in the NatCap forums here:
http://community.naturalcapitalproject.org

Bugs may be reported at http://bitbucket.org/natcap/invest
