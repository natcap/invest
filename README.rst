InVEST: Integrated Valuation of Ecosystem Services and Tradeoffs
================================================================

InVEST (Integrated Valuation of Ecosystem Services and Tradeoffs) is a family
of tools for quantifying the values of natural capital in clear, credible, and
practical ways. In promising a return (of societal benefits) on investments in
nature, the scientific community needs to deliver knowledge and tools to
quantify and forecast this return. InVEST enables decision-makers to quantify
the importance of natural capital, to assess the tradeoffs associated with
alternative choices, and to integrate conservation and human development.

Older versions of InVEST ran as script tools in the ArcGIS ArcToolBox environment,
but have almost all been ported over to a purely open-source python environment.

.. note::
    **This repository is for InVEST 3.2.1 and later**

    This replaces our Google Code project formerly
    located at http://code.google.com/p/invest-natcap.  If you are looking to build
    InVEST <= 3.2.0, use the archived release-framework repository at
    https://bitbucket.org/natcap/invest-natcap.release-framework, and the InVEST repository
    at https://bitbucket.org/natcap/invest-natcap.invest-3.


General Information
-------------------

* Website: https://naturalcapitalproject.stanford.edu/software/invest
* Source code: https://github.com/natcap/invest
* Issue tracker: https://github.com/natcap/invest/issues
* Users' guide: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/index.html
* API documentation: http://invest.readthedocs.io/en/latest/


Dependencies
------------

OS-specific installation instructions are found either online at
http://invest.readthedocs.io/en/latest/installing.html or locally at ``doc/api-docs/installing.rst``.

.. note::
    The ``make`` commands for InVEST require a BASH shell environment. Windows
    users can use Git Bash within the Git for Windows suite. More information
    can be found at https://gitforwindows.org

Managing python dependencies
++++++++++++++++++++++++++++
We recommend using a virtual environment to manage your python dependencies, and there is
a Makefile target to assist with this::

    $ make env
    $ source env/bin/activate

Or on Windows, use the following instead from a CMD prompt::

    > make env
    > .\env\bin\activate

This makefile target is included for convenience. It uses ``conda`` and installs packages from ``conda-forge``.
It also uses the `-p` flag with `conda create`, creating a `./env` folder containing the environment.

Using a different environment folder name
"""""""""""""""""""""""""""""""""""""""""
If you prefer a different path for your environment, you may pass the environment path as
a parameter to make::

    $ make ENV=myEnv env

You could then activate the environment created at ``./myEnv``.


Using a different environment management tool
"""""""""""""""""""""""""""""""""""""""""""""
You may of course choose to manage your own virtual environment without using the Makefile.

We suggest using ``conda`` or ``mamba`` and ``conda-forge``.

``requirements.txt``, ``requirements-dev.txt`` and ``requirements-docs.txt`` list the python
dependencies needed.

Installing ``natcap.invest`` from local source code
"""""""""""""""""""""""""""""""""""""""""""""""""""
From an activated virtual environment, it's safest to uninstall any existing installation
and then install `natcap.invest`::

    $ pip uninstall natcap.invest
    $ make install

In practice, it can be convenient to use an "editable install" instead to avoid needing
to uninstall & re-install after making changes to source code::

   $ pip install -e .

Note that with an editable install any changes to non-Python (Cython, C++) files will
require compilation using one of the above installation methods.

*The Workbench is not part of the* ``natcap.invest`` *Python package. See*
``workbench/readme.md`` *for developer details.*

A successful ``natcap.invest`` installation will include the InVEST CLI::

    $ invest list

Building InVEST Distributions
-----------------------------

Once the required tools and packages are available, we can build InVEST.
On Windows, you must indicate the location of the GDAL libraries with the environment
variable ``NATCAP_INVEST_GDAL_LIB_PATH``. If you are using conda to manage dependencies
as we recommend, you can add ``NATCAP_INVEST_GDAL_LIB_PATH="$CONDA_PREFIX/Library"`` to
the commands below. (On Mac and Linux, the gdal library path is determined for you
automatically using ``gdal-config``, which isn't available on Windows.)


Building ``natcap.invest`` python package
+++++++++++++++++++++++++++++++++++++++++

A Makefile target has been created for your convenience::

    $ make python_packages

This will create a wheel for your platform and a zip source archive in ``dist/``.
Both of these files (``dist/natcap.invest*.whl`` and ``dist/natcap.invest*.zip``)
can be installed by pip.

Building python packages without GNU make
"""""""""""""""""""""""""""""""""""""""""
Python distributions may be built with the standard distutils/setuptools commands::

    $ python -m pip install build
    $ python -m build --wheel
    $ python -m build --sdist

InVEST Standalone Binaries
++++++++++++++++++++++++++

Once the appropriate dependencies are available, InVEST can also be built as a
standalone application::

    $ make binaries

An important detail about building binaries is that ``natcap.invest`` must be
installed as a wheel to ensure that the distribution information is in the
correct location.

This will create a directory at ``dist/invest`` holding the application binaries
and relevant shared libraries.

Binaries cannot be cross-compiled for other operating systems.


InVEST Workbench
++++++++++++++++++++++++

See developer instructions at ``workbench/readme.md``.



Building InVEST Documentation
-----------------------------

User's Guide
++++++++++++

To build the user's guide::

    $ make userguide

This will build HTML and PDF documentation, writing them to ``dist/userguide``
and ``dist/InVEST_*_Documentation.pdf``, respectively.

The User's Guide is maintained in a separate git repository. InVEST will build
the User's Guide with the commit defined in the ``Makefile``::

   GIT_UG_REPO                 := https://github.com/natcap/invest.users-guide
   GIT_UG_REPO_PATH            := doc/users-guide
   GIT_UG_REPO_REV             := f203ec069f9f03560c9a85b268e67ebb6b994953


API Documentation
+++++++++++++++++

To build the ``natcap.invest`` python API documentation and developer's guide::

    $ make apidocs

This will build an HTML version of the API documentation, writing it to
``dist/apidocs``.


InVEST Sample Data
------------------

InVEST is typically distributed with sample data, though, in the interest of
disk space, these data are not included in any of the standard installers.  To
build zip archives of the sample data::

    $ make sampledata

This will write the data zipfiles to ``dist/data``. ``git`` command is needed.

Sample data is tracked in a ``git-lfs`` repo and will be packaged based on the commit
defined in the ``Makefile``::

   GIT_SAMPLE_DATA_REPO        := https://bitbucket.org/natcap/invest-sample-data.git
   GIT_SAMPLE_DATA_REPO_PATH   := $(DATA_DIR)/invest-sample-data
   GIT_SAMPLE_DATA_REPO_REV    := 0f8b41557753dad3670ba8220f41650b51435a93

Tests
-----

InVEST includes a battery of tests to ensure software quality.

Model tests
+++++++++++

To run tests on the suite of Ecosystem Service models in InVEST::

    $ make test

Tests depend on test data that is tracked in a ``git-lfs`` repo defined in the ``Makefile``::

   GIT_TEST_DATA_REPO          := https://bitbucket.org/natcap/invest-test-data.git
   GIT_TEST_DATA_REPO_PATH     := $(DATA_DIR)/invest-test-data
   GIT_TEST_DATA_REPO_REV      := 324abde73e1d770ad75921466ecafd1ec6297752

Test data (and Sample Data) can be retrieved using::

   $ make fetch


Changing how GNU make runs tests
++++++++++++++++++++++++++++++++

The InVEST Makefile setup depends on ``pytest`` and ``coverage`` to display
line coverage and produce HTML and XML reports.  You can force ``make`` to use
``coverage`` with a different test runner by setting a parameter at the
command line.  For example, to run the tests with ``nose``::

    $ make TESTRUNNER=nose test


Running tests on binaries
+++++++++++++++++++++++++++++++++++

This repository includes a python script to automatically
execute and check the exit status of all InVEST models, running on the
installed InVEST sample data. Once all sample data have been fetched
and binaries built on the target computer::

    $ make invest_autotest


Copyright and license information
---------------------------------

A file called ``LICENSE.txt`` should have accompanied this distribution.  If it
is missing, the license may be found on our project page,
https://github.com/natcap/invest
