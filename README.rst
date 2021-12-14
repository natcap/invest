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
* Users' guide: http://data.naturalcapitalproject.org/nightly-build/invest-users-guide/html/
* API documentation: http://invest.readthedocs.io/en/latest/


Dependencies
------------

Run ``make check`` to test if all required dependencies are installed on your system.
OS-specific installation instructions are found either online at
http://invest.readthedocs.io/en/latest/installing.html or locally at ``doc/api-docs/installing.rst``.

.. note::
    The ``make`` commands for InVEST require a BASH shell environment. Windows
    users can use Git Bash within the Git for Windows suite. More infomration
    can be found at https://gitforwindows.org

NSIS-specific requirements
++++++++++++++++++++++++++
The InVEST NSIS installer requires the following:

* NSIS
* Installed Plugins:
    * Nsisunz: http://nsis.sourceforge.net/Nsisunz_plug-in
    * InetC: http://nsis.sourceforge.net/Inetc_plug-in
    * NsProcess: http://nsis.sourceforge.net/NsProcess_plugin

Managing python dependencies
++++++++++++++++++++++++++++
We recommend using a virtual environment to manage your python dependencies, and there is
a Makefile target to assist with this::

    $ make env
    $ source env/bin/activate

Or on Windows, use the following instead from a CMD prompt::

    > make env
    > .\env\bin\activate

This makefile target is included for convenience ... you may of course choose to
manage your own virtual environment.  ``requirements.txt``,
``requirements-dev.txt`` and ``requirements-gui.txt`` list the python
dependencies needed.

Using a different environment name
""""""""""""""""""""""""""""""""""
If you prefer a different name for your environment, you may pass the environment name as
a parameter to make::

    $ make ENV=myEnv env

You could then activate the environment created at ``myEnv``.


Using a different environment management tool
"""""""""""""""""""""""""""""""""""""""""""""
The InVEST Makefile uses ``virtualenv`` to set up an environment, but this is
not the only `environment management tool out there
<https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_.
You may elect to manage your virtual environment a different way, independent
of ``make env``.  The only requirement for the build process is that the required
tools are available on your PATH and the required python packages can be imported.


Building InVEST Distributions
-----------------------------

Once the required tools and packages are available, we can build InVEST.


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


InVEST Windows Installer
++++++++++++++++++++++++

The InVEST installer for Windows can be built with::

    > make windows_installer

This will create the installer at ``dist/InVEST_*_Setup.exe``.


InVEST Mac Disk Image
+++++++++++++++++++++

The InVEST disk image for Mac can be built with::

    $ make mac_installer

This will create the installed at ``dist/InVEST_*.dmg``.



Building InVEST Documentation
-----------------------------

User's Guide
++++++++++++

To build the user's guide::

    $ make userguide

This will build HTML and PDF documentation, writing them to ``dist/userguide``
and ``dist/InVEST_*_Documentation.pdf``, respectively.


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

Single archive of sample data
+++++++++++++++++++++++++++++

For trainings, it is especially convenient to distribute all sample data as a
single zip archive.  As an added bonus, this single zip archive can be provided
to the InVEST installer for Windows as either the 'Advanced' input on the front
page of the installer, or by a CLI flag, thus preventing the installer from
downloading datasets from the internet.  See
``installer/windows/invest_installer.nsi`` for more details.  To build a single
archive of all InVEST sample data::

    $ make sampledata_single

This will write the single sampledata archive to
``dist/InVEST_*_sample_data.zip``.


Tests
-----

InVEST includes a battery of tests to ensure software quality.

Model tests
+++++++++++

To run tests on the suite of Ecosytem Service models in InVEST::

    $ make test


User interface tests
++++++++++++++++++++

To run tests for user interface functionality::

    $ make test_ui


Changing how GNU make runs tests
++++++++++++++++++++++++++++++++

The InVEST Makefile setup depends on ``pytest`` and ``coverage`` to display
line coverage and produce HTML and XML reports.  You can force ``make`` to use
``coverage`` with a different test runner by setting a parameter at the
command line.  For example, to run the tests with ``nose``::

    $ make TESTRUNNER=nose test


Running tests on installed binaries
+++++++++++++++++++++++++++++++++++

The InVEST binaries for Windows include a python script to automatically
execute and check the exit status of all InVEST models, running on the
installed InVEST sample data.  This script requires Python version 2.7 to be on
the PATH.  Once InVEST and all sample data have been installed on the target
computer::

    > cd C:\InVEST_<version>_x86\invest-3-x86
    > .\invest-autotest.bat


Copyright and license information
---------------------------------

A file called ``LICENSE.txt`` should have accompanied this distribution.  If it
is missing, the license may be found on our project page,
https://github.com/natcap/invest
