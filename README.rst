InVEST: Integrated Valuation of Ecosystem Services and Tradeoffs
================================================================

+-----------------------+-------------------------------+
| Build type            | Windows                       |
+=======================+===============================+
| Nightly Binary builds | |nightly_binary_build_badge|  |
+-----------------------+-------------------------------+
| Dev builds            | |dev_windows_build_badge|     |
+-----------------------+-------------------------------+
| Tests                 | |windows_test_badge|          |
+-----------------------+-------------------------------+
| Test coverage         | |windows_test_coverage_badge| |
+-----------------------+-------------------------------+

.. |nightly_binary_build_badge| image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=invest-nightly-develop
  :target: http://builds.naturalcapitalproject.org/job/invest-nightly-develop

.. |dev_windows_build_badge| image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=natcap.invest/label=GCE-windows-1
  :target: http://builds.naturalcapitalproject.org/job/natcap.invest/label=GCE-windows-1

.. |windows_test_badge| image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=test-natcap.invest/label=GCE-windows-1
  :target: http://builds.naturalcapitalproject.org/job/test-natcap.invest/label=GCE-windows-1

.. |windows_test_coverage_badge| image:: http://builds.naturalcapitalproject.org:9931/jenkins/c/http/builds.naturalcapitalproject.org/job/test-natcap.invest/label=GCE-windows-1/
  :target: http://builds.naturalcapitalproject.org/job/test-natcap.invest/label=GCE-windows-1


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

.. note::
    **This repository is for InVEST 3.2.1 and later**

    This replaces our Google Code project formerly
    located at http://code.google.com/p/invest-natcap.  If you are looking to build
    InVEST <= 3.2.0, use the archived release-framework repository at
    https://bitbucket.org/natcap/invest-natcap.release-framework, and the InVEST repository
    at https://bitbucket.org/natcap/invest-natcap.invest-3.


Contributing to Development
===========================

The best way to participate is to contribute a fix to a bug you are
experiencing or to implement a pet feature!

Issues, including ongoing work, are tracked in our issue tracker on this
bitbucket project.  If you encounter a bug, please let us know!

If you have something you'd like to contribute, please fork the repository
and submit a pull request.  Since mercurial tracks branch names in the metadata
of each commit, please be sure to make a feature branch off of ``develop``.  For example: ::

    hg up develop
    hg branch feature/<my-new-branch>

``<my-new-branch>`` would be a short string describing your branch.  It need not be long :).
Adhering to proper branching will help us retain a descriptive history as the project
matures and will also help greatly with the pull request review process.

As always, be sure to add a note about your change to the HISTORY file before
submitting your PR.

*Thanks for contributing!*

InVEST Dependencies
===================
.. note::
    Do this:

    ``$ paver check``

    This will verify required applications are available and that
    you have some of the python packages that are more difficult to install
    (especially those that depend on low-level C libraries).

InVEST relies on the following python packages:
  * GDAL
  * shapely
  * numpy
  * scipy
  * pyqt4  *(if running a model user interface)*
  * matplotlib
  * bs4
  * pyAMG
  * cython
  * setuptools
  * dbfpy
  * poster
  * pygeoprocessing
  * natcap.versioner
  * rtree

For development, we recommend using a virtual environment (such as provided by
``virtualenv``).  We provide a paver command (``paver env``) to help with this process.
See `Building Binaries`_ for an example.

Binaries are generated with ``pyinstaller`` via ``paver build_bin``.  See `Building Binaries`_.

To work with this repository, you'll also need these tools to be available
on the command-line somewhere on your PATH, depending on what you'd like to build:

  * ``hg`` (Mercurial, for cloning repositories)
  * ``make`` (GNU Make, for building documentation)
  * ``fpm`` (for generating .deb and .rpm packages, can be installed via ``gem``)
  * ``makensis`` (NSIS, for generating Windows installers)
  * ``hdiutil`` (for generating mac DMGs)
  * ``pdflatex`` (for generating PDF versions of the User's Guide)
  * ``pandoc`` (for converting .docx files to rst when building the User's
    Guide.  See http://pandoc.org/installing.html)


For building InVEST binaries, you will also need to have a compiler configured.
On linux, gcc/g++ will be sufficient.  On Windows, MinGW and MSVC work.  On Mac,
you'll likely need the XCode command-line tools to be installed.


Building Binaries
=================

One-Step Binary Builds
----------------------
The easiest way to build binaries is to call ``paver build``.  If your system
is properly configured, this will do all of the heavy lifting to:

    + Clone any hg, git, and svn repositories needed for the given steps
    + Set up a virtual environment with needed package dependencies (skip with
      ``--python=<your python interpreter here>``
    + Build binaries out of the virtual environment (skip with ``--skip-bin``)
    + Build User's Guide documentation (HTML, PDF) (skip with ``--skip-guide``)
    + Build InVEST API documentation (HTML) (skip with ``--skip-api``)
    + Build archives of sample data (skip with ``--skip-data``)
    + Build a system-appropriate installer (skip with ``--skip-installer``)

Assembled binaries are placed in ``dist/release_invest-<version>`` with the
following directory structure: ::

    dist/
        natcap.invest-<version>.tar.gz          # Python source distribution
        release_invest-<version>/
            data/
                # All data zipfiles available for this version
            documentation/
                # HTML documentation for InVEST
            invest-<version>-apidics.zip        # Archived HTML API documentation
            invest-<version>-userguide.zip      # Archived HTML User's Guide
            InVEST_<version>_Documentation.pdf  # PDF User's Guide
            invest-<version>.deb                # Debian dpkg
            invest-<version>.rpm                # RPM package
            InVEST_<version>_Setup.exe          # Windows installer
            InVEST <version>.dmg                # Mac disk image

.. note::
    ``paver build`` will only build binaries and and installer for the system
    you are running.



Just building binaries
----------------------
The easiest way to build pyinstaller binaries on your platform is to use our
one-step binary build.  This paver task will
Binaries are built through ``paver build_bin``.  The simplest way to call this is
``paver build_bin``, but this assumes that you have all dependencies (including natcap.invest)
installed to your global python distribution.  More commonly, you'll want to install InVEST to
a virtual environment before running build_bin.

For example, if you want to build a new virtualenv via the paver command and then build the binaries
using this new environment:

::

    #!/bin/sh
    # Example for linux or mac

    $ ENVNAME=release_env
    $ paver env \
        --system-site-packages \
        --clear \
        --envname=$ENVNAME \
        --with-invest

    $ paver build_bin --python=release_env/bin/python

This will build the pyinstaller binaries for whatever platform you're running this on and place them
into ``dist/invest_dist``.  Console files will also be written to this folder, one for each model in InVEST.
These console files simply call the ``invest`` binary with the corresponding InVEST modelname.  For example,
the console files for Habitat Risk Assessment would look like:

**Windows:** ``dist\invest_dist\invest_hra.bat`` ::

    .\invest.exe hra

**Linux/Mac:** ``dist/invest_dist/invest_hra.sh`` ::

    ./invest hra

InVEST currently uses a single CLI entry point, an executable within ``dist/invest-dist``.  This exe is not
sensitive to your CWD, so if the binary (or a symlink to the binary) is available on your system PATH, you
should be able to execute it like so: ::

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
                  models/tools.

    optional arguments:
      -h, --help  show this help message and exit
      --version   show program's version number and exit
      --list      List available models

On Windows, running ``invest.exe`` will also prompt you for user input if a modelname is not provided.


Building Data Zipfiles
======================

Building data zipfiles is done by calling ``paver build_data``: ::

    Options:
      -h, --help   display this help information
      --force-dev  Zip data folders even if repo version does not match the known
      state


      Build data zipfiles for sample data.

      Expects that sample data zipfiles are provided in the invest-data repo.
      Data files should be stored in one directory per model, where the directory
      name matches the model name.  This creates one zipfile per folder, where
      the zipfile name matches the folder name.

      options:
      --force-dev : Provide this option if you know that the invest-data version
                    does not match the version tracked in versions.json.  If the
                    versions do not match and the flag is not provided, the task
                    will print an error and quit.


This will build the data zipfiles and store them in ``dist``.


Building Documentation
======================

All documentation is built through ``paver build_docs`` via sphinx.  Building
the User's Guide requires that you have GNU make, sphinx, and LaTex installed.
Building the API documentation requires only virtualenv and a compiler, as
sphinx will be installed into a new virtualenv at build time.

The ``paver build_docs`` command has these options: ::

    Usage: paver build_docs [options]

    Options:
      -h, --help    display this help information
      --force-dev   Force development
      --skip-api    Skip building the API docs
      --skip-guide  Skip building the User's Guide


      Build the sphinx user's guide for InVEST.

      Builds the sphinx user's guide in HTML, latex and PDF formats.
      Compilation of the guides uses sphinx and requires that all needed
      libraries are installed for compiling html, latex and pdf.

      Requires make for the user's guide
      The API docs requires sphinx and setuptools only.

Note that building API documentation via ``paver build_docs`` is only currently supported
on POSIX systems.  Documentation can still be built on Windows, but you'll need to run
something like this: ::

    :: build_docs.bat
    :: Example batch file for building documentation in a virtualenv
    ::

    set ENV=doc_env
    paver env --clear --system-site-packages --with-invest --envdir=%ENV% -r requirements-dev.txt
    call %ENV%\Scripts\activate.bat
    paver build_docs

On Linux or Mac, setting up a virtual environment to be able to build documentation
look like this: ::

    #!/bin/sh
    ENV=doc_env
    paver env --clear \
        --system-site-packages \
        --with-invest \
        --envdir=$ENV
        -r requirements-dev.txt
    source $ENV/bin/activate
    paver build_docs


Building Installer
==================

Our paver configuraton supports 4 different installer types: ::

    NSIS (Windows executable installer)
    DMG  (Mac Disk Imagage)
    DEB  (Debian binary package)
    RPM  (RPM Package Manager binary package)

I suppose it's probably possible to cross-compile binaries for other platforms, but I wouldn't promise that
it will work.  Try at your own risk!

To build an installer, you'll first need to build the InVEST binary folder through ``paver build_bin``.
Under normal conditions, this will save your binaries to ``dist/invest_dist``.  To build an installer
from this folder, execute ::

    $ paver build_installer --bindir=dist/invest_dist

If the ``--insttype`` flag is not provided, the system default will be used.  System defaults are:

 * Linux: ``deb``
 * Mac: ``dmg``
 * Windows: ``nsis``


Developing InVEST
=================

Debian Systems
--------------

.. note::
    **Debian builds require GLIBC >= 2.15**

    Pyinstaller builds using a recent enough version of ``libpython2.7`` require that you have
    GLIBC >= 2.15, which is available on Debian Jessie (8), or on Wheezy (7) through the testing
    APT repository.


Specific package dependencies include:

 * ``sudo apt-get install python-gdal``
 * ``sudo apt-get install python-matplotlib``
 * ``sudo apt-get install libgeos-dev python-dev``
 * ``sudo apt-get install python-qt4`` Install PyQt4
 * ``sudo pip install --upgrade sphinxcontrib-napoleon`` We use the Napoleon theme for the API documentation.
 * ``sudo apt-get install python-setuptools``  Fixes some path issues with setuptools (see https://bitbucket.org/pypa/setuptools/issue/368/module-object-has-no-attribute-packaging)


Mac Systems
-----------

The easiest way to set up your system is to install all binary dependencies through the Homebrew
package manager (http://brew.sh).

Setting up an InVEST virtual environment
----------------------------------------

Most likely, the easiest way to run InVEST from your source tree is to build a
virtual environment using the popular ``virtualenv``
(https://virtualenv.pypa.io/en/latest/).  This can be done manually, but there
is a paver task (``paver env``) to build up a virtual environment for you.  Here are a few
examples:  ::

    # Build an env with all dependencies installed only to this environment.
    # This does not install InVEST, just the dependencies.
    # The environment is created at test_env/
    $ paver env -e test_env

    # Build an env with access to system site-packages and also install InVEST
    $ paver env --system-site-packages --clear --with-invest -e test_env

    # You can also specify additional requirement to be installed with the -r
    # flag.
    $ paver env --sytem-site-packages -r requirements-dev.txt

natcap.versioner ImportError
----------------------------

Since June, 2015, we have been moving our python projects to the ``natcap``
package namespace and gradually publishing our projects on the Python Package
Index.  Unfortunately, using a namespace package does not appear to work quite
as seamlessly across multiple virtual python installations as one might hope.

A common example of this breakdown comes when trying to run ``python setup.py
install`` on the ``invest`` repository (this repository).  Example: ::

    $ python setup.py install
    Traceback (most recent call last):
      File "setup.py", line 19, in <module>
          import natcap.versioner
    ImportError: No module named natcap.versioner

To fix this, install ``natcap.versioner`` to the python environment that you're
trying to install ``natcap.invest`` to before calling natcap.invest's setup.py.
So if you're trying to install natcap.invest to your global site-packages,
install natcap.versioner there.  If you're trying to install natcap.invest to
your virtual environment, activate your virtual environment, ``pip install
natcap.versioner`` and then ``python setup.py install`` for natcap.invest.

**Using python setup.py develop for natcap.invest**

``python setup.py develop`` appears to have some odd behavior when trying to
import natcap.invest.  If you find that you need to import natcap.versioner
before you can import natcap.invest, do this: ::

    $ pip uninstall natcap.versioner
    $ pip install --egg natcap.versioner

`The relevant issue`_ on the python packaging authority's issue tracked has some
more information if you're interested.

.. _The relevant issue: https://bitbucket.org/pypa/setuptools/issues/250/develop-and-install-single-version#comment-19426088

Matplotlib ImportError
----------------------

On Fedora systems, some users encounter this exception when trying to run an
InVEST model that uses matplotlib:

::

    ...
    line 17, in <module>
        from .backend_qt5agg import NavigationToolbar2QTAgg
    ImportError: No module named backend_qt5agg

This is a `known issue`_ with the RedHat build of ``python-matplotlib-qt4``.  The workaround
is to ``yum install python-matplotlib-qt5``.

.. _known issue: https://bugzilla.redhat.com/show_bug.cgi?id=1219556


GDAL
----

InVEST relies on GDAL/OGR for its raster and vector handling.  This library is
usually available in your system's package index.

Debian: ``sudo apt-get install python-gdal``

Mac:  ``brew install gdal``

Installing GDAL on a windows computer is a little more complicated.  Christoph
Gohlke has prebuilt binaries for the Python GDAL package
(http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal), though these have often
given side-by-side configuration errors.  Use at your own risk.

An alternative is to install the GDAL binaries from here:
http://www.gisinternals.com/, and then install the GDAL python package
separately.  To install in this way:

  * Download and install the correct version of the GDAL binaries.
  * Add a ``GDAL_DATA`` environment variable pointing to the folder containing
    these installed binaries.

Then, download and install the gdal python package.

RTREE
-----

InVEST Coastal Vulnerability relies on the rtree package for spatial indexing
geometries. Rtree depends on the libspatialindex library from
http://libspatialindex.github.com.

To install on \*nix download the libspatialindex library and run:

    ``sudo ./configure; sudo make; sudo make install``

    ``sudo pip install rtree``

Installing on a windows computer is a little more complicated. Christoph
Gohlke has prebuilt binaries for the Rtree at
http://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree.

Please see the packages PYPI page https://pypi.python.org/pypi/Rtree/ for
more details as well as the installation instruction page
http://toblerity.org/rtree/install.html.


Running Tests
=============

To run the full suite of tests:

::

    $ paver test

To specify a test (or multiple tests) to run via `paver test`, use the nosetests
format to specify test files, classes, and/or test methods to run.  For example:

::

    $ paver test tests/test_example.py:ExampleTest.test_regression

This will only run this one test, ignoring all other tests that would normally be
run.

If you're looking for some extra verbosity (or you're building on jenkins):

::

    $ paver test --jenkins

You may also launch tests from the python shell:

::

    >>> import natcap.invest
    >>> natcap.invest.test()

Tests are implemented with ``unittest``, so any appropriate test runner should work.


Releasing InVEST
================
This repository uses paver as a single entry point for common distribution needs.
Run ``paver help`` for a list of commands provided by this repository's pavement.py.

Note that while paver can in some cases replace a classic setup.py, this repository
has its own setup.py file already created.  We therefore do not use this part of the
paver functionality.




