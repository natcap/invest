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

.. note::
    **This repository is for InVEST 3.2.1 and later**

    This replaces our Google Code project formerly
    located at http://code.google.com/p/invest-natcap.  If you are looking to build
    InVEST <= 3.2.0, use the archived release-framework repository at
    https://bitbucket.org/natcap/invest-natcap.release-framework, and the InVEST repository
    at https://bitbucket.org/natcap/invest-natcap.invest-3.


InVEST Dependencies
===================
InVEST relies on the following python packages:
  * GDAL
  * shapely
  * numpy
  * scipy
  * poster
  * psycopg2
  * pyqt4
  * matplotlib
  * bs4
  * python-dateutil
  * pyparsing
  * six
  * pyAMG
  * pillow
  * cython

For development, we recommend using a virtual environment (such as provided by
``virtualenv``).

Binaries are generated with ``pyinstaller``.

To work with this repository, you'll also need these tools to be available
on the command-line somewhere on your PATH:

  * ``hg`` (Mercurial, for cloning repositories)
  * ``make`` (GNU Make, for building documentation)
  * ``fpm`` (for generating .deb and .rpm packages, can be installed via ``gem``)
  * ``pyinstaller`` (for generating binaries)
  * ``makensis`` (NSIS, for generating Windows installers)
  * ``hdiutil`` (for generating mac DMGs)


For building InVEST binaries, you will also need to have a compiler configured.
On linux, gcc/g++ will be sufficient.  On Windows, MinGW and MSVC work.  On Mac,
you'll likely need the XCode command-line tools to be installed.


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
something like this:

::
    :: build_docs.bat
    :: Example batch file for building documentation in a virtualenv
    ::

    set ENV=doc_env
    paver env --clear --system-site-packages --with-invest --envdir=%ENV% -r requirements-docs.txt
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
        -r requirements-docs.txt
    source $ENV/bin/activate
    paver build_docs


*Dependencies on Debian Systems*

 * ``sudo apt-get install libhdf5-dev && sudo pip install h5py`` Install HDF5.
 * ``sudo pip install --upgrade sphinxcontrib-napoleon`` We use the Napoleon theme for the API documentation.
 * ``sudo apt-get install python-setuptools``  Fixes some path issues with setuptools (see https://bitbucket.org/pypa/setuptools/issue/368/module-object-has-no-attribute-packaging)



Developing InVEST
=================

To use `python setup.py develop` during development, you must symbolically link the src/natcap directory to the top-level directory of the repository with `ln -s src/natcap natcap` (from the top-level directory).


Contributing to Development
===========================

Issues, including ongoing work, are tracked in our issue tracker on this bitbucket project.  If you encounter a bug, please let us know!

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


Releasing InVEST
================
This repository uses paver as a single entry point for common distribution needs.
Run ``paver help`` for a list of commands provided by this repository's pavement.py.

Note that while paver can in some cases replace a classic setup.py, this repository
has its own setup.py file already created.  We therefore do not use this part of the
paver functionality.


