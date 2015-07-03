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

Building documentation is supported in several ways, and is best done through a virtualenv. 
The primary way to build documentation is through the ``paver build_docs`` command.  If you
are trying to build the API documentation, you'll need to have the ``natcap.invest`` package
available to the python instance you're running paver with.  

On Linux, this might look like: ::

    #!/bin/sh
    ENV=doc_env
    paver env --clear \           # clear out an existing env if it already exists
        --system-site-packages \  # Grant the new env access to the system python
        --with-invest \           # Install natcap.invest to the new repo
        --envdir=$ENV             # Create the env at this dir.
        -r requirements-docs.txt  # Install the docs requirements as well.
    source $ENV/bin/activate
    paver build_docs

On Windows, this might look like: ::
    
    :: build_docs.bat
    :: Example batch file for building documentation in a virtualenv
    ::

    set ENV=doc_env
    paver env --clear --system-site-packages --with-invest --envdir=%ENV% -r requirements-docs.txt
    call %ENV%\Scripts\activate.bat
    paver build_docs
    
The ``paver build_docs`` command has these options: ::

    Usage: paver pavement.build_docs [options]

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


*Dependencies on Debian Systems*

 * ``sudo apt-get install libhdf5-dev && sudo pip install h5py`` Install HDF5.
 * ``sudo pip install --upgrade sphinxcontrib-napoleon`` We use the Napoleon theme for the API documentation.
 * ``sudo apt-get install python-setuptools``  Fixes some path issues with setuptools (see https://bitbucket.org/pypa/setuptools/issue/368/module-object-has-no-attribute-packaging)



Developing InVEST
=================


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


