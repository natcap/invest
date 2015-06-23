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
`virtualenv`).

Binaries are generated with `pyinstaller`.

To work with this repository, you'll also need these tools to be available 
on the command-line somewhere on your PATH:
* mercurial (for cloning repositories)
* make (for building documentation)
* fpm (for generating .deb and .rpm packages)
* pyinstaller (for generating binaries)
* NSIS (for generating Windows installers)
* hdiutil (for generating mac DMGs)


For building InVEST binaries, you will also need to have a compiler configured.
On linux, gcc/g++ will be sufficient.  On Windows, MinGW and MSVC work.  On Mac,
you'll likely need the XCode command-line tools to be installed.


Developing InVEST
=================

Fork and pull request, please!

Also, please be sure to work on a feature branch branched from develop.
Feature branches should be named `feature/your_feature_name`.

Also, be sure to add a note about your change to the HISTORY file before
submitting your PR.

Thanks for contributing!


Releasing InVEST
================
This repository uses paver as a single entry point for common distribution needs.
Run `paver help` for a list of commands provided by this repository's pavement.py.

Note that while paver can in some cases replace a classic setup.py, this repository
has its own setup.py file already created.  We therefore do not use this part of the
paver functionality.


