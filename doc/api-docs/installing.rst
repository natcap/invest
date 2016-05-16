.. _installing:

=================
Installing InVEST
=================

.. note::

    The ``natcap.invest`` python package is currently only supported in Python
    2.7.  Other versions of python may be supported at a later date.

.. warning::

    Python 2.7.11 or later is required to be able to use the InVEST
    Recreation model on Windows.


Binary Dependencies
-------------------

InVEST itself depends only on python packages, but many of these package 
dependencies depend on low-level libraries or have complex build processes.
In recent history, some of these packages (notably, numpy and scipy) have
started to release precompiled binary packages of their own, removing the
need to install these packages through a system package manager.  Others,
however, remain easiest to install through a package manager.


Linux
*****

Linux users have it easy, as almost every package required to use
natcap.invest is available in the package repositories. The provided
commands will install only the libararies and binaries that are needed, allowing
``pip`` to install the rest.


Ubuntu & Debian
^^^^^^^^^^^^^^^

.. attention::
    The package versions in the debian:stable repositories often lag far
    behind the latest releases.  It may be necessary to install a later
    version of a libarary from a different package repository, or else build
    the library from source.


::

    $ sudo apt-get install python-setuptools python-yaml python-gdal python-h5py python-rtree python-shapely python-matplotlib python-qt4


Fedora
^^^^^^

::

    $ sudo yum install python-setuptools libyaml gdal-python h5py python-rtree python-shapely python-matplotlib PyQt4
   


Mac OS X
********

The easiest way to install binary packages on Mac OS X is through a package
manager such as `Homebrew <http://brew.sh>`_::

    $ brew install gdal libyaml hdf5 spatialindex pyqt matplotlib

The GDAL, PyQt and matplotlib packages include their respective python packages.
The others will allow their corresponding python packages to be compiled
against these binaries via ``pip``.


Windows
*******

While many packages are available for Windows on the Python Package Index, some
may need to be fetched from a different source. Many are available from
Christogh Gohlke's unofficial build page: 
http://www.lfd.uci.edu/~gohlke/pythonlibs/


Python Dependencies
-------------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt``:

.. include:: ../../requirements.txt
    :literal:
    :start-line: 9

Additionally, ``PyQt4`` is required to use the ``invest`` cli, but is not
required for development against ``natcap.invest``.


Installing from Source
----------------------

Assuming you have a C/C++ compiler installed and configured for your system, and
dependencies installed, the easiest way to install InVEST as a python package 
is::

    $ pip install natcap.invest

If you are working within virtual environments, there is a `documented issue
with namespaces 
<https://bitbucket.org/pypa/setuptools/issues/250/develop-and-install-single-version>`_
in setuptools that may cause problems when importing packages within the
``natcap`` namespace.  The current workaround is to use these extra pip flags::

    $ pip install natcap.invest --egg --no-binary :all:


Installing the latest development version
-----------------------------------------


Pre-built binaries for Windows
******************************

Pre-built installers and wheels of development versions of ``natcap.invest``
for 32-bit Windows python installations are available from
http://data.naturalcapitalproject.org/invest-releases/#dev, along with other
distributions of InVEST.  Once downloaded, wheels can be installed locally via
pip::

    > pip install .\\natcap.invest-3.3.0.post89+nfc4a8d4de776-cp27-none-win32.whl


Installing from our source tree
*******************************

The latest development version of InVEST can be installed from our source tree::

    $ pip install hg+https://bitbucket.org/natcap/invest@develop

