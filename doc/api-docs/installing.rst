.. _installing:

======================================
Installing InVEST From Source or Wheel
======================================

.. attention::
     Most users will want to install the InVEST app (Mac disk image or Windows 
     installer) from the `InVEST download page <https://naturalcapitalproject.stanford.edu/software/invest>`_.
     The instructions here are for more advanced use cases.


.. _BinaryDependencies:

Binary Dependencies
-------------------

InVEST itself depends only on python packages, but many of these package
dependencies depend on low-level libraries or have complex build processes.
Some of these packages (notably, numpy and scipy) have started to release 
precompiled binary packages of their own. Recently we have had success 
installing all dependencies through ``conda`` and ``pip``; however you may 
find it easier to install some through a system package manager.


Conda 
--------------

If you're using a conda environment to manage your ``natcap.invest`` installation,
it's easiest to install a few binary packages first before using pip to install
the rest::

    $ conda install "gdal>=3" numpy shapely rtree
    $ pip install natcap.invest


System Package Managers
-----------------------

.. _InstallingOnLinux:

Linux
*****

Linux users have it easy, as almost every package required to use
natcap.invest is available in the package repositories. The provided
commands will install only the libraries and binaries that are needed, allowing
``pip`` to install the rest.


Ubuntu & Debian
^^^^^^^^^^^^^^^

.. attention::
    The package versions in the debian:stable repositories often lag far
    behind the latest releases.  It may be necessary to install a later
    version of a library from a different package repository, or else build
    the library from source.


::

    $ sudo apt-get install python3-dev python3-setuptools python3-gdal python3-rtree python3-shapely


Fedora
^^^^^^

::

    $ sudo yum install python3-devel python3-setuptools python3-gdal python3-rtree python3-shapely

.. _InstallingOnMac:

Mac OS X
********

The easiest way to install binary packages on Mac OS X is through a package
manager such as `Homebrew <http://brew.sh>`_::

    $ brew install gdal spatialindex pyqt

The GDAL and PyQt packages include their respective python packages.
The others will allow their corresponding python packages to be compiled
against these binaries via ``pip``.

.. _InstallingOnWindows:

Windows
*******

While many packages are available for Windows on the Python Package Index, some
may need to be fetched from a different source. Many are available from
Christogh Gohlke's unofficial build page:
http://www.lfd.uci.edu/~gohlke/pythonlibs/

PyQt4 installers can also be downloaded from the `Riverbank Computing website <https://www.riverbankcomputing.com/software/pyqt/download>`_.



Python Dependencies
-------------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt``:

.. include:: ../../requirements.txt
    :literal:
    :start-line: 9




Optional Qt User Interface
--------------------------

InVEST's user interface is built with PyQt.  Because of the hefty binary
requirement of Qt and the relative difficulty of installing PyQt, these
dependencies will not be installed with the standard
``pip install natcap.invest``.  These dependencies are available
as extras, however, and can be installed via pip::

    $ pip install natcap.invest[ui]


.. _installing-from-source:

Installing from Source
----------------------

.. note::

    Python 3.6 users will need to install Microsoft Visual Studio 2017, or at
    least the Build Tools for Visual Studio 2017.
    See the `python wiki page on compilation under Windows <https://wiki.python.org/moin/WindowsCompilers>`_
    for more information.

Assuming you have a C/C++ compiler installed and configured for your system, and
dependencies installed, the easiest way to install InVEST as a python package
is::

    $ pip install natcap.invest


Installing the latest development version
-----------------------------------------


Pre-built binaries for Windows
******************************

Pre-built installers and wheels of development versions of ``natcap.invest``
for 32-bit Windows python installations are available from
http://releases.naturalcapitalproject.org/?prefix=invest/, along with other
distributions of InVEST.  Once downloaded, wheels can be installed locally via
pip.


Installing from our source tree
*******************************

The latest development version of InVEST can be installed from our
git source tree if you have a compiler installed::

    $ pip install "git+https://github.com/natcap/invest@master#egg=natcap.invest"

