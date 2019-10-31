.. _installing:

=================
Installing InVEST
=================


.. _BinaryDependencies:

Binary Dependencies
-------------------

InVEST itself depends only on python packages, but many of these package
dependencies depend on low-level libraries or have complex build processes.
In recent history, some of these packages (notably, numpy and scipy) have
started to release precompiled binary packages of their own, removing the
need to install these packages through a system package manager.  Others,
however, remain easiest to install through a package manager.

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

    $ sudo apt-get install python3-dev python3-setuptools python3-gdal python3-rtree python3-shapely python3-matplotlib


Fedora
^^^^^^

::

    $ sudo yum install python3-devel python3-setuptools python3-gdal python3-rtree python3-shapely python3-matplotlib

.. _InstallingOnMac:

Mac OS X
********

The easiest way to install binary packages on Mac OS X is through a package
manager such as `Homebrew <http://brew.sh>`_::

    $ brew install gdal spatialindex pyqt matplotlib

The GDAL, PyQt and matplotlib packages include their respective python packages.
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
``pip install natcap.invest``.  Several of these dependencies are available
as extras, however, and can be installed via pip::

    $ pip install natcap.invest[ui]

These extras do not include a distribution of PyQt, so you will need to
install PyQt in an appropriate way on your system. PyQt4 is not currently
available from the Python Package Index, but other sources and package managers
allow for straightforward installation on :ref:`InstallingOnWindows`,
:ref:`InstallingOnMac`, and :ref:`InstallingOnLinux`.

The InVEST user interface uses a wrapper layer to support both PyQt4 and PyQt5,
one of which must be installed on your system for the UI to be able to run.
If both are installed, PyQt5 is preferred, but you can force the UI to use PyQt4
by defining an environment variable before launching the UI::

    $ QT_API=pyqt4 invest pollination

We have had the best luck running the UI under PyQt4 and PySide2.


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

    > pip install .\natcap.invest-3.3.0.post89+nfc4a8d4de776-cp27-none-win32.whl


Installing from our source tree
*******************************

The latest development version of InVEST can be installed from our
Mercurial source tree::

    $ pip install hg+https://bitbucket.org/natcap/invest@develop

