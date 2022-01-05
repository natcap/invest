.. _installing:

====================================
Installing the InVEST Python Package
====================================

.. attention::
     Most users will want to install the InVEST app (Mac disk image or Windows 
     installer) from the `InVEST download page <https://naturalcapitalproject.stanford.edu/software/invest>`_.
     The instructions here are for more advanced use cases.

.. note::
    To install the ``natcap.invest`` package, you must have a C/C++ compiler
    installed and configured for your system. MacOS and Linux users should
    not need to do anything. Windows users should install Microsoft Visual
    Studio, or at least the Build Tools for Visual Studio, if they have
    not already. See the `python wiki page on compilation under Windows <https://wiki.python.org/moin/WindowsCompilers>`_ for more information.

Suggested method
----------------

**Pattern**::

    conda create -y -c conda-forge -n <name> python=<python version>
    conda activate <name>
    conda install -y -c conda-forge gdal=<gdal version>
    pip install natcap.invest==<invest version>

Replace ``<name>`` with any name you'd like to give your environment.
Replace ``<python version>`` with a python version known to be compatible with the desired invest version.
Replace ``<gdal version>`` with a GDAL version known to be compatible with the desired invest version.
Replace ``<invest version>`` with the desired invest version.

Most of the time, it is not really necessary to specify the versions of ``python``, ``gdal``, and ``natcap.invest``. If you do not specify a version, the latest version will be installed. Usually the latest versions are compatible with each other, but not always. Specifying versions that are known to work can prevent some problems. You can find the supported range of GDAL versions in the `requirements.txt <https://github.com/natcap/invest/blob/main/requirements.txt>`_ (be sure to switch to the desired release tag in the dropdown).

**Example for InVEST 3.9.1**::

    conda create -y -c conda-forge -n invest391 python=3.9.7
    conda activate invest391
    conda install -y -c conda-forge gdal=3.3.1
    pip install natcap.invest==3.9.1


**Condensed into one line**::

    conda create -y -c conda-forge -n invest391 python=3.9.7 gdal=3.3.1 && conda activate invest391 && pip install natcap.invest==3.9.1


Details
-------
Here is an explanation of what the commands are doing:

1. Create a brand-new environment with the correct python version.

   ``conda create -y -c conda-forge -n <name> python=<python version>``

   To be safe, you should **always install** ``natcap.invest`` **into a brand-new virtual environment**. This way you can be sure you have all the right versions of dependencies. Many issues with installing or using the ``natcap.invest`` package arise from dependency problems, and it's a lot easier to create a new environment than it is to fix an existing one.

2. Activate the brand-new environment just created.

   ``conda activate <name>``

   If you run ``conda list`` after this, you'll see the specified python version is there along with around 15 other packages that are included with python by default. None of these are specific to invest. You're now in an isolated environment so you can control which versions of dependencies are available to invest.

3. Install GDAL before installing invest

   ``conda install -y -c conda-forge gdal=<gdal version>``

   This is important because GDAL is not an ordinary python package. When you install the ``natcap.invest`` package in step 4, ``pip`` will also install all the dependencies of ``natcap.invest``. When ``pip`` tries to install GDAL, you will get an error unless the underlying GDAL binaries are already installed. That's because the ``gdal`` package that ``pip`` installs is just a python wrapper that depends on the GDAL binaries. GDAL itself is not a python package and can't be installed with ``pip``. Luckily, it can be installed with ``conda``!

4. Install invest

   ``pip install natcap.invest=<invest version>``

   ``pip`` will also install the correct versions of all dependencies of ``natcap.invest``.

   Since sometimes we don't need to use the UI at all, the basic ``natcap.invest`` package does not include the dependencies required for the UI. If you try to use the UI without having installed the UI dependencies, you'll get an error. If you do want to use the invest UI via the python package, install ``natcap.invest`` with the UI package extra: ::

      pip install natcap.invest[ui]=<invest version>

   The ``[ui]`` tells ``pip`` to also install all the dependencies needed for the UI.


Python Dependencies
-------------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt``:

.. include:: ../../requirements.txt
    :literal:
    :start-line: 14

Additional dependencies for the UI are listed in ``requirements-gui.txt``:

.. include:: ../../requirements-gui.txt
    :literal:
    :start-line: 9

Please use ``conda`` and ``pip`` to install the correct versions of these dependencies automatically as described above.

.. _BinaryDependencies:

Binary Dependencies
-------------------

InVEST itself depends only on python packages, but many of these package
dependencies, such as numpy, scipy, and GDAL, depend on low-level libraries
or have complex build processes. Precompiled binaries of all these dependencies
are now available through ``conda`` and/or ``pip``. We recommend using ``conda``
to manage these dependencies because it simplifies the install process and
helps ensure versions are compatible. However, they may also be available through
your system package manager.

Installing the latest development version
-----------------------------------------

Pre-built binaries
******************

Pre-built installers and wheels of development versions are available from
http://releases.naturalcapitalproject.org/?prefix=invest/, along with other
distributions of InVEST. Once downloaded, wheels can be installed locally via
pip.


Installing from source
**********************

The latest development version of InVEST can be installed from our
git source tree if you have a compiler installed::

    $ pip install "git+https://github.com/natcap/invest@main#egg=natcap.invest"
