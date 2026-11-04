.. _installing:

====================================
Installing the InVEST Python Package
====================================

.. attention::
     Most users will want to install the InVEST Workbench app (Mac disk image or Windows
     installer) from the `InVEST download page <https://naturalcapitalalliance.stanford.edu/software/invest>`_.
     The instructions here are for advanced users who wish to install the
     standalone python package, which does not include the Workbench GUI.

The ``natcap.invest`` python package is available for installation from three
distribution channels:

1. As a precompiled binary in the ``conda-forge`` ecosystem (`conda-forge <https://anaconda.org/conda-forge/natcap.invest>`_, `feedstock <https://github.com/conda-forge/natcap.invest-feedstock>`_)
2. As a binary wheel or source package from the `Python Package Index <https://pypi.org/project/natcap.invest/>`_
3. As a source installation, directly from our `git source tree <https://github.com/natcap/invest/>`_


Installing from ``conda-forge``
-------------------------------
The ``conda-forge`` distribution is the easiest way to install ``natcap.invest``
and its dependencies. The `conda <https://docs.conda.io/en/latest/miniconda.html>`_
and `mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_
package managers are able to install the underlying GDAL libraries that
``natcap.invest`` requires. ``conda-forge`` builds are published for every
release of ``natcap.invest``.::

    # conda and mamba can be used interchangeably.
    # We recommend mamba because it's faster.
    mamba install natcap.invest
    conda install natcap.invest

    # It's usually not necessary to specify the conda-forge
    # channel, but you can do so like this
    mamba install -c conda-forge natcap.invest
    conda install -c conda-forge natcap.invest


Installing from a pre-built wheel
---------------------------------
Pre-built wheels are available on the
`Python Package Index <https://pypi.org/project/natcap.invest/>`_. We provide
wheels for 64-bit x86 architectures and for each currently supported python
version at the time of release. If you have a computer with a non-x86-compatible
architecture (such as a Mac with one of the M-series chips), you will either
need to install from ``conda-forge`` (see above) or install from source (see below).

Our pre-built wheels are pinned to require a specific minor version of GDAL.
This is to ensure compatibility between the libgdal version in the runtime
environment and the libgdal version that ``natcap.invest`` and its dependencies
were built against. If you need to use a different GDAL version, you will
have to install from ``conda-forge`` (see above) or install from source (see below).

``natcap.invest`` depends on underlying GDAL libraries, which cannot be
installed by python-only package managers like ``pip`` and ``uv``. You must
install the correct minor version of GDAL before installing a ``natcap.invest``
wheel; otherwise, it will fail with a compilation error. We recommend ``conda`` as
the easiest way to install GDAL. See https://gdal.org/download.html for
details and alternative options.

To install with the ``uv`` package manager::

    mamba install gdal==3.10
    uv pip install natcap.invest

To install with ``pip``::

    mamba install gdal==3.10
    pip install natcap.invest

Note that if you run these commands in an environment for which a pre-built
wheel is not available, it will trigger an install from source (see below).


Installing from source
----------------------
You may need to install from source if you

* want to install the latest unreleased changes
* are using a python version for which a pre-built wheel or conda-forge distribution are not available
* are using a platform for which a pre-built wheel is not available

GDAL must already be installed in your environment as described above.
Additionally, the ``natcap.invest`` wheel build process depends on
``pygeoprocessing``, which requires the same minor version of GDAL that the
``pygeoprocessing`` wheel was built against. This can be difficult or
impossible to manage in the isolated build environment.

If using the ``uv`` package manager, our ``pyproject.toml`` configuration
ensures that ``pygeoprocessing`` is built from source, which guarantees that
the same libgdal version is used::

    uv pip install natcap.invest  # from PyPI source distribution
    uv pip install git+https://github.com/natcap/invest.git  # from remote source tree

If using ``pip``, it is usually necessary to disable build isolation to avoid
libgdal version conflicts. You will need to manually install the build
dependencies, then run::

    pip install --no-build-isolation natcap.invest  # from PyPI source distribution
    pip install --no-build-isolation git+https://github.com/natcap/invest.git  # from remote source tree

.. note::
    To install the ``natcap.invest`` package from source, you must have a C/C++
    compiler installed and configured for your system. MacOS and Linux users
    should not need to do anything. Windows users should install Microsoft
    Visual Studio, or at least the Build Tools for Visual Studio, if they have
    not already. See the `python wiki page on compilation under Windows <https://wiki.python.org/moin/WindowsCompilers>`_ for more information.


Errors Compiling GDAL
+++++++++++++++++++++

If you see one of these errors, you need to install GDAL:

* ``FileNotFoundError: [Errno 2] No such file or directory: 'gdal-config'``
* ``gdal_config_error: [Errno 2] No such file or directory: 'gdal-config'``

It is also possible to have a compatible version of GDAL installed, but for
``pip`` to try to compile the GDAL bindings anyways.  This happens when a newer
version of the GDAL python bindings are available on the Python Package Index
than are installed on your system.  The error message you see will look
something like this::

        bool CPL_DLL GDALDatasetAddFieldDomain(GDALDatasetH hDS,
                     ^
        fatal error: too many errors emitted, stopping now [-ferror-limit=]
        1 warning and 20 errors generated.
        error: command '/usr/bin/clang' failed with exit code 1
        [end of output]

    note: This error originates from a subprocess, and is likely not a problem with pip.
    ERROR: Failed building wheel for GDAL
    Running setup.py clean for GDAL

To work around this error when you have a compatible version of GDAL already installed,
instruct ``pip`` to only upgrade packages if needed::

    pip install natcap.invest --upgrade-strategy=only-if-needed

Dependencies
------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt`` and
included here for your reference.  Your package manager should, under most
circumstances, handle the dependency resolution for you.

.. include:: ../../requirements.txt
    :literal:
    :start-line: 12
