.. _installing:

====================================
Installing the InVEST Python Package
====================================

.. attention::
     Most users will want to install the InVEST app (Mac disk image or Windows
     installer) from the `InVEST download page <https://naturalcapitalproject.stanford.edu/software/invest>`_.
     The instructions here are for more advanced use cases.

The InVEST package is available for installation from three distribution channels:

1. As a precompiled binary in the ``conda-forge`` ecosystem (`conda-forge <https://anaconda.org/conda-forge/natcap.invest>`_, `feedstock <https://github.com/conda-forge/natcap.invest-feedstock>`_)
2. As a binary wheel or source package from the `Python Package Index <https://pypi.org/project/natcap.invest/>`_
3. As a source installation, directly from our `git source tree <https://github.com/natcap/invest/>`_

Note that installing the python package does not include a user interface.

Suggested Method: ``conda-forge``
---------------------------------

The easiest way to install ``natcap.invest`` is using the
`conda <https://docs.conda.io/en/latest/miniconda.html>`_ or
`mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_ package managers::

    mamba install -c conda-forge natcap.invest

If you prefer to use ``conda``, the command is otherwise the same::

    conda install -c conda-forge natcap.invest

Installing with ``pip``
-----------------------

The ``natcap.invest`` package is also available from the
`Python Package Index <https://pypi.org/project/natcap.invest/>`_
and installable with ``pip``.  Binary builds of ``natcap.invest`` are available for
64-bit x86 architectures and also as a source distribution.  Note that if you
have a computer with a non-x86-compatible architecture (such as a Mac with
one of the M-series chips), you will either need to have a compiler installed
or you will want to install ``natcap.invest`` from ``conda-forge`` (see above).

To install ``natcap.invest`` via ``pip``::

    pip install natcap.invest

Note that ``pip install natcap.invest`` will fail with a compilation error if
GDAL is not already installed. ``natcap.invest`` depends on the underlying
GDAL binaries, which cannot be installed via ``pip``. For details on installing
GDAL on your system, see https://gdal.org/download.html.

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

Installing from source
----------------------

.. note::
    To install the ``natcap.invest`` package from source, you must have a C/C++
    compiler installed and configured for your system. MacOS and Linux users
    should not need to do anything. Windows users should install Microsoft
    Visual Studio, or at least the Build Tools for Visual Studio, if they have
    not already. See the `python wiki page on compilation under Windows <https://wiki.python.org/moin/WindowsCompilers>`_ for more information.

If you are looking for the latest unreleased changes to InVEST, you can use the
latest changes in the source tree.  This approach requires both a C/C++ compiler
and ``git`` to be available on your system::

    pip install "git+https://github.com/natcap/invest.git@main#egg=natcap.invest"

Note that GDAL must be installed on your system before installing in this way.
For details on installing GDAL, see https://gdal.org/download.html.

Dependencies
------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt`` and
included here for your reference.  Your package manager should, under most
circumstances, handle the dependency resolution for you.

.. include:: ../../requirements.txt
    :literal:
    :start-line: 12
