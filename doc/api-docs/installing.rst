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

Note that ``natcap.invest`` depends on GDAL, which must already be compiled and
available on your system.  GDAL can be installed from your system package manager,
from ``conda``/``mamba``, or from
`Christoph Gohlke's python builds for Windows <https://github.com/cgohlke/geospatial.whl/>`_.

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

Dependencies
------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt`` and
included here for your reference.  Your package manager should, under most
circumstances, handle the dependency resolution for you.

.. include:: ../../requirements.txt
    :literal:
    :start-line: 12
