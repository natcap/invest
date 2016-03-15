======================
InVEST Build Framework
======================

Scripts, hooks, and instructions for creating a binary build of InVEST.

Requirements
------------

* PyInstaller (see PyInstaller project for own OS-specific dependencies)
* InVEST3

Usage
-----

Currently must call pyinstaller from same directory as spec file (b/c of spec file)

.. code:: shell

    $ pyinstaller invest.spec

Distributable build stored in top-level `dist/` folder.

Planning
--------

* Support for Linux
* Support for Windows
* Support for Mac OSX
* Scripts to install InVEST dependencies

Notes on PyInstaller
--------------------

Repository: https://github.com/pyinstaller/pyinstaller

Documentation: http://pythonhosted.org/PyInstaller/

Definitions
~~~~~~~~~~~

scripts: the python scripts named at command line

pure: pure python modules needed by the scripts

binaries: non-python modules needed by the scripts

Stage 1: Dependency Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    a = Analysis()  # contains lists of scripts, pure, and binaries

hooks.hookutils functions:

* get_package_paths(package)
* hiddenimports = collect_submodules(package)
* datas = collect_data_files(package)

.. code:: python

    MERGE( [ (a, 'script_name', 'exec_name'), (b, 'b', 'b'), ... ] )

MERGE is used in multi-package bundles.


Stage 2: Python File Compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    pyz = PYZ(a.pure)  # contains the modules listed in a.pure

Stage 3: Create Executable Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # contains what will be the executable file 'myscript'
    exe = EXE(a.scripts, a.binaries, a.zipfiles, a.datas, pyz, name="myscript", exclude_binaries=1)

Stage 4: Collect Files into Distributable Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # creates the output folder
    COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="dist")


Notes on Preparing InVEST Repo for PyInstaller
----------------------------------------------

In Setup.py's setup() function:

    include_package_data=True,

Add MANIFEST.in with following:

    recursive-include invest_natcap/

Add additional elif block to iui.executor import switch statement

    from importlib import import_module

    elif getattr(sys, 'frozen', False) and getattr(sys, '_MEIPASS', False):
        model = import_module(module)
        model_name = os.path.splitext(os.path.basename(module))[0]
        LOGGER.debug('Loading %s in frozen environment', model)
