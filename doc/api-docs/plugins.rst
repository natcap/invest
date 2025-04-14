.. _plugins:

InVEST Plugins: Developer's Guide
=================================

What is a plugin?
-----------------

Conceptually, an InVEST plugin is an ecosystem services model. Like the core InVEST
models, it takes in data of various formats (usually including some geospatial data),
processes that data, and produces output files that contain the results.
Unlike the core models, a plugin is not "official". Plugins may be developed, used,
and distributed totally independently of the ``natcap/invest`` repo and the Natural
Capital Project.

In a technical sense, an InVEST plugin is a python package that conforms to the
``natcap.invest`` plugin API. This makes it possible to run the plugin from the
InVEST workbench and the ``invest`` command line tool. The plugin can execute any
arbitrary code when it runs. Commonly the ecosystem services model logic will be
implemented in the python package, but it could also invoke another software tool -
for example, if your model is already implemented as a command line tool, you could
develop the plugin as a python wrapper for it.

Why make a plugin?
------------------

A plugin can be run in the InVEST workbench, which provides a graphical interface
where a user can enter the model inputs, run the model, watch its progress, and
access the results. All the necessary information to display the model in the
workbench is pulled from the plugin python package - no frontend development needed.
This is handy when resources are too limited to develop a separate GUI for a project.

The data validation component of InVEST is also very useful for projects that don't
have enough resources to develop this independently. The plugin API requires
that data inputs are rigorously specified. Before running a model, InVEST validates
that the provided data meets all of the requirements, and provides helpful feedback
if it does not. This prevents a lot of trouble with invalid data.

Even if resources were unlimited, we think there is value in having a shared interface
for ecosystem services models. Seeing different ecosystem service models, or different
versions of the same model, side-by-side in the workbench facilitates running them
together and comparing them.

The plugin API is a useful framework in which to think of developing a model.
This framework is helpful when tackling the task of turning a "model" (which may
exist in the form of mathematical equations, scripts, or other software)
into a well-documented, reusable, distributable software tool.

Implementing the plugin API requires attention to many details that are easily
overlooked when writing a basic script. Going through the process of developing
a model into a plugin will help to catch bugs and identify assumptions that may
exist in your math or your code.


How to develop a plugin
-----------------------
.. note:: This guide assumes the reader is familiar with python.

The plugin template repo is a great place to start.

At its most basic, a plugin is a python package. Begin by creating a directory
with a simple python package structure:
``
foo_model/
|- pyproject.toml
|- src/
|  |- foo.py
``
The model code is in ``src/foo.py``. ``pyproject.toml`` is a standard configuration
file that defines the package. A full-fledged plugin will contain many other optional
files as well, like a readme, license, test suite, and sample data, but these are not
required. (and demonstrated in the plugin template repo).

Writing the ``pyproject.toml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
The ``pyproject.toml`` contains standard information used to build your python package,
as well as custom configuration for other software (like InVEST) that uses the package.
Configuration specific to InVEST is defined in the ``[tool.natcap.invest]`` block.

Supported keys within this block are:
 - ``api_version``: the plugin API version that this plugin conforms to
 - ``model_title``: the plugin's user-facing name
 - ``pyname``: the python importable package name of the plugin
 - ``model_id``: plugin identifier used internally. Using snake-case is recommended.
 - ``conda_dependencies``: list of the plugin's conda dependencies. These are usually
   dependencies that are not pure python and not available through pip. When invest installs
   the plugin, it creates a conda environment that the plugin will run in. At a minimum,
   python is required.


Managing dependencies
~~~~~~~~~~~~~~~~~~~~~

When InVEST installs a plugin, it uses ``micromamba`` to create an isolated environment
in which to run the plugin. This environment can contain specific versions of dependencies
that the plugin needs.

.. note:: If you are familiar with ``conda``, note that ``micromamba`` is very similar. It can install the same packages and has a similar API.

A plugin will likely depend on some other python packages like ``numpy`` or ``pandas``.
It may also depend on some other software that is not available as a pure python package,
like GDAL. This is why we use ``micromamba`` instead of a python-specific package manager like ``pip`` - ``micromamba`` can manage both python and non-python dependencies (and ``python`` itself).


Writing the main module
^^^^^^^^^^^^^^^^^^^^^^^

The plugin python package must have

Plugin API specification
------------------------

A plugin python pacakge must have the following attributes -

``MODEL_SPEC``
~~~~~~~~~~~~~~~
This is a dictionary that describes key information about the model, its inputs, and its outputs. The structure of this dictionary is fully specified below.

``model_id``
============
The unique identifier for the plugin model, used internally by invest. This identifier should be concise, meaningful, and unique - a good choice is often a short version of the ``model_title``, or the name of the github repo. Using snake-case is recommended for consistancy. Including the word "model" is redundant and not recommended.
type: ``string``
Good: ``'carbon_storage'``
Bad: ``'Carbon storage'``, ``carbon_storage_model``

``model_title``
===============
The user-facing title for the plugin. This is displayed in the workbench. Title-case is recommended. Including the word "model" is redundant and not recommended.
type: ``string``
Good: ``'Carbon Storage'``
Bad: ``'Carbon storage'``, ``The Carbon Storage Model``, ``carbon_storage``

``pyname``
==========

``userguide``
=============

``aliases``
===========


``ui_spec``
===========
type: ``dict``. This sub-dictionary tells the workbench how to display the input form for the plugin.

``ui_spec['order']``
====================
This is a list that specifies the order and grouping of model inputs. Inputs will displayed in the input form from top to bottom in the order listed here. Sub-lists represent groups of inputs that will be visually separated by a horizontal line. This improves UX by breaking up long lists and visually grouping related inputs. If you do not wish to use groups, all inputs may go in the same sub-list. It is a convention to begin with a group of ``workspace_dir`` and ``results_suffix``.

Example: ``[['workspace_dir', 'results_suffix'], ['foo'], ['bar', baz']]``

type: ``list`` of ``list``s of ``string``s
Constraints: each item in the sub-lists must match a key in ``MODEL_SPEC['args']``. Each key in ``MODEL_SPEC['args']`` must be included exactly once, unless it is included in ``ui_spec['hidden']``.

``ui_spec['hidden']``
=====================
This is a list of arg keys to hide from the input form. For most models, this should be ``["n_workers"]`` because the value of ``n_workers`` is provided from the workbench settings menu, and not from the model input form.

type: ``list`` of ``string``s
Constraints: each item in ``ui_spec['hidden']`` must match a key in ``MODEL_SPEC['args']``.

``args``
========
This is a sub-dictionary where the keys identify model inputs and the values describe the specification for those inputs.

Keys should be snake-cased strings that uniquely and concisely identify inputs within the model.
Good: ``precipitation``
Bad: ``precipitation map``

Values are dictionaries that store information about and constraints on the data inputs provided for each key.

``args[arg]``
==============

Attributes that apply to all types:

``args[arg]['name']`` (all types)
=================================
The user-facing name of this input. The workbench UI displays this property as a label for each input. The name should be as short as possible. Any extra description should go in ``args[arg]['about']``. It should be all lower-case, except for things that are always capitalized (acronyms, proper names). Any capitalization rules such as "always capitalize the first letter" will be applied on the workbench side.
type: ``string``
Good: ``precipitation``, ``Kc factor``, ``valuation table``
Bad: ``PRECIPITATION``, ``kc_factor``, ``table of valuation parameters``


``args[arg]['about']`` (all types)
==================================

``args[arg]['type']`` (all types)
=================================
Specifies the data type of the input. The supported data types reflect the types of data commonly used in our models.

This controls how the input is rendered in the model input form. Path types (``'directory'``, ``'file'``, ``'raster'``, ``'vector'``, ``'csv'``) display as a text box with a file finder button, so users may type in the path or select one through the OS file navigation. Booleans display as a toggle button, where ``False`` is "off" and ``True`` is "on". ``'option_string'``s display as a dropdown menu. All other types display as a regular text box.

type: ``string``, one of:

- ``'directory'``: a path to a directory on the user's computer
- ``'raster'``: a path to a GDAL-supported raster file
- ``'vector'`` - a path to a GDAL-supported vector file
- ``'csv'`` - a path to a CSV file (comma-or-semicolon delimited, possibly with a UTF-8 BOM)
- ``'file'``: a path to a file on the user's computer. Use this only for files which do not fall under one of the other types (``'raster'``, ``'vector'``, ``'csv'``, or ``'directory``').
- ``'number'`` - a decimal (floating-point) number
- ``'integer'`` - an integer number
- ``'ratio'``: a decimal number representing a ratio (scale of 0 to 1), though values less than 0 or greater than 1 are also allowed. This is important to distinguish from a percent (1.0 = 100%).
- ``'percent'``: a decimal number representing a percent (scale of 0 to 100), though values less than 0 or greater than 100 are also allowed. This is important to distinguish from a ratio (100% = 1.0).
- ``'freestyle_string'`` - a string that may contain any UTF-8 characters.
- ``'option_string'`` - a string where the value must belong to a set of options.
- ``'boolean'``: a boolean value (either true or false, or something that can be cast to true or false).


``args[arg]['required']`` (all types)
=====================================
Indicates whether the input is required to be provided. Defaults to ``True``. If the input is optional, set ``args[arg]['required']: False``.

type: ``bool``. Required.


``args[arg]['units']`` (``number`` types only)
==============================================
The unit of measurement of this number. Required.
type: ``pint.Unit``
Example: ``pint.UnitRegistry().meter``

``args[arg]['expression']`` (``number`` types only)
===================================================
Optional. This is an expression that allows you to set custom constraints on the input value, such as upper and lower bounds. The expression must contain the string ``value``, which will represent the user-provided value (after it has been cast to a float). The expression must evaluate to a boolean, or a type that is castable to boolean. If ``bool(eval(expression)) is False``, validation will reject the input.

type: ``string``
Example: ``"(value >= 0) & (value <= 1)"``


``args[arg]['columns']`` (``csv`` types only)
=============================================
A sub-dictionary specifying columns in the CSV table. Keys are column names. Values are nested arg specs.
type: ``dict`` mapping ``string``s to ``dict``s. Required.
Example:
``
'columns': {
    'lucode': {
        'type': 'integer',
        'about': 'LULC codes matching those in the LULC raster.'
    },
    'root_depth': {
        'type': 'number',
        'units': u.millimeter,
        'about': 'Maximum root depth for plants in this LULC class.'
    },
    'kc': {
        'type': 'number',
        'units': u.none,
        'about': 'Crop coefficient for this LULC class.'
    }
}
``

``args[arg]['bands']`` (``raster`` types only)
==============================================
A sub-dictionary specifying bands in the raster. Keys are integer band IDs. Values are nested arg specs. For single-band rasters, the band ID is ``1``.

type: ``dict`` mapping ``int``s to ``dict``s. Required.
Example:
``
'bands': {
    1: {
        'type': 'number',
        'units': u.meter
    }
}
``

``args[arg]['fields']`` (``vector`` types only)
===============================================
A sub-dictionary specifying fields in the vector. Keys are field names. Values are nested arg specs. If no fields are required, set this to an empty dictionary.

type: ``dict`` mapping ``string``s to ``dict``s. Required.

Example:
``
'fields': {
    'watershed_id': {
        'type': 'string',
        'about': 'Unique identifier for each watershed'
    },
    'weight': {
        'type': 'ratio',
        'about': 'Relative weight to give to the watershed in calculation'
    }
}
``


``args[arg]['geometries']`` (``vector`` types only)
===================================================
The set of allowed geometry types in the vector.
types are defined in spec_utils.py and should be imported from there.
type: ``set``. Required.
Example: ``{'POLYGON', 'MULTIPOLYGON'}``

``args[arg]['projected']`` (``raster`` and ``vector`` types only)
=================================================================
If ``True``, the dataset must have a projected (as opposed to geographic) coordinate system.
type: ``bool``. Required. Defaults to ``False``.

``args[arg]['projection_units']`` (``raster`` and ``vector`` types only)
========================================================================
If the model has a requirement for the linear units of the coordinate system, this may be specified. May only be used if ``args[arg]['projected'] is True``.

type: ``pint.Unit``. Optional.
Example: ``pint.UnitRegistry().meter``

``args[arg]['options']`` (``option_string`` types only)
=======================================================

.. note::
Required args: All models must include the args ``workspace_dir``, ``results_suffix`` and ``n_workers``. Standard specs for these args are provided in ``natcap.invest.spec_utils``.

``outputs``

``execute``
~~~~~~~~~~~
This function executes the model. When a user runs the model, this function is invoked with the inputs that the user provided. When this function returns, the model run is complete.

Arguments: ``args`` (dictionary). Maps input keys (which match those in ``MODEL_SPEC['args']``) to their values to run the model on.
Returns: ``None``. When ``execute`` returns, the model run is complete.

``validate``
~~~~~~~~~~~~
This function validates the model inputs. Its purpose is to identify problems with the user's data before running the model, and give helpful feedback so the problems can be fixed. When a user enters data into the workbench UI, ``validate`` is called and its output is used to provide instant feedback (for instance, highlighting problematic inputs in red). The "Run" button will be disabled until all inputs validate successfully and ``validate`` returns ``[]``.
    Arguments: ``args`` (dictionary). Maps input keys (which match those in ``MODEL_SPEC['args']``) to their values to run the model on.
    Returns: A list of tuples where the first element of the tuple is an iterable of
        keys affected by the error in question and the second element of the
        tuple is the string message of the error. If no validation errors were
        found, an empty list is returned.


