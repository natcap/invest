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
A plugin will likely depend on some other python packages like ``numpy`` or ``pandas``.
It may also depend on some other software that is not available as a pure python package,
like GDAL.

When InVEST installs a plugin, it uses ``micromamba`` to create an isolated environment
in which to run the plugin. This environment can contain specific versions of dependencies
that the plugin needs.

Writing the main module
^^^^^^^^^^^^^^^^^^^^^^^

The plugin python package must have

Plugin API specification
------------------------

A plugin python pacakge must have the following attributes -

- ``MODEL_SPEC``: a dictionary that describes the model's inputs and outputs.

- ``execute``: a function that executes the model.
    Arguments: ``args`` (dictionary)
    Returns: ``None``

- ``validate``: a function that validates the model inputs.
    Arguments: ``args`` (dictionary)
    Returns: A list of tuples where the first element of the tuple is an iterable of
        keys affected by the error in question and the second element of the
        tuple is the string message of the error. If no validation errors were
        found, an empty list is returned.


