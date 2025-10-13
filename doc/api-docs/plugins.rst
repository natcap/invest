.. _plugins:

InVEST® Plugins: Developer's Guide
==================================

What is a plugin?
-----------------

Conceptually, an InVEST plugin is an ecosystem services model. Like the core InVEST
models, it takes in data of various formats (usually including some geospatial data),
processes that data, and produces output files that contain the results.
Unlike the core models, a plugin is not "official", i.e., not reviewed or maintained
by NatCap. Plugins may be developed, used, and distributed totally independently of
the ``natcap/invest`` repo and the Natural Capital Project.

In a technical sense, an InVEST plugin is a python package that conforms to the
``natcap.invest`` plugin API. This makes it possible to run the plugin from the
InVEST workbench and the ``invest`` command line tool. The plugin can execute any
arbitrary code when it runs. Commonly the ecosystem services model logic will be
implemented in the python package, but it could also invoke another software tool -
for example, if your model is already implemented in another language, you could
develop the plugin as a python wrapper for it.

Why make a plugin?
------------------

A plugin can be run in the InVEST workbench, which provides a graphical interface
where a user can enter the model inputs, run the model, watch its progress, and
access the results. All the necessary information to display the model in the
workbench is pulled from the plugin python package - no frontend development needed.
This is handy when resources are too limited to develop a separate GUI for a project.
It is a major benefit for developers to be able to focus on their model and not
worry about maintaining a desktop application or distributing it across multiple
operating systems.

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
into a well-documented, reusable, distributable software tool. Implementing the
plugin API requires attention to many details that are easily overlooked when
writing a basic script. Going through the process of developing a model into a
plugin will help to catch bugs and identify assumptions that may exist in your
math or your code.

Example plugins
---------------

The NatCap team maintains a small number of InVEST plugins. The following are some examples you may find useful as you develop your own plugin:

- `InVEST Demo Plugin <https://github.com/natcap/invest-demo-plugin>`_ (referenced in more detail in later sections of this guide)
- `InVEST GCM Downscaling <https://github.com/natcap/invest-gcm-downscaling>`_
- `Sediment Delivery Ratio with USLE C Raster Input <https://github.com/natcap/invest-sdr-usle-c-raster>`_

In addition, you may wish to consult the source code of the `core InVEST models <https://github.com/natcap/invest/tree/main/src/natcap/invest>`_ for examples of model code.

How to develop a plugin
-----------------------
.. note:: This guide is written for python developers. If you are unfamiliar with python packaging, the `Python Packaging User Guide <https://packaging.python.org/en/latest/>`__ is a helpful resource.

At its most basic, a plugin is a python package. Begin by creating a directory
with a simple python package structure: ::

    invest-demo-plugin/
    |- pyproject.toml
    |- src/
       |- invest_demo_plugin/
          |- __init__.py
          |- foo.py

The model code is in ``src/invest_demo_plugin/foo.py``. ``pyproject.toml`` is a standard configuration
file that defines the package. A full-fledged plugin will contain many other optional
files as well, like a readme, license, test suite, and sample data, but these are not
required.

Writing the ``pyproject.toml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``pyproject.toml`` contains standard information used to build your python package,
as well as custom configuration for other software that interacts with the package. InVEST looks for metadata
about your package in the ``pyproject.toml``. Configuration specific to InVEST is defined in the ``[tool.natcap.invest]`` block. See the `Python Packaging User Guide <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>`__ for more general information on ``pyproject.toml``.

.. literalinclude:: pyproject.toml
   :language: TOML

.. note::

    When InVEST installs a plugin, it uses ``micromamba`` to create an isolated environment
    in which to run the plugin. This environment contains the dependencies that the plugin needs,
    and prevents dependency conflicts with other plugins.

    If you are familiar with ``conda``, note that ``micromamba`` is very similar. It can install the same packages and has a similar API.

    A plugin will likely depend on some other python packages like ``numpy`` or ``pandas``.
    It may also depend on some other software that is not available as a pure python package,
    like GDAL. This is why we use ``micromamba`` instead of a python-specific package manager like ``pip`` - ``micromamba`` can manage both python and non-python dependencies (and ``python`` itself).


Writing the main module
^^^^^^^^^^^^^^^^^^^^^^^
The plugin python package must have the attributes ``MODEL_SPEC``, ``execute``, and ``validate``:

``MODEL_SPEC``
~~~~~~~~~~~~~~
An instance of :func:`natcap.invest.spec.ModelSpec`. This object stores key information about the model, its inputs, and its outputs. See the :ref:`API Reference<api>` for the specifics on instantiating this object. Note that the ``model_id`` should be globally unique, i.e. different from the ``model_id`` of any core InVEST model or any other plugin. To help ensure uniqueness among other plugins, it is a good idea for the ``model_id`` to match the python package name.

Here is an example ``MODEL_SPEC`` taken from the demo plugin. It describes a model that takes in three inputs: a raster file, an integer multiplication factor, and a workspace directory in which to produce the results. The model produces a raster file output which is the result of multiplying the input raster pixelwise by the multiplication factor. ::

    from natcap.invest import spec

    MODEL_SPEC = spec.ModelSpec(
        model_id="demo",
        model_title="Demo Plugin",
        userguide='',
        input_field_order=[
            ['workspace_dir', 'results_suffix'],
            ['raster_path', 'factor']],
        inputs=[
            spec.DirectoryInput(
                id="workspace_dir",
                name=gettext("workspace"),
                about=gettext(
                    "The folder where all the model's output files will be written. If "
                    "this folder does not exist, it will be created. If data already "
                    "exists in the folder, it will be overwritten."),
                contents={},
                must_exist=False,
                permissions="rwx"
            ),
            spec.SingleBandRasterInput(
                id="raster_path",
                name="Input Raster",
                data_type=float,
                units=None
            ),
            spec.IntegerInput(
                id="factor",
                name="Multiplication Factor"
            )
        ],
        outputs=[
            spec.SingleBandRasterOutput(
                id="result",
                path="result.tif",
                about="Raster multiplied by factor",
                data_type=float,
                units=None
            )
        ]
    )

``execute``
~~~~~~~~~~~
This function executes the model. When a user runs the model, this function is invoked with the inputs that the user provided. When this function returns, the model run is complete.

Arguments: ``args`` (dictionary). Maps input ids (matching the ``id`` of each ``Input`` in ``MODEL_SPEC.inputs``) to their values to run the model on.

Returns: Dictionary mapping output IDs to the absolute file paths where those outputs were created. All invest models use ``natcap.invest.file_registry.FileRegistry`` to manage and track file paths throughout the model. Returning the ``registry`` attribute of the ``FileRegistry`` satisfies this requirement.

Here is an example implementation of ``execute`` corresponding to the example ``MODEL_SPEC`` above. It multiplies a raster pixelwise by an integer value, and writes out the result to a new raster file: ::

    import os
    from natcap.invest import utils
    from natcap.invest.file_registry import FileRegistry

    MODEL_SPEC = ... # see above

    def execute(args):
        utils.make_directories([args['workspace_dir']])
        file_registry = FileRegistry(MODEL_SPEC, args['workspace_dir'])
        graph = taskgraph.TaskGraph(args['workspace_dir'], -1)
        graph.add_task(
            func=multiply_op,
            kwargs={
                'raster_path': args['raster_path'],
                'factor': int(args['factor']),
                'target_path': file_registry['result']
            },
            target_path_list=[file_registry['result']],
            task_name='multiply raster by factor')
        graph.join()
        return file_registry.registry

.. note::
    All core InVEST models use `taskgraph <https://github.com/natcap/taskgraph>`_ to organize the steps of execution. This is optional, but ``taskgraph`` has several benefits including avoided recomputation, distributing tasks over multiple CPUs, and logically organizing the model as a workflow of tasks that process data. See the InVEST source code for many examples of using ``taskgraph``.

``validate``
~~~~~~~~~~~~
This function validates the model inputs. Its purpose is to identify problems with the user's data before running the model, and give helpful feedback so the problems can be fixed. When a user enters data into the workbench UI, ``validate`` is called and its output is used to provide instant feedback (for instance, highlighting problematic inputs in red). The "Run" button will be disabled until all inputs validate successfully and ``validate`` returns ``[]``.

Arguments: ``args`` (dictionary). Maps input ids (matching the ``id`` of each ``Input`` in ``MODEL_SPEC.inputs``) to their values to run the model on.

Returns: A list of tuples where the first element of the tuple is an iterable of keys affected by the error in question and the second element of the tuple is the string message of the error. If no validation errors were found, an empty list is returned.

The following implementation of ``validate`` will suffice for most models: ::

    from natcap.invest import validation

    @validation.invest_validator
    def validate(args):
        return validation.validate(args, MODEL_SPEC)

``validation.validate`` performs pre-defined validation for each input type based on its properties. See the ``validate`` method of each ``Input`` class to see exactly what checks are performed. If you need to validate properties of the input data that are not covered by the pre-defined checks, you may add on to this basic ``validate`` function.

Specifying model inputs and outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Model inputs are specified in the ``inputs`` attribute of the ``MODEL_SPEC``. Many different types of model inputs are supported, including numbers, CSVs, raster and vector files, etc. Each input in ``inputs`` is an instance of a subclass of :func:`natcap.invest.spec.Input` that represents the data type. Choose the most appropriate ``Input`` type available in ``spec``. You may also subclass from :func:`natcap.invest.spec.Input` if you wish to create a custom type. Each ``Input`` type should define an ``id`` (which uniquely identifies the input within the model), a ``name`` (user-facing input name, displayed in the workbench), and an ``about`` (user-facing description of the input), in addition to the properties specific to the input type.

User-provided values for all input types are ultimately passed to the ``execute`` function as strings or numbers. For instance, all file-based types will accept a path string.

Model outputs are specified in the ``outputs`` attribute of the ``MODEL_SPEC``. All InVEST model outputs are files - there are no plain number or string outputs. Choose the most appropriate ``Output`` type available in ``spec``. You may also subclass from :func:`natcap.invest.spec.Output` if you wish to create a custom type.

.. note::

    Common inputs: All core InVEST models include the inputs ``workspace_dir`` (a target directory where all model results are written), ``results_suffix`` (a suffix appended to all model results, which may be used to differentiate model runs in the same workspace), and ``n_workers`` (passed to ``taskgraph`` to configure the number of CPUs used). Standard specs for these inputs are provided in :func:`natcap.invest.spec`.

    The ``n_workers`` input is normally hidden from display in core InVEST models (using the ``hidden`` attribute). ``n_workers`` is configured as a workbench-level setting, rather than a model-level input. The workbench adds the ``n_workers`` value from its settings to the ``args`` passed to the model. If you do not use ``taskgraph``, you are free to ignore this value.

Specifying units
^^^^^^^^^^^^^^^^
Some input and output types have a ``units`` attribute representing the units of measurement of the data. We use `pint <https://github.com/hgrecco/pint/tree/master>`_ to manage units. In ``pint``, all unit objects must derive from the same ``UnitRegistry`` in order to be used together. Therefore, you should reference ``natcap.invest``'s shared unit registry, :func:`natcap.invest.spec.u`. Example: ``spec.u.meter ** 3`` (cubic meters).

Nested data
^^^^^^^^^^^
Certain input and output types contain multiple types of data (such as columns in a CSV, or fields in a vector).

- :func:`.CSVInput` and :func:`.CSVOutput`: The ``columns`` attribute is an iterable of ``Input``\ s or ``Output``\ s that represent the data stored in each column of the CSV. The ``id`` of each ``Input``/``Output`` must match the column header.

- :func:`.VectorInput` and :func:`.VectorOutput`: The ``fields`` attribute is an iterable of ``Input``\ s or ``Output``\ s that represent the data stored in each field of the Vector. The ``id`` of each ``Input``/``Output`` must match the field name.

- :func:`.DirectoryInput`: The ``contents`` attribute is an iterable of ``Input``\ s that represent the file contents of the directory. The ``id`` of each ``Input`` must match the file name.

Example: ::

    CSVInput(
        id="biophysical_table_path",
        name="biophysical table",
        about="Table of crop coefficients for each LULC class.",
        columns=[
            IntegerInput(
                id="lulc_code",
                about="Land use/land cover code"
            ),
            NumberInput(
                id="kc_factor",
                about="Crop coefficient for each land use/land cover class",
                units=None
            )
        ],
        index_col="lulc_code"
    )

``__init__.py``
^^^^^^^^^^^^^^^
If you are following the project layout described above, and demonstrated in the demo plugin repo, ``MODEL_SPEC``, ``execute``, and ``validate`` will be properties of the ``foo`` submodule. You must make them available at the level of the ``invest_plugin`` package by importing them into ``__init__.py``. This is demonstrated in the demo plugin repo.


How InVEST interacts with plugins
---------------------------------
At the python level, when ``natcap.invest`` is imported, it searches for other installed python packages that look like plugins. Plugins are identified by having a package name beginning with ``invest`` and having the expected attributes described above. All identified plugin packages are recorded in ``natcap.invest.models`` and become available to ``natcap.invest`` just like the core models.

At the workbench level, plugins are installed through the "Manage Plugins" window. The user provides the plugin source code (as either a git URL or a local path), and the workbench follows this process to install it:

1. If installing from a git URL, download the ``pyproject.toml``.
2. Parse metadata from the ``pyproject.toml``.
3. Create a micromamba environment for the plugin to run in. The environment contains ``python`` and ``git`` by default, plus any dependencies specified in ``pyproject.toml`` in ``tool.natcap.invest.conda_dependencies``.
4. Use ``pip`` to install the plugin into the environment created in step 3.
5. Import the plugin and access metadata from the ``MODEL_SPEC``.
6. Save the plugin information to the workbench settings store.

The settings store is where the workbench tracks what plugins are installed. When a user launches a plugin, a new ``natcap.invest`` server is launched from the plugin's environment. This server runs on a different port than the ``natcap.invest`` server that serves core models. All model-specific requests related to running the plugin are sent to that port.

.. note::
   The workbench uniquely identifies plugins by a hash of a combination of the ``model_id`` from the ``MODEL_SPEC`` and the ``package.version`` from the ``pyproject.toml``. Therefore it is possible to have multiple versions of the same plugin simultaneously installed. At the python level, only one plugin with the same package name can be used at a time.

Testing Your Plugin
-------------------
Automated tests help ensure your plugin continues to work as expected as you develop it. See the InVEST repo for examples of model tests.
