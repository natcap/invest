.. _docker:

=============================
Running InVEST in a Container
=============================

In addition to being available as a python package for installation in your
environment of choice, the entire InVEST runtime is available within a
container for direct use in scientific workflows or as a dependency
for custom containers.

Running InVEST with Docker
==========================

Prerequisites
-------------

To use this container within Docker, you will need a docker runtime.
See `docker.com for installation instructions <https://docs.docker.com/get-started/get-docker/>`_.

These examples assume that you already have a python script with
your model run parameters defined called ``run_model.py``.  Please see
:ref:`CreatingPythonScripts` for more information about how to do this
from the InVEST Workbench if needed.

.. note::

        Because docker containers are executing within a virtual machine, your
        script must work with paths that are valid within the container.  In this
        example, you will be safest by keeping filepaths relative to your
        current working directory.

These docker examples include a few extra parameters that are very useful for
scientific analyses:

* ``--rm`` removes the container after execution.  Doing so will save disk
  space with many repeated runs.
* ``-ti`` allows output to be printed to the console and if your script has any
  interactivity (e.g. python's ``pdb`` debugger), then your input will be fed
  back into the container.
* ``-v <dirname>:/natcap`` mounts the current working directory into the
  ``/natcap`` directory within the container.  This is useful for sharing your
  input data with your script.  See the note above about filepaths used in your
  script.
* ``-w /natcap`` will make your program execute from the current working directory.


Running the Latest Container on Windows
---------------------------------------

.. code-block:: shell

   docker run --rm -ti -v %CD%:/natcap -w /natcap ghcr.io/natcap/invest:latest python3 run_model.py


Running the Latest Container on Mac/Linux
-----------------------------------------

.. code-block:: shell

   docker run --rm -ti -v $(pwd):/natcap -w /natcap ghcr.io/natcap/invest:latest python3 run_model.py

Running a Specific Container Version
------------------------------------

Docker has two ways to define the version of a container that you would like to
run: you can use a tag or you can refer to a very specific container version,
identified by a SHA256 digest.

For reproducibility in scientific analyses, we recommend using the SHA256
digest of the container to refer to a very specific version of the container.
For example, ``ghcr.io/natcap/invest@sha256:8a4a3c621c09e1df74821efa0eb8e2cd90ea6e228e393e953cb4ea0ff8d37454``.
Digests can be found on the `InVEST containers page on github <https://github.com/natcap/invest/pkgs/container/invest>`_.

Alternatively, you could use a tag to refer to a container. InVEST containers
have several tags that you could use, including:

* InVEST version (e.g. ``3.16.2``).  This refers to the InVEST version built in
  this container, but is only present for tagged releases.
  * Example: ``ghcr.io/natcap/invest:3.16.2``

* Git commit SHA256.  Each container build is tagged with the git revision in the
  ``natcap/invest`` repository that was used to build the container.
  * Example: ``ghcr.io/natcap/invest:e756a7a43d598f2130199bf1c9ad4d2df4975cfd``

* ``latest``.  This is a moveable tag and always points to the most recent
  InVEST container build.  Do not use this if you care about reproducibility of
  your analysis.
  * Example: ``ghcr.io/natcap/invest:latest``

Running a Container on an HPC Cluster
-------------------------------------

On HPC clusters, you will generally need to use a container-compatible software
like `Apptainer <https://apptainer.org>`_ (previously known as ``singularity``).  Fortunately,
``apptainer`` will convert a docker container for you when it needs to, so all you
need to do is tell ``apptainer`` which container you want to run:

.. code-block:: shell

   apptainer run docker://ghcr.io/natcap/invest python my_script.py

Or if you'd like to refer to a very specific container, you can refer to the
SHA256 digest just like with a docker container:

.. code-block:: shell

   apptainer run docker://ghcr.io/natcap/invest@sha256:8a4a3c621c09e1df74821efa0eb8e2cd90ea6e228e393e953cb4ea0ff8d37454 python3 my_script.py
