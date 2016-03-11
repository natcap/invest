Installing InVEST
=================

Python Dependencies
-------------------

Dependencies for ``natcap.invest`` are listed in ``requirements.txt``:

.. include:: ../../requirements.txt
    :literal:
    :start-line: 9

Additionally, ``PyQt4`` is required to use the ``invest`` cli, but is not
required for development against ``natcap.invest``.

Installing from Source
----------------------

If you have a compiler installed and configured for your system, and
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

The latest development version of InVEST can be installed from our source tree::

    $ pip install hg+https://bitbucket.org/natcap/invest@develop

