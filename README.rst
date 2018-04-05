InVEST: Integrated Valuation of Ecosystem Services and Tradeoffs
================================================================

+-----------------------+-------------------------------+
| Build type            | Windows                       |
+=======================+===============================+
| Nightly Binary builds | |nightly_binary_build_badge|  |
+-----------------------+-------------------------------+
| Dev builds            | |dev_windows_build_badge|     |
+-----------------------+-------------------------------+
| Tests                 | |windows_test_badge|          |
+-----------------------+-------------------------------+
| Test coverage         | |windows_test_coverage_badge| |
+-----------------------+-------------------------------+

.. |nightly_binary_build_badge| image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=invest-nightly-develop
  :target: http://builds.naturalcapitalproject.org/job/invest-nightly-develop

.. |dev_windows_build_badge| image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=natcap.invest/label=GCE-windows-1
  :target: http://builds.naturalcapitalproject.org/job/natcap.invest/label=GCE-windows-1

.. |windows_test_badge| image:: http://builds.naturalcapitalproject.org/buildStatus/icon?job=test-natcap.invest/label=GCE-windows-1
  :target: http://builds.naturalcapitalproject.org/job/test-natcap.invest/label=GCE-windows-1

.. |windows_test_coverage_badge| image:: http://builds.naturalcapitalproject.org:9931/jenkins/c/http/builds.naturalcapitalproject.org/job/test-natcap.invest/label=GCE-windows-1/
  :target: http://builds.naturalcapitalproject.org/job/test-natcap.invest/label=GCE-windows-1


InVEST (Integrated Valuation of Ecosystem Services and Tradeoffs) is a family
of tools for quantifying the values of natural capital in clear, credible, and
practical ways. In promising a return (of societal benefits) on investments in
nature, the scientific community needs to deliver knowledge and tools to
quantify and forecast this return. InVEST enables decision-makers to quantify
the importance of natural capital, to assess the tradeoffs associated with
alternative choices, and to integrate conservation and human development.

Older versions of InVEST ran as script tools in the ArcGIS ArcToolBox environment,
but have almost all been ported over to a purely open-source python environment.

.. note::
    **This repository is for InVEST 3.2.1 and later**

    This replaces our Google Code project formerly
    located at http://code.google.com/p/invest-natcap.  If you are looking to build
    InVEST <= 3.2.0, use the archived release-framework repository at
    https://bitbucket.org/natcap/invest-natcap.release-framework, and the InVEST repository
    at https://bitbucket.org/natcap/invest-natcap.invest-3.


General Information
-------------------

* Website: https://naturalcapitalproject.org/invest
* Source code: https://bitbucket.org/natcap/invest
* Issue tracker: https://bitbucket.org/natcap/invest/issues
* Users' guide: http://data.naturalcapitalproject.org/nightly-build/invest-users-guide/html/
* API documentation: http://invest.readthedocs.io/en/latest/


Building InVEST
---------------

Dependencies
++++++++++++

Run ``make check`` to test if all required dependencies are installed on your system.
OS-specific installation instructions are found either online at
http://invest.readthedocs.io/en/latest/installing.html or locally at ``doc/api-docs/installing.rst``.

Building ``natcap.invest`` python package
+++++++++++++++++++++++++++++++++++++++++

A Makefile target has been created for your convenience::

    $ make python_packages

This will create a wheel for your platform and a zip source archive in ``dist/``.















Matplotlib ImportError
----------------------

On Fedora systems, some users encounter this exception when trying to run an
InVEST model that uses matplotlib:

::

    ...
    line 17, in <module>
        from .backend_qt5agg import NavigationToolbar2QTAgg
    ImportError: No module named backend_qt5agg

This is a `known issue`_ with the RedHat build of ``python-matplotlib-qt4``.  The workaround
is to ``yum install python-matplotlib-qt5``.

.. _known issue: https://bugzilla.redhat.com/show_bug.cgi?id=1219556



Running Tests
=============

To run the full suite of tests:

::

    $ paver test

To specify a test (or multiple tests) to run via `paver test`, use the nosetests
format to specify test files, classes, and/or test methods to run.  For example:

::

    $ paver test tests/test_example.py:ExampleTest.test_regression

This will only run this one test, ignoring all other tests that would normally be
run.

If you're looking for some extra verbosity (or you're building on jenkins):

::

    $ paver test --jenkins

You may also launch tests from the python shell:

::

    >>> import natcap.invest
    >>> natcap.invest.test()



