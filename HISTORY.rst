.. :changelog:

3.3.0
=====

* Fixed a crash with the NDR model that could occur if the DEM and landcover maps were different resolutions.
* Introduced a new InVEST model, "Scenario Generator: Proximity Based" and renamed the previous "Scenario Generator" to "Scenario Generator: Rule Based".
* Introduced a forest carbon edge effect model.
* Refactored all the InVEST model user interfaces so that Workspace defaults to the user's home "Documents" directory.

3.2.1
=====
* Turning setuptools' zip_safe to False for consistency across the Natcap Namespace.
* GLOBIO no longer requires user to specify a keyfield in the AOI.
* new feature to GLOBIO to summarize MSA by AOI.
* new feature to GLOBIO to use a user defined MSA parameter table to do the MSA
	thresholds for infrastructure, connectivity, and landuse type
* documentation to the GLOBIO code base including the large docstring for 'execute'.
