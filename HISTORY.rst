.. :changelog:

3.3.0
=====
* Implemented a blockwise exponential decay kernel generation function, which is now used in the Pollination and Habitat Quality models.
* GLOBIO now uses an intensification parameter and not a map to average all agriculture across the GLOBIO 8 and 9 classes.
* GLOBIO outputs modified so core outputs are in workspace and intermediate outputs are in a subdirectory called 'intermediate_outputs'.
* Fixed a crash with the NDR model that could occur if the DEM and landcover maps were different resolutions.
* Introduced a new InVEST model, "Scenario Generator: Proximity Based" and renamed the previous "Scenario Generator" to "Scenario Generator: Rule Based".
* Introduced a forest carbon edge effect model.
* Refactored all the InVEST model user interfaces so that Workspace defaults to the user's home "Documents" directory.
* Fixed an HRA bug where stessors with a buffer of zero were being buffered by 1 pixel
* HRA enhancement which creates a common raster to burn all input shapefiles onto, ensuring consistent alignment.

3.2.1
=====
* Turning setuptools' zip_safe to False for consistency across the Natcap Namespace.
* GLOBIO no longer requires user to specify a keyfield in the AOI.
* new feature to GLOBIO to summarize MSA by AOI.
* new feature to GLOBIO to use a user defined MSA parameter table to do the MSA
	thresholds for infrastructure, connectivity, and landuse type
* documentation to the GLOBIO code base including the large docstring for 'execute'.
