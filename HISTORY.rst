.. :changelog:

3.3.0
=====
* Fixed a bug in the Area Change rule of the Rule-Based Scenario Generator, where units were being converted incorrectly. (Issue `#3472 <https://bitbucket.org/natcap/invest/issues/3472>`_) Thanks to Fosco Vesely for this fix.
* InVEST Seasonal Water Yield model released.
* InVEST Forest Carbon Edge Effect model released.
* InVEST Scenario Generator: Proximity Based model released and renamed the previous "Scenario Generator" to "Scenario Generator: Rule Based".
* Implemented a blockwise exponential decay kernel generation function, which is now used in the Pollination and Habitat Quality models.
* GLOBIO now uses an intensification parameter and not a map to average all agriculture across the GLOBIO 8 and 9 classes.
* GLOBIO outputs modified so core outputs are in workspace and intermediate outputs are in a subdirectory called 'intermediate_outputs'.
* Fixed a crash with the NDR model that could occur if the DEM and landcover maps were different resolutions.
* Refactored all the InVEST model user interfaces so that Workspace defaults to the user's home "Documents" directory.
* Fixed an HRA bug where stessors with a buffer of zero were being buffered by 1 pixel
* HRA enhancement which creates a common raster to burn all input shapefiles onto, ensuring consistent alignment.
* Fixed an issue in SDR model where a landcover map that was smaller than the DEM would create extraneous "0" valued cells.
* New HRA feature which allows for "NA" values to be entered into the "Ratings" column for a habitat / stressor pair in the Criteria Ratings CSV. If ALL ratings are set to NA, the habitat / stressor will be treated as having no interaction. This means in the model, that there will be no overlap between the two sources. All rows parameters with an NA rating will not be used in calculating results.
* Habitat Quality bug fix when given land cover rasters with different pixel sizes than threat rasters. Model would use the wrong pixel distance for the convolution kernel.
* Light refactor of Timber model. Now using CSV input attribute file instead of DBF file.
* Fixed clipping bug in Wave Energy model that was not properly clipping polygons correctly. Found when using global data.
* Made the following changes / updates to the coastal vulnerability model:
    * Fixed a bug in the model where the geomorphology ranks were not always being used correctly.
    * Removed the HTML summary results output and replaced with a link to a dashboard that helps visualize and interpret CV results.
    * Added a point shapefile output: 'outputs/coastal_exposure.shp' that is a shapefile representation of the corresponding CSV table.
    * The model UI now requires the 'Relief' input. No longer optional.

3.2.1
=====
* Turning setuptools' zip_safe to False for consistency across the Natcap Namespace.
* GLOBIO no longer requires user to specify a keyfield in the AOI.
* new feature to GLOBIO to summarize MSA by AOI.
* new feature to GLOBIO to use a user defined MSA parameter table to do the MSA
	thresholds for infrastructure, connectivity, and landuse type
* documentation to the GLOBIO code base including the large docstring for 'execute'.
