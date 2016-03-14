.. :changelog:

Changes
=======

3.3.0 (2016-03-14)
------------------
* Refactored Wind Energy model to use a CSV input for wind data instead of a Binary file.
* Redesigned InVEST recreation model for a single input streamlined interface, advanced analytics, and refactored outputs.  While the model is still based on "photo user days" old model runs are not backward compatable with the new model or interface. See the Recreation Model user's guide chapter for details.
    * The refactor of this model requires an upgrade to ``GDAL >=1.11.0 <2.0`` and ``numpy >= 1.10.2``.
* Removed nutrient retention (water purification) model from InVEST suite and replaced it with the nutrient delivery ratio (NDR) model.  NDR has been available in development relseases, but has now officially been added to the set of Windows Start Menu models and the "under development" tag in its users guide has been removed.  See the InVEST user's guide for details between the differences and advantages of NDR over the old nutrient model.
* Modified NDR by adding a required "Runoff Proxy" raster to the inputs.  This allows the model to vary the relative intensity of nutrient runoff based on varying precipitation variability.
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
* Refactored Coastal Blue Carbon model for greater speed, maintainability and clearer documentation.
* Habitat Quality bug fix when given land cover rasters with different pixel sizes than threat rasters. Model would use the wrong pixel distance for the convolution kernel.
* Light refactor of Timber model. Now using CSV input attribute file instead of DBF file.
* Fixed clipping bug in Wave Energy model that was not properly clipping polygons correctly. Found when using global data.
* Made the following changes / updates to the coastal vulnerability model:
    * Fixed a bug in the model where the geomorphology ranks were not always being used correctly.
    * Removed the HTML summary results output and replaced with a link to a dashboard that helps visualize and interpret CV results.
    * Added a point shapefile output: 'outputs/coastal_exposure.shp' that is a shapefile representation of the corresponding CSV table.
    * The model UI now requires the 'Relief' input. No longer optional.
    * CSV outputs and Shapefile outputs based on rasters now have x, y coorinates of the center of the pixel instead of top left of the pixel.
* Turning setuptools' zip_safe to False for consistency across the Natcap Namespace.
* GLOBIO no longer requires user to specify a keyfield in the AOI.
* New feature to GLOBIO to summarize MSA by AOI.
* New feature to GLOBIO to use a user defined MSA parameter table to do the MSA thresholds for infrastructure, connectivity, and landuse type
* Documentation to the GLOBIO code base including the large docstring for 'execute'.
