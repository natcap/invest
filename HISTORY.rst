.. :changelog:

.. Unreleased Changes

3.3.2 (2016-10-17)
------------------
* Partial test coverage for HRA model.
* Full test coverage for Overlap Analysis model.
* Full test coverage for Finfish Aquaculture.
* Full test coverage for DelineateIT.
* Full test coverage for RouteDEM.
* Fixed an issue in Habitat Quality where an error in the sample table or malformed threat raster names would display a confusing message to the user.
* Full test coverage for scenario generator proximity model.
* Patching an issue in seasonal water yield that causes an int overflow error if the user provides a floating point landcover map and the nodata value is outside of the range of an int64.
* Full test coverage for the fisheries model.
* Patched an issue that would cause the Seasonal Water Edge model to crash when the curve number was 100.
* Patching a critical issue with forest carbon edge that would give incorrect results for edge distance effects.
* Patching a minor issue with forest carbon edge that would cause the model to crash if only one  interpolation point were selected.
* Full test coverage for pollination model.
* Removed "farms aggregation" functionality from the InVEST pollination model.
* Full test coverage for the marine water quality model.
* Full test coverage for GLOBIO model.
* Full test coverage for carbon forest edge model.
* Upgraded SciPy dependancy to 0.16.1.
* Patched bug in NDR that would cause a phosphorus density to be reported per pixel rather than total amount of phosporous in a pixel.
* Corrected an issue with the uses of buffers in the euclidean risk function of Habitat Risk Assessment.  (issue #3564)
* Complete code coverage tests for Habitat Quality model.
* Corrected an issue with the ``Fisheries_Inputs.csv`` sample table used by Overlap Analysis.  (issue #3548)
* Major modifications to Terrestrial Carbon model to include removing the harvested wood product pool, uncertainty analysis, and updated efficient raster calculations for performance.
* Fixed an issue in GLOBIO that would cause model runs to crash if the AOI marked as optional was not present.
* Removed the deprecated and incomplete Nearshore Wave and Erosion model (``natcap.invest.nearshore_wave_and_erosion``).
* Removed the deprecated Timber model (``natcap.invest.timber``).
* Fixed an issue where seasonal water yield would raise a divide by zero error if a watershed polygon didn't cover a valid data region.  Now sets aggregation quantity to zero and reports a warning in the log.
* ``natcap.invest.utils.build_file_registry`` now raises a ``ValueError`` if a path is not a string or list of strings.
* Fixed issues in NDR that would indicate invalid values were being processed during runtimes by skipping the invalid calculations in the first place rather than calculating them and discarding after the fact.
* Complete code coverage tests for NDR model.
* Minor (~10% speedup) performance improvements to NDR.
* Added functionality to recreation model so that the `monthly_table.csv` file now receives a file suffix if one is provided by the user.
* Fixed an issue in SDR where the m exponent was calculated incorrectly in many situations resulting in an error of about 1% in total export.
* Fixed an issue in SDR that reported runtime overflow errors during normal processing even though the model completed without other errors.

3.3.1 (2016-06-13)
------------------
* Refactored API documentation for readability, organization by relevant topics, and to allow docs to build on `invest.readthedocs.io <http://invest.readthedocs.io>`_,
* Installation of ``natcap.invest`` now requires ``natcap.versioner``.  If this is not available on the system at runtime, setuptools will make it available at runtime.
* InVEST Windows installer now includes HISTORY.rst as the changelog instead of the old ``InVEST_Updates_<version>`` files.
* Habitat suitability model is generalized and released as an API only accessible model.  It can be found at ``natcap.invest.habitat_suitability.execute``.  This model replaces the oyster habitat suitability model.
    * The refactor of this model requires an upgrade to ``numpy >= 1.11.0``.
* Fixed a crash in the InVEST CLI where calling ``invest`` without a parameter would raise an exception on linux-based systems.  (Issue `#3528 <https://bitbucket.org/natcap/invest/issues/3515>`_)
* Patched an issue in Seasonal Water Yield model where a nodata value in the landcover map that was equal to ``MAX_INT`` would cause an overflow error/crash.
* InVEST NSIS installer will now optionally install the Microsoft Visual C++ 2008 redistributable on Windows 7 or earlier.  This addresses a known issue on Windows 7 systems when importing GDAL binaries (Issue `#3515 <https://bitbucket.org/natcap/invest/issues/3515>`_).  Users opting to install this redistributable agree to abide by the terms and conditions therein.
* Removed the deprecated subpackage ``natcap.invest.optimization``.
* Updated the InVEST license to legally define the Natural Capital Project.
* Corrected an issue in Coastal Vulnerability where an output shapefile was being recreated for each row, and where field values were not being stored correctly.
* Updated Scenario Generator model to add basic testing, file registry support, PEP8 and PEP257 compliance, and to fix several bugs.
* Updated Crop Production model to add a simplified UI, faster runtime, and more testing.

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

3.2.0 (2015-05-31)
------------------
InVEST 3.2.0 is a major release with the addition of several experimental models and tools as well as an upgrade to the PyGeoprocessing core:

* Upgrade to PyGeoprocessing v0.3.0a1 for miscelaneous performance improvements to InVEST's core geoprocessing routines.
* An alpha unstable build of the InVEST crop production model is released with partial documentation and sample data.
* A beta build of the InVEST fisheries model is released with documentation and sample data.
* An alpha unstable build of the nutrient delivery ratio (NDR) model is available directly under InVEST's instalation directory at  ``invest-x86/invest_ndr.exe``; eventually this model will replace InVEST's current "Nutrient" model.  It is currently undocumented and unsupported but inputs are similar to that of InVEST's SDR model.
* An alpha unstable build of InVEST's implementation of GLOBIO is available directly under InVEST's instalation directory at ``invest-x86/invest_globio.exe``.  It is currently undocumented but sample data are provided.
* DelinateIT, a watershed delination tool based on PyGeoprocessing's d-infinity flow algorithm is released as a standalone tool in the InVEST repository with documentation and sample data.
* Miscelaneous performance patches and bug fixes.

3.1.3 (2015-04-23)
------------------
InVEST 3.1.3 is a hotfix release patching a memory blocking issue resolved in PyGeoprocessing version 0.2.1.  Users might have experienced slow runtimes on SDR or other routed models.

3.1.2 (2015-04-15)
------------------
InVEST 3.1.2 is a minor release patching issues mostly related to the freshwater routing models and signed GDAL Byte datasets.

* Patching an issue where some projections were not regognized and InVEST reported an UnprojectedError.
* Updates to logging that make it easier to capture logging messages when scripting InVEST.
* Shortened water yield user interface height so it doesn't waste whitespace.
* Update PyGeoprocessing dependency to version 0.2.0.
* Fixed an InVEST wide issue related to bugs stemming from the use of signed byte raster inputs that resulted in nonsensical outputs or KeyErrors.
* Minor performance updates to carbon model.
* Fixed an issue where DEMS with 32 bit ints and INT_MAX as the nodata value nodata value incorrectly treated the nodata value in the raster as a very large DEM value ultimately resulting in rasters that did not drain correctly and empty flow accumulation rasters.
* Fixed an issue where some reservoirs whose edges were clipped to the edge of the watershed created large plateaus with no drain except off the edge of the defined raster.  Added a second pass in the plateau drainage algorithm to test for these cases and drains them to an adjacent nodata area if they occur.
* Fixed an issue in the Fisheries model where the Results Suffix input was invariably initializing to an empty string.
* Fixed an issue in the Blue Carbon model that prevented the report from being generated in the outputs file.

3.1.1 (2015-03-13)
------------------
InVEST 3.1.1 is a major performance and memory bug patch to the InVEST toolsuite.  We recommend all users upgrade to this version.

* Fixed an issue surrounding reports of SDR or Nutrient model outputs of zero values, nodata holes, excessive runtimes, or out of memory errors.  Some of those problems happened to be related to interesting DEMs that would break the flat drainage algorithm we have inside RouteDEM that adjusted the heights of those regions to drain away from higher edges and toward lower edges, and then pass the height adjusted dem to the InVEST model to do all its model specific calculations.  Unfortunately this solution was not amenable to some degenerate DEM cases and we have now adjusted the algorithm to treat each plateau in the DEM as its own separate region that is processed independently from the other regions. This decreases memory use so we never effectively run out of memory at a minor hit to overall runtime.  We also now adjust the flow direction directly instead of adjust the dem itself.  This saves us from having to modify the DEM and potentially get it into a state where a drained plateau would be higher than its original pixel neighbors that used to drain into it.

There are side effects that result in sometimes large changes to un calibrated runs of SDR or nutrient.  These are related to slightly different flow directions across the landscape and a bug fix on the distance to stream calculation.

* InVEST geoprocessing now uses the PyGeoprocessing package (v0.1.4) rather than the built in functionality that used to be in InVEST.  This will not affect end users of InVEST but may be of interest to users who script InVEST calls who want a standalone Python processing package for raster stack math and hydrological routing.  The project is hosted at https://bitbucket.org/richpsharp/pygeoprocessing.

* Fixed an marine water quality issue where users could input AOIs that were unprojected, but output pixel sizes were specified in meters.  Really the output pixel size should be in the units of the polygon and are now specified as such.  Additionally an exception is raised if the pixel size is too small to generate a numerical solution that is no longer a deep scipy error.

* Added a suffix parameter to the timber and marine water quality models that append a user defined string to the output files; consistent with most of the other InVEST models.

* Fixed a user interface issue where sometimes the InVEST model run would not open a windows explorer to the user's workspace.  Instead it would open to C:\User[..]\My Documents.  This would often happen if there were spaces in the the workspace name or "/" characters in the path.

* Fixed an error across all InVEST models where a specific combination of rasters of different cell sizes and alignments and unsigned data types could create errors in internal interpolation of the raster stacks.  Often these would appear as 'KeyError: 0' across a variety of contexts.  Usually the '0' was an erroneous value introduced by a faulty interpolation scheme.

* Fixed a MemoryError that could occur in the pollination and habitat quality models when the the base landcover map was large and the biophysical properties table allowed the effect to be on the order of that map.  Now can use any raster or range values with only a minor hit to runtime performance.

* Fixed a serious bug in the plateau resolution algorithm that occurred on DEMs with large plateau areas greater than 10x10 in size.  The underlying 32 bit floating point value used to record small height offsets did not have a large enough precision to differentiate between some offsets thus creating an undefined flow direction and holes in the flow accumulation algorithm.

* Minor performance improvements in the routing core, in some cases decreasing runtimes by 30%.

* Fixed a minor issue in DEM resolution that occurred when a perfect plateau was encountered.  Rather that offset the height so the plateau would drain, it kept the plateau at the original height.  This occurred because the uphill offset was nonexistent so the algorithm assumed no plateau resolution was needed.  Perfect plateaus now drain correctly.  In practice this kind of DEM was encountered in areas with large bodies of water where the remote sensing algorithm would classify the center of a lake 1 meter higher than the rest of the lake.

* Fixed a serious routing issue where divergent flow directions were not getting accumulated 50% of the time. Related to a division speed optimization that fell back on C-style modulus which differs from Python.

* InVEST SDR model thresholded slopes in terms of radians, not percent thus clipping the slope tightly between 0.001 and 1%.  The model now only has a lower threshold of 0.00005% for the IC_0 factor, and no other thresholds.  We believe this was an artifact left over from an earlier design of the model.


* Fixed a potential memory inefficiency in Wave Energy Model when computing the percentile rasters. Implemented a new memory efficient percentile algorithm and updated the outputs to reflect the new open source framework of the model. Now outputting csv files that describe the ranges and meaning of the percentile raster outputs.

* Fixed a bug in Habitat Quality where the future output "quality_out_f.tif" was not reflecting the habitat value given in the sensitivity table for the specified landcover types.


3.1.0 (2014-11-19)
------------------
InVEST 3.1.0 (http://www.naturalcapitalproject.org/download.html) is a major software and science milestone that includes an overhauled sedimentation model, long awaited fixes to exponential decay routines in habitat quality and pollination, and a massive update to the underlying hydrological routing routines.  The updated sediment model, called SDR (sediment delivery ratio), is part of our continuing effort to improve the science and capabilities of the InVEST tool suite.  The SDR model inputs are backwards comparable with the InVEST 3.0.1 sediment model with two additional global calibration parameters and removed the need for the retention efficiency parameter in the biophysical table; most users can run SDR directly with the data they have prepared for previous versions.  The biophysical differences between the models are described in a section within the SDR user's guide and represent a superior representation of the hydrological connectivity of the watershed, biophysical parameters that are independent of cell size, and a more accurate representation of sediment retention on the landscape.  Other InVEST improvements to include standard bug fixes, performance improvements, and usability features which in part are described below:

* InVEST Sediment Model has been replaced with the InVEST Sediment Delivery Ratio model.  See the SDR user's guide chapter for the difference between the two.
* Fixed an issue in the pollination model where the exponential decay function decreased too quickly.
* Fixed an issue in the habitat quality model where the exponential decay function decreased too quickly and added back linear decay as an option.
* Fixed an InVEST wide issue where some input rasters that were signed bytes did not correctly map to their negative nodata values.
* Hydropower input rasters have been normalized to the LULC size so sampling error is the same for all the input watersheds.
* Adding a check to make sure that input biophysical parameters to the water yield model do not exceed invalid scientific ranges.
* Added a check on nutrient retention in case the upstream water yield was less than 1 so that the log value did not go negative.  In that case we clamp upstream water yield to 0.
* A KeyError issue in hydropower was resolved that occurred when the input rasters were at such a coarse resolution that at least one pixel was completely contained in each watershed.  Now a value of -9999 will be reported for watersheds that don't contain any valid data.
* An early version of the monthly water yield model that was erroneously included in was in the installer; it was removed in this version.
* Python scripts necessary for running the ArcGIS version of Coastal Protection were missing.  They've since been added back to the distribution.
* Raster calculations are now processed by raster block sizes.  Improvements in raster reads and writes.
* Fixed an issue in the routing core where some wide DEMs would cause out of memory errors.
* Scenario generator marked as stable.
* Fixed bug in HRA where raster extents of shapefiles were not properly encapsulating the whole AOI.
* Fixed bug in HRA where any number of habitats over 4 would compress the output plots. Now extends the figure so that all plots are correctly scaled.
* Fixed a bug in HRA where the AOI attribute 'name' could not be an int. Should now accept any type.
* Fixed bug in HRA which re-wrote the labels if it was run immediately without closing the UI.
* Fixed nodata masking bug in Water Yield when raster extents were less than that covered by the watershed.
* Removed hydropower calibration parameter form water yield model.
* Models that had suffixes used to only allow alphanumeric characters.  Now all suffix types are allowed.
* A bug in the core platform that would occasionally cause routing errors on irregularly pixel sized rasters was fixed.  This often had the effect that the user would see broken streams and/or nodata values scattered through sediment or nutrient results.
* Wind Energy:
        * Added new framework for valuation component. Can now input a yearly price table that spans the lifetime of the wind farm. Also if no price table is made, can specify a price for energy and an annual rate of change.
        * Added new memory efficient distance transform functionality
        * Added ability to leave out 'landing points' in 'grid connection points' input. If not landing points are found, it will calculate wind farm directly to grid point distances
* Error message added in Wave Energy if clip shape has no intersection
* Fixed an issue where the data type of the nodata value in a raster might be different than the values in the raster.  This was common in the case of 64 bit floating point values as nodata when the underlying raster was 32 bit.  Now nodata values are cast to the underlying types which improves the reliability of many of the InVEST models.


3.0.1 (2014-05-19)
------------------
* Blue Carbon model released.

* HRA UI now properly reflects that the Resolution of Analysis is in meters, not meters squared, and thus will be applied as a side length for a raster pixel.

* HRA now accepts CSVs for ratings scoring that are semicolon separated as well as comma separated.

* Fixed a minor bug in InVEST's geoprocessing aggregate core that now consistently outputs correct zonal stats from the underlying pixel level hydro outputs which affects the water yield, sediment, and nutrient models.

* Added compression to InVEST output geotiff files.  In most cases this reduces output disk usage by a factor of 5.

* Fixed an issue where CSVs in the sediment model weren't open in universal line read mode.

* Fixed an issue where approximating whether pixel edges were the same size was not doing an approximately equal function.

* Fixed an issue that made the CV model crash when the coastline computed from the landmass didn't align perfectly with that defined in the geomorphology layer.

* Fixed an issue in the CV model where the intensity of local wave exposure was very low, and yielded zero local wave power for the majority of coastal segments.

* Fixed an issue where the CV model crashes if a coastal segment is at the edge of the shore exposure raster.

* Fixed the exposure of segments surrounded by land that appeared as exposed when their depth was zero.

* Fixed an issue in the CV model where the natural habitat values less than 5 were one unit too low, leading to negative habitat values in some cases.

* Fixed an exponent issue in the CV model where the coastal vulnerability index was raised to a power that was too high.

* Fixed a bug in the Scenic Quality model that prevented it from starting, as well as a number of other issues.

* Updated the pollination model to conform with the latest InVEST geoprocessing standards, resulting in an approximately 33% speedup.

* Improved the UI's ability to remember the last folder visited, and to have all file and folder selection dialogs have access to this information.

* Fixed an issue in Marine Water Quality where the UV points were supposed to be optional, but instead raised an exception when not passed in.

3.0.0 (2014-03-23)
------------------
The 3.0.0 release of InVEST represents a shift away from the ArcGIS to the InVEST standalone computational platform.  The only exception to this shift is the marine coastal protection tier 1 model which is still supported in an ArcGIS toolbox and has no InVEST 3.0 standalone at the moment.  Specific changes are detailed below

* A standalone version of the aesthetic quality model has been developed and packaged along with this release.  The standalone outperforms the ArcGIS equivalent and includes a valuation component.  See the user's guide for details.

* The core water routing algorithms for the sediment and nutrient models have been overhauled.  The routing algorithms now correctly adjust flow in plateau regions, address a bug that would sometimes not route large sections of a DEM, and has been optimized for both run time and memory performance.  In most cases the core d-infinity flow accumulation algorithm out performs TauDEM.  We have also packaged a simple interface to these algorithms in a standalone tool called RouteDEM; the functions can also be referenced from the scripting API in the invest_natcap.routing package.

* The sediment and nutrient models are now at a production level release.  We no longer support the ArcGIS equivalent of these models.

* The sediment model has had its outputs simplified with major changes including the removal of the 'pixel mean' outputs, a direct output of the pixel level export and retention maps, and a single output shapefile whose attribute table contains aggregations of sediment output values.  Additionally all inputs to the sediment biophysical table including p, c, and retention coefficients are now expressed as a proportion between 0 and 1; the ArcGIS model had previously required those inputs were integer values between 0 and 1000.  See the "Interpreting Results" section of sediment model for full details on the outputs.

* The nutrient model has had a similar overhaul to the sediment model including a simplified output structure with many key outputs contained in the attribute table of the shapefile.  Retention coefficients are also expressed in proportions between 0 and 1.  See the "Interpreting Results" section of nutrient model for full details on the outputs.

* Fixed a bug in Habitat Risk Assessment where the HRA module would incorrectly error if a criteria with a 0 score (meant to be removed from the assessment) had a 0 data quality or weight.

* Fixed a bug in Habitat Risk Assessment where the average E/C/Risk values across the given subregion were evaluating to negative numbers.

* Fixed a bug in Overlap Analysis where Human Use Hubs would error if run without inter-activity weighting, and Intra-Activity weighting would error if run without Human Use Hubs.

* The runtime performance of the hydropower water yield model has been improved.

* Released InVEST's implementation of the D-infinity flow algorithm in a tool called RouteDEM available from the start menu.

* Unstable version of blue carbon available.

* Unstable version of scenario generator available.

* Numerous other minor bug fixes and performance enhacnements.



2.6.0 (2013-12-16)
------------------
The 2.6.0 release of InVEST removes most of the old InVEST models from the Arc toolbox in favor of the new InVEST standalone models.  While we have been developing standalone equivalents for the InVEST Arc models since version 2.3.0, this is the first release in which we removed support for the deprecated ArcGIS versions after an internal review of correctness, performance, and stability on the standalones.  Additionally, this is one of the last milestones before the InVEST 3.0.0 release later next year which will transition InVEST models away from strict ArcGIS dependence to a standalone form.

Specifically, support for the following models have been moved from the ArcGIS toolbox to their Windows based standalones: (1) hydropower/water yield, (2) finfish aquaculture, (3) coastal protection tier 0/coastal vulnerability, (4) wave energy, (5) carbon, (6) habitat quality/biodiversity, (7) pollination, (8) timber, and (9) overlap analysis.  Additionally, documentation references to ArcGIS for those models have been replaced with instructions for launching standalone InVEST models from the Windows start menu.

This release also addresses minor bugs, documentation updates, performance tweaks, and new functionality to the toolset, including:

*  A Google doc to provide guidance for scripting the InVEST standalone models: https://docs.google.com/document/d/158WKiSHQ3dBX9C3Kc99HUBic0nzZ3MqW3CmwQgvAqGo/edit?usp=sharing

* Fixed a bug in the sample data that defined Kc as a number between 0 and 1000 instead of a number between 0 and 1.

* Link to report an issue now takes user to the online forums rather than an email address.

* Changed InVEST Sediment model standalone so that retention values are now between 0 and 1 instead of 0 and 100.

* Fixed a bug in Biodiversity where if no suffix were entered output filenames would have a trailing underscore (_) behind them.

* Added documentation to the water purification/nutrient retention model documentation about the standalone outputs since they differ from the ArcGIS version of the model.

* Fixed an issue where the model would try to move the logfile to the workspace after the model run was complete and Windows would erroneously report that the move failed.

* Removed the separation between marine and freshwater terrestrial models in the user's guide.  Now just a list of models.

* Changed the name of InVEST "Biodiversity" model to "Habitat Quality" in the module names, start menu, user's guide, and sample data folders.

* Minor bug fixes, performance enhancements, and better error reporting in the internal infrastructure.

* HRA risk in the unstable standalone is calculated differently from the last release. If there is no spatial overlap within a cell, there is automatically a risk of 0. This also applies to the E and C intermediate files for a given pairing. If there is no spatial overlap, E and C will be 0 where there is only habitat. However, we still create a recovery potential raster which has habitat- specific risk values, even without spatial overlap of a stressor. HRA shapefile outputs for high, medium, low risk areas are now calculated using a user-defined maximum number of overlapping stressors, rather than all potential stressors. In the HTML subregion averaged output, we now attribute what portion of risk to a habitat comes from each habitat-stressor pairing. Any pairings which don't overlap will have an automatic risk of 0.

* Major changes to Water Yield : Reservoir Hydropower Production. Changes include an alternative equation for calculating Actual Evapotranspiration (AET) for non-vegetated land cover types including wetlands. This allows for a more accurate representation of processes on land covers such as urban, water, wetlands, where root depth values aren't applicable. To differentiate between the two equations a column 'LULC_veg' has been added to the Biophysical table in Hydropower/input/biophysical_table.csv. In this column a 1 indicates vegetated and 0 indicates non-vegetated.

* The output structure and outputs have also change in Water Yield : Reservoir Hydropower Production. There is now a folder 'output' that contains all output files including a sub directory 'per_pixel' which has three pixel raster outputs. The subwatershed results are only calculated for the water yield portion and those results can be found as a shapefile, 'subwatershed_results.shp', and CSV file, 'subwatershed_results.csv'. The watershed results can be found in similar files: watershed_results.shp and watershed_results.csv. These two files for the watershed outputs will aggregate the Scarcity and Valuation results as well.

* The evapotranspiration coefficients for crops, Kc, has been changed to a decimal input value in the biophysical table. These values used to be multiplied by 1000 so that they were in integer format, that pre processing step is no longer necessary.

* Changing support from richsharp@stanford.edu to the user support forums at http://ncp-yamato.stanford.edu/natcapforums.

2.5.6 (2013-09-06)
------------------
The 2.5.6 release of InVEST that addresses minor bugs, performance
tweaks, and new functionality of the InVEST standalone models.
Including:

* Change the changed the Carbon biophysical table to use code field
  name from LULC to lucode so it is consistent with the InVEST water
  yield biophysical table.

* Added Monte Carlo uncertainty analysis and documentation to finfish
  aquaculture model.

* Replaced sample data in overlap analysis that was causing the model
  to crash.

* Updates to the overlap analysis user's guide.

* Added preprocessing toolkit available under
  C:\{InVEST install directory}\utils

* Biodiversity Model now exits gracefully if a threat raster is not
  found in the input folder.

* Wind Energy now uses linear (bilinear because its over 2D space?)
  interpolation.

* Wind Energy has been refactored to current API.

* Potential Evapotranspiration input has been properly named to
  Reference Evapotranspiration.

* PET_mn for Water Yield is now Ref Evapotranspiration times Kc
  (evapotranspiration coefficient).

* The soil depth field has been renamed 'depth to root restricting
  layer' in both the hydropower and nutrient retention models.

* ETK column in biophysical table for Water Yield is now Kc.

* Added help text to Timber model.

* Changed the behavior of nutrient retention to return nodata values
  when the mean runoff index is zero.

* Fixed an issue where the hydropower model didn't use the suffix
  inputs.

* Fixed a bug in Biodiversity that did not allow for numerals in the
  threat names and rasters.

* Updated routing algorithm to use a modern algorithm for plateau
  direction resolution.

* Fixed an issue in HRA where individual risk pixels weren't being
  calculated correctly.

* HRA will now properly detect in the preprocessed CSVs when criteria
  or entire habitat-stressor pairs are not desired within an
  assessment.

* Added an infrastructure feature so that temporary files are created
  in the user's workspace rather than at the system level
  folder.  This lets users work in a secondary workspace on a USB
  attached hard drive and use the space of that drive, rather than the
  primary operating system drive.

2.5.5 (2013-08-06)
------------------
The 2.5.5 release of InVEST that addresses minor bugs, performance
tweaks, and new functionality of the InVEST standalone models.  Including:

 * Production level release of the 3.0 Coastal Vulnerability model.
    - This upgrades the InVEST 2.5.4 version of the beta standalone CV
      to a full release with full users guide.  This version of the
      CV model should be used in all cases over its ArcGIS equivalent.

 * Production level release of the Habitat Risk Assessment model.
    - This release upgrades the InVEST 2.5.4 beta version of the
      standalone habitat risk assessment model. It should be used in
      all cases over its ArcGIS equivalent.

 * Uncertainty analysis in Carbon model (beta)
    - Added functionality to assess uncertainty in sequestration and
      emissions given known uncertainty in carbon pool stocks.  Users
      can now specify standard  deviations of carbon pools with
      normal distributions as well as desired uncertainty levels.
      New outputs include masks for regions which both sequester and
      emit carbon with a high probability of confidence.  Please see
      the "Uncertainty Analysis" section of the carbon user's guide
      chapter for more information.

 * REDD+ Scenario Analysis in Carbon model (beta)
    - Additional functionality to assist users evaluating REDD
      and REDD+ scenarios in the carbon model.  The uncertainty analysis
      functionality can also be used with these scenarios.
      Please see the "REDD Scenario Analysis" section of the
      carbon user's guide chapter for more information.

 * Uncertainty analysis in Finfish Aquaculture model (beta)
    - Additionally functionality to account for uncertainty in
      alpha and beta growth parameters as well as histogram
      plots showing the distribution of harvest weights and
      net present value.   Uncertainty analysis is performed
      through Monte Carlo runs that normally sample the
      growth parameters.

 * Streamlined Nutrient Retention model functionality
    - The nutrient retention module no longer requires users to explicitly
      run the water yield model.  The model now seamlessly runs water yield
      during execution.

 * Beta release of the recreation model
    - The recreation is available for beta use with limited documentation.

 * Full release of the wind energy model
    - Removing the 'beta' designation on the wind energy model.


Known Issues:

 * Flow routing in the standalone sediment and nutrient models has a
   bug that prevents routing in some (not all) landscapes.  This bug is
   related to resolving d-infinity flow directions across flat areas.
   We are implementing the solution in Garbrecht and Martx (1997).
   In the meanwhile the sediment and nutrient models are still marked
   as beta until this issue is resolved.

2.5.4 (2013-06-07)
------------------
This is a minor release of InVEST that addresses numerous minor bugs and performance tweaks in the InVEST 3.0 models.  Including:

 * Refactor of Wave Energy Model:
    - Combining the Biophysical and Valuation modules into one.
    - Adding new data for the North Sea and Australia
    - Fixed a bug where elevation values that were equal to or greater than zero
      were being used in calculations.
    - Fixed memory issues when dealing with large datasets.
    - Updated core functions to remove any use of depracated functions

 * Performance updates to the carbon model.

 * Nodata masking fix for rarity raster in Biodiversity Model.
    - When computing rarity from a base landuse raster and current or future
      landuse raster, the intersection of the two was not being properly taken.

 * Fixes to the flow routing algorithms in the sediment and nutrient
   retention models in cases where stream layers were burned in by ArcGIS
   hydro tools.  In those cases streams were at the same elevation and caused
   routing issues.

 * Fixed an issue that affected several InVEST models that occured
   when watershed polygons were too small to cover a pixel.  Excessively
   small watersheds are now handled correctly

 * Arc model deprecation.  We are deprecating the following ArcGIS versions
   of our InVEST models in the sense we recommend ALL users use the InVEST
   standalones over the ArcGIS versions, and the existing ArcGIS versions
   of these models will be removed entirely in the next release.

        * Timber
        * Carbon
        * Pollination
        * Biodiversity
        * Finfish Aquaculture

Known Issues:

 * Flow routing in the standalone sediment and nutrient models has a
   bug that prevents routing in several landscapes.  We're not
   certain of the nature of the bug at the moment, but we will fix by
   the next release.  Thus, sediment and nutrient models are marked
   as (beta) since in some cases the DEM routes correctly.

2.5.3 (2013-03-21)
------------------
This is a minor release of InVEST that fixes an issue with the HRA model that caused ArcGIS versions of the model to fail when calculating habitat maps for risk hotspots. This upgrade is strongly recommended for users of InVEST 2.5.1 or 2.5.2.

2.5.2 (2013-03-17)
------------------
This is a minor release of InVEST that fixes an issue with the HRA sample data that caused ArcGIS versions of the model to fail on the training data.  There is no need to upgrade for most users unless you are doing InVEST training.

2.5.1 (2013-03-12)
------------------
This is a minor release of InVEST that does not add any new models, but
does add additional functionality, stability, and increased performance to
one of the InVEST 3.0 standalones:

  - Pollination 3.0 Beta:
        - Fixed a bug where Windows users of InVEST could run the model, but
          most raster outputs were filled with nodata values.

Additionally, this minor release fixes a bug in the InVEST user interface where
collapsible containers became entirely non-interactive.

2.5.0 (2013-03-08)
------------------
This a major release of InVEST that includes new standalone versions (ArcGIS
is not required) our models as well as additional functionality, stability,
and increased performance to many of the existing models.  This release is
timed to support our group's annual training event at Stanford University.
We expect to release InVEST 2.5.1 a couple of weeks after to address any
software issues that arise during the training.  See the release notes
below for details of the release, and please contact richsharp@stanford.edu
for any issues relating to software:

  - *new* Sediment 3.0 Beta:
      - This is a standalone model that executes an order of magnitude faster
        than the original ArcGIS model, but may have memory issues with
	larger datasets. This fix is scheduled for the 2.5.1 release of InVEST.
      - Uses a d-infinity flow algorithm (ArcGIS version uses D8).
      - Includes a more accurate LS factor.
      - Outputs are now summarized by polygon rather than rasterized polygons.
        Users can view results directly as a table rather than sampling a
	GIS raster.
  - *new* Nutrient 3.0 Beta:
      - This is a standalone model that executes an order of magnitude faster
        than the original ArcGIS model, but may have memory issues with
	larger datasets. This fix is scheduled for the 2.5.1 release of InVEST.
      - Uses a d-infinity flow algorithm (ArcGIS version uses D8).
      - Includes a more accurate LS factor.
      - Outputs are now summarized by polygon rather than rasterized polygons.
        Users can view results directly as a table rather than sampling a
	GIS raster.
  - *new* Wind Energy:
      - A new offshore wind energy model.  This is a standalone-only model
        available under the windows start menu.
  - *new* Recreation Alpha:
      - This is a working demo of our soon to be released future land and near
        shore recreation model.  The model itself is incomplete and should only
	be used as a demo or by NatCap partners that know what they're doing.
  - *new* Habitat Risk Assessment 3.0 Alpha:
      - This is a working demo of our soon to be released 3.0 version of habitat
        risk assessment.  The model itself is incomplete and should only
	be used as a demo or by NatCap partners that know what they're doing.
	Users that need to use the habitat risk assessment should use the ArcGIS
	version of this model.

  - Improvements to the InVEST 2.x ArcGIS-based toolset:
      - Bug fixes to the ArcGIS based Coastal Protection toolset.

  - Removed support for the ArcGIS invest_VERSION.mxd map.  We expect to
    transition the InVEST toolset exclusive standalone tools in a few months.  In
    preparation of this we are starting to deprecate parts of our old ArcGIS
    toolset including this ArcMap document.  The InVEST ArcToolbox is still
    available in C:\InVEST_2_5_0\invest_250.tbx.

  - Known issues:

    - The InVEST 3.0 standalones generate open source GeoTiffs as
      outputs rather than the proprietary ESRI Grid format.  ArcGIS 9.3.1
      occasionally displays these rasters incorrectly.  We have found
      that these layers can be visualized in ArcGIS 9.3.1 by following
      convoluted steps: Right Click on the layer and select Properties; click on
      the Symbology tab; select Stretch, agree to calculate a histogram (this will
      create an .aux file that Arc can use for visualization), click "Ok", remove
      the raster from the layer list, then add it back. As an alternative, we
      suggest using an open source GIS Desktop Tool like Quantum GIS or ArcGIS
      version 10.0 or greater.

   - The InVEST 3.0 carbon model will generate inaccurate sequestration results
     if the extents of the current and future maps don't align.  This will be
     fixed in InVEST 2.5.1; in the meanwhile a workaround is to clip both LULCs
     so they have identical overlaps.

   - A user reported an unstable run of InVEST 3.0 water yield.  We are not
     certain what is causing the issue, but we do have a fix that will go out
     in InVEST 2.5.1.

   - At the moment the InVEST standalones do not run on Windows XP.  This appears
     to be related to an incompatibility between Windows XP and GDAL, the an open
     source gis library we use to create and read GIS data.  At the moment we are
     uncertain if we will be able to fix this bug in future releases, but will
     pass along more information in the future.

2.4.5 (2013-02-01)
------------------
This is a minor release of InVEST that does not add any new models, but
does add additional functionality, stability, and increased performance to
many of the InVEST 3.0 standalones:

  - Pollination 3.0 Beta:
      - Greatly improved memory efficiency over previous versions of this model.
      - 3.0 Beta Pollination Biophysical and Valuation have been merged into a
        single tool, run through a unified user interface.
      - Slightly improved runtime through the use of newer core InVEST GIS libraries.
      - Optional ability to weight different species individually.  This feature
        adds a column to the Guilds table that allows the user to specify a
        relative weight for each species, which will be used before combining all
        species supply rasters.
      - Optional ability to aggregate pollinator abundances at specific points
        provided by an optional points shapefile input.
      - Bugfix: non-agricultural pixels are set to a value of 0.0 to indicate no
        value on the farm value output raster.
      - Bugfix: sup_val_<beename>_<scenario>.tif rasters are now saved to the
        intermediate folder inside the user's workspace instead of the output
        folder.
  - Carbon Biophysical 3.0 Beta:
        * Tweaked the user interface to require the user to
          provide a future LULC raster when the 'Calculate Sequestration' checkbox
          is checked.
        * Fixed a bug that restricted naming of harvest layers.  Harvest layers are
          now selected simply by taking the first available layer.
  - Better memory efficiency in hydropower model.
  - Better support for unicode filepaths in all 3.0 Beta user interfaces.
  - Improved state saving and retrieval when loading up previous-run parameters
    in all 3.0 Beta user interfaces.
  - All 3.0 Beta tools now report elapsed time on completion of a model.
  - All 3.0 Beta tools now provide disk space usage reports on completion of a
    model.
  - All 3.0 Beta tools now report arguments at the top of each logfile.
  - Biodiversity 3.0 Beta: The half-saturation constant is now allowed to be a
    positive floating-point number.
  - Timber 3.0 Beta: Validation has been added to the user interface for this
    tool for all tabular and shapefile inputs.
  - Fixed some typos in Equation 1 in the Finfish Aquaculture user's guide.
  - Fixed a bug where start menu items were not getting deleted during an InVEST
    uninstall.
  - Added a feature so that if the user selects to download datasets but the
    datasets don't successfully download the installation alerts the user and
    continues normally.
  - Fixed a typo with tau in aquaculture guide, originally said 0.8, really 0.08.

  - Improvements to the InVEST 2.x ArcGIS-based toolset:
      - Minor bugfix to Coastal Vulnerability, where an internal unit of
        measurements was off by a couple digits in the Fetch Calculator.
      - Minor fixes to various helper tools used in InVEST 2.x models.
      - Outputs for Hargreaves are now saved as geoTIFFs.
      - Thornwaite allows more flexible entering of hours of sunlight.

2.4.4 (2012-10-24)
------------------
- Fixes memory errors experienced by some users in the Carbon Valuation 3.0 Beta model.
- Minor improvements to logging in the InVEST User Interface
- Fixes an issue importing packages for some officially-unreleased InVEST models.

2.4.3 (2012-10-19)
------------------
- Fixed a minor issue with hydropower output vaulation rasters whose statistics were not pre-calculated.  This would cause the range in ArcGIS to show ther rasters at -3e38 to 3e38.
- The InVEST installer now saves a log of the installation process to InVEST_<version>\install_log.txt
- Fixed an issue with Carbon 3.0 where carbon output values were incorrectly calculated.
- Added a feature to Carbon 3.0 were total carbon stored and sequestered is output as part of the running log.
- Fixed an issue in Carbon 3.0 that would occur when users had text representations of floating point numbers in the carbon pool dbf input file.
- Added a feature to all InVEST 3.0 models to list disk usage before and after each run and in most cases report a low free space error if relevant.

2.4.2 (2012-10-15)
------------------
- Fixed an issue with the ArcMap document where the paths to default data were not saved as relative paths.  This caused the default data in the document to not be found by ArcGIS.
- Introduced some more memory-efficient processing for Biodiversity 3.0 Beta.  This fixes an out-of-memory issue encountered by some users when using very large raster datasets as inputs.

2.4.1 (2012-10-08)
------------------
- Fixed a compatibility issue with ArcGIS 9.3 where the ArcMap and ArcToolbox were unable to be opened by Arc 9.3.

2.4.0 (2012-10-05)
------------------
Changes in InVEST 2.4.0

General:

This is a major release which releases two additional beta versions of the
InVEST models in the InVEST 3.0 framework.  Additionally, this release
introduces start menu shortcuts for all available InVEST 3.0 beta models.
Existing InVEST 2.x models can still be found in the included Arc toolbox.

Existing InVEST models migrated to the 3.0 framework in this release
include:

- Biodiversity 3.0 Beta
    - Minor bug fixes and usability enhancements
    - Runtime decreased by a factor of 210
- Overlap Analysis 3.0 Beta
    - In most cases runtime decreased by at least a factor of 15
    - Minor bug fixes and usability enhancements
    - Split into two separate tools:
        * Overlap Analysis outputs rasters with individually-weighted pixels
        * Overlap Analysis: Management Zones produces a shapefile output.
    - Updated table format for input activity CSVs
    - Removed the "grid the seascape" step

Updates to ArcGIS models:

- Coastal vulnerability
    - Removed the "structures" option
    - Minor bug fixes and usability enhancements
- Coastal protection (erosion protection)
    - Incorporated economic valuation option
    - Minor bug fixes and usability enhancements

Additionally there are a handful of minor fixes and feature
enhancements:

- InVEST 3.0 Beta standalones (identified by a new InVEST icon) may be run
  from the Start Menu (on windows navigate to
  Start Menu -> All Programs -> InVEST 2.4.0
- Bug fixes for the calculation of raster statistics.
- InVEST 3.0 wave energy no longer requires an AOI for global runs, but
  encounters memory issues on machines with less than 4GB of RAM.  This
  is a known issue that will be fixed in a minor release.
- Minor fixes to several chapters in the user's guide.
- Minor bug fix to the 3.0 Carbon model: harvest maps are no longer required
  inputs.
- Other minor bug fixes and runtime performance tweaks in the 3.0 framework.
- Improved installer allows users to remove InVEST from the Windows Add/Remove
  programs menu.
- Fixed a visualization bug with wave energy where output rasters did not have the min/max/stdev calculations on them.  This made the default visualization in arc be a gray blob.

2.3.0 (2012-08-02)
------------------
Changes in InVEST 2.3.0

General:

This is a major release which releases several beta versions of the
InVEST models in the InVEST 3.0 framework.  These models run as
standalones, but a GIS platform is needed to edit and view the data
inputs and outputs.  Until InVEST 3.0 is released the original ArcGIS
based versions of these tools will remain the release.

Existing InVEST models migrated to the 3.0 framework in this release
include:

- Reservoir Hydropower Production 3.0 beta
    - Minor bug fixes.
- Finfish Aquaculture
    - Minor bug fixes and usability enhancements.
- Wave Energy 3.0 beta
    - Runtimes for non-global runs decreased by a factor of 7
    - Minor bugs in interpolation that exist in the 2.x model is fixed in
      3.0 beta.
- Crop Pollination 3.0 beta
    - Runtimes decreased by a factor of over 10,000

This release also includes the new models which only exist in the 3.0
framework:

- Marine Water Quality 3.0 alpha with a preliminary  user's guide.

InVEST models in the 3.0 framework from previous releases that now
have a standalone executable include:

- Managed Timber Production Model
- Carbon Storage and Sequestration

Additionally there are a handful of other minor fixes and feature
enhancements since the previous release:

- Minor bug fix to 2.x sedimentation model that now correctly
  calculates slope exponentials.
- Minor fixes to several chapters in the user's guide.
- The 3.0 version of the Carbon model now can value the price of carbon
  in metric tons of C or CO2.
- Other minor bug fixes and runtime performance tweaks in the 3.0 framework.

2.2.2 (2012-03-03)
------------------
Changes in InVEST 2.2.2

General:

This is a minor release which fixes the following defects:

-Fixed an issue with sediment retention model where large watersheds
 allowed loading per cell was incorrectly rounded to integer values.

-Fixed bug where changing the threshold didn't affect the retention output
 because function was incorrectly rounded to integer values.

-Added total water yield in meters cubed to to output table by watershed.

-Fixed bug where smaller than default (2000) resolutions threw an error about
 not being able to find the field in "unitynew".  With non-default resolution,
 "unitynew" was created without an attribute table, so one was created by
 force.

-Removed mention of beta state and ecoinformatics from header of software
 license.

-Modified overlap analysis toolbox so it reports an error directly in the
 toolbox if the workspace name is too long.

2.2.1 (2012-01-26)
------------------
Changes in InVEST 2.2.1

General:

This is a minor release which fixes the following defects:

-A variety of miscellaneous bugs were fixed that were causing crashes of the Coastal Protection model in Arc 9.3.
-Fixed an issue in the Pollination model that was looking for an InVEST1005 directory.
-The InVEST "models only" release had an entry for the InVEST 3.0 Beta tools, but was missing the underlying runtime.  This has been added to the models only 2.2.1 release at the cost of a larger installer.
-The default InVEST ArcMap document wouldn't open in ArcGIS 9.3.  It can now be opened by Arc 9.3 and above.
-Minor updates to the Coastal Protection user's guide.

2.2.0 (2011-12-22)
------------------
In this release we include updates to the habitat risk assessment
model, updates to Coastal Vulnerability Tier 0 (previously named
Coastal Protection), and a new tier 1 Coastal Vulnerability tool.
Additionally, we are releasing a beta version of our 3.0 platform that
includes the terrestrial timber and carbon models.

See the "Marine Models" and "InVEST 3.0 Beta" sections below for more details.

**Marine Models**

1. Marine Python Extension Check

   This tool has been updated to include extension requirements for the new
   Coastal Protection T1 model.  It also reflects changes to the Habitat Risk
   Assessment and Coastal Protection T0 models, as they no longer require the
   PythonWin extension.

2. Habitat Risk Assessment (HRA)

   This model has been updated and is now part of three-step toolset.  The
   first step is a new Ratings Survey Tool which eliminates the need for
   Microsoft Excel when users are providing habitat-stressor ratings.  This
   Survey Tool now allows users to up- and down-weight the importance of
   various criteria.  For step 2, a copy of the Grid the Seascape tool has been
   placed in the HRA toolset.  In the last step, users will run the HRA model
   which includes the following updates:

   - New habitat outputs classifying risk as low, medium, and high
   - Model run status updates (% complete) in the message window
   - Improved habitat risk plots embedded in the output HTML

3. Coastal Protection

   This module is now split into sub-models, each with two parts.  The first
   sub-model is Coastal Vulnerability (Tier 0) and the new addition is Coastal
   Protection (Tier 1).

   Coastal Vulnerability (T0)
   Step 1) Fetch Calculator - there are no updates to this tool.
   Step 2) Vulnerability Index

   - Wave Exposure: In this version of the model, we define wave exposure for
     sites facing the open ocean as the maximum of the weighted average of
     wave's power coming from the ocean or generated by local winds.  We
     weight wave power coming from each of the 16 equiangular sector by the
     percent of time that waves occur in that sector, and based on whether or
     not fetch in that sector exceeds 20km.  For sites that are sheltered, wave
     exposure is the average of wave power generated by the local storm winds
     weighted by the percent occurrence of those winds in each sector.  This
     new method takes into account the seasonality of wind and wave patterns
     (storm waves generally come from a preferential direction), and helps
     identify regions that are not exposed to powerful waves although they are
     open to the ocean (e.g. the leeside of islands).

   - Natural Habitats: The ranking is now computed using the rank of all
     natural habitats present in front of a segment, and we weight the lowest
     ranking habitat 50% more than all other habitats.  Also, rankings and
     protective distance information are to be provided by CSV file instead of
     Excel.  With this new method, shoreline segments that have more habitats
     than others will have a lower risk of inundation and/or erosion during
     storms.

   - Structures: The model has been updated to now incorporate the presence of
     structures by decreasing the ranking of shoreline segments that adjoin
     structures.

   Coastal Protection (T1) - This is a new model which plots the amount of
   sandy beach erosion or consolidated bed scour that backshore regions
   experience in the presence or absence of natural habitats.  It is composed
   of two steps: a Profile Generator and Nearshore Waves and Erosion.  It is
   recommended to run the Profile Generator before the Nearshore Waves and
   Erosion model.

   Step 1) Profile Generator:  This tool helps the user generate a 1-dimensional
   bathymetric and topographic profile perpendicular to the shoreline at the
   user-defined location.  This model provides plenty of guidance for building
   backshore profiles for beaches, marshes and mangroves.  It will help users
   modify bathymetry profiles that they already have, or can generate profiles
   for sandy beaches if the user has not bathymetric data.  Also, the model
   estimates and maps the location of natural habitats present in front of the
   region of interest.  Finally, it provides sample wave and wind data that
   can be later used in the Nearshore Waves and Erosion model, based on
   computed fetch values and default Wave Watch III data.

   Step 2) Nearshore Waves and Erosion: This model estimates profiles of beach
   erosion or values of rates of consolidated bed scour at a site as a function
   of the type of habitats present in the area of interest.  The model takes
   into account the protective effects of vegetation, coral and oyster reefs,
   and sand dunes.  It also shows the difference of protection provided when
   those habitats are present, degraded, or gone.

4. Aesthetic Quality

   This model no longer requires users to provide a projection for Overlap
   Analysis.  Instead, it uses the projection from the user-specified Area of
   Interest (AOI) polygon.  Additionally, the population estimates for this
   model have been fixed.

**InVEST 3.0 Beta**

The 2.2.0 release includes a preliminary version of our InVEST 3.0 beta
platform.  It is included as a toolset named "InVEST 3.0 Beta" in the
InVEST220.tbx.  It is currently only supported with ArcGIS 10.  To launch
an InVEST 3.0 beta tool, double click on the desired tool in the InVEST 3.0
toolset then click "Ok" on the Arc toolbox screen that opens. The InVEST 3.0
tool panel has inputs very similar to the InVEST 2.2.0 versions of the tools
with the following modifications:

InVEST 3.0 Carbon:
  * Fixes a minor bug in the 2.2 version that ignored floating point values
    in carbon pool inputs.
  * Separation of carbon model into a biophysical and valuation model.
  * Calculates carbon storage and sequestration at the minimum resolution of
    the input maps.
  * Runtime efficiency improved by an order of magnitude.
  * User interface streamlined including dynamic activation of inputs based
    on user preference, direct link to documentation, and recall of inputs
    based on user's previous run.

InVEST 3.0 Timber:
  * User interface streamlined including dynamic activation of inputs based
    on user preference, direct link to documentation, and recall of inputs
    based on user's previous run.


2.1.1 (2011-10-17)
------------------
Changes in InVEST 2.1.1

General:

This is a minor release which fixes the following defects:

-A truncation error was fixed on nutrient retention and sedimentation model that involved division by the number of cells in a watershed.  Now correctly calculates floating point division.
-Minor typos were fixed across the user's guide.

2.1 Beta (2011-05-11)
---------------------
Updates to InVEST Beta

InVEST 2.1 . Beta

Changes in InVEST 2.1

General:

1.	InVEST versioning
We have altered our versioning scheme.  Integer changes will reflect major changes (e.g. the addition of marine models warranted moving from 1.x to 2.0).  An increment in the digit after the primary decimal indicates major new features (e.g the addition of a new model) or major revisions.  For example, this release is numbered InVEST 2.1 because two new models are included).  We will add another decimal to reflect minor feature revisions or bug fixes.  For example, InVEST 2.1.1 will likely be out soon as we are continually working to improve our tool.
2.	HTML guide
With this release, we have migrated the entire InVEST users. guide to an HTML format.  The HTML version will output a pdf version for use off-line, printing, etc.


**MARINE MODELS**

1.Marine Python Extension Check

-This tool has been updated to allow users to select the marine models they intend to run.  Based on this selection, it will provide a summary of which Python and ArcGIS extensions are necessary and if the Python extensions have been successfully installed on the user.s machine.

2.Grid the Seascape (GS)

-This tool has been created to allow marine model users to generate an seascape analysis grid within a specified area of interest (AOI).

-It only requires an AOI and cell size (in meters) as inputs, and produces a polygon grid which can be used as inputs for the Habitat Risk Assessment and Overlap Analysis models.

3. Coastal Protection

- This is now a two-part model for assessing Coastal Vulnerability.  The first part is a tool for calculating fetch and the second maps the value of a Vulnerability Index, which differentiates areas with relatively high or low exposure to erosion and inundation during storms.

- The model has been updated to now incorporate coastal relief and the protective influence of up to eight natural habitat input layers.

- A global Wave Watch 3 dataset is also provided to allow users to quickly generate rankings for wind and wave exposure worldwide.

4. Habitat Risk Assessment (HRA)

This new model allows users to assess the risk posed to coastal and marine habitats by human activities and the potential consequences of exposure for the delivery of ecosystem services and biodiversity.  The HRA model is suited to screening the risk of current and future human activities in order to prioritize management strategies that best mitigate risk.

5. Overlap Analysis

This new model maps current human uses in and around the seascape and summarizes the relative importance of various regions for particular activities.  The model was designed to produce maps that can be used to identify marine and coastal areas that are most important for human use, in particular recreation and fisheries, but also other activities.

**FRESHWATER MODELS**

All Freshwater models now support ArcMap 10.


Sample data:

1. Bug fix for error in Water_Tables.mdb Biophysical table where many field values were shifted over one column relative to the correct field name.

2. Bug fix for incorrect units in erosivity layer.


Hydropower:

1.In Water Yield, new output tables have been added containing mean biophysical outputs (precipitation, actual and potential evapotranspiration, water yield)  for each watershed and sub-watershed.


Water Purification:

1. The Water Purification Threshold table now allows users to specify separate thresholds for nitrogen and phosphorus.   Field names thresh_n and thresh_p replace the old ann_load.

2. The Nutrient Retention output tables nutrient_watershed.dbf and nutrient_subwatershed.dbf now include a column for nutrient retention per watershed/sub-watershed.

3. In Nutrient Retention, some output file names have changed.

4. The user's guide has been updated to explain more accurately the inclusion of thresholds in the biophysical service estimates.


Sedimentation:

1. The Soil Loss output tables sediment_watershed.dbf and sediment_subwatershed.dbf now include a column for sediment retention per watershed/sub-watershed.

2. In Soil Loss, some output file names have changed.

3. The default input value for Slope Threshold is now 75.

4. The user's guide has been updated to explain more accurately the inclusion of thresholds in the biophysical service estimates.

5. Valuation: Bug fix where the present value was not being applied correctly.





2.0 Beta (2011-02-14)
---------------------
Changes in InVEST 2.0

InVEST 1.005 is a minor release with the following modification:

1. Aesthetic Quality

    This new model allows users to determine the locations from which new nearshore or offshore features can be seen.  It generates viewshed maps that can be used to identify the visual footprint of new offshore development.


2. Coastal Vulnerability

    This new model produces maps of coastal human populations and a coastal exposure to erosion and inundation index map.  These outputs can be used to understand the relative contributions of different variables to coastal exposure and to highlight the protective services offered by natural habitats.


3. Aquaculture

    This new model is used to evaluate how human activities (e.g., addition or removal of farms, changes in harvest management practices) and climate change (e.g., change in sea surface temperature) may affect the production and economic value of aquacultured Atlantic salmon.


4. Wave Energy

    This new model provides spatially explicit information, showing potential areas for siting Wave Energy conversion (WEC) facilities with the greatest energy production and value.  This site- and device-specific information for the WEC facilities can then be used to identify and quantify potential trade-offs that may arise when siting WEC facilities.


5. Avoided Reservoir Sedimentation

    - The name of this model has been changed to the Sediment Retention model.

    - We have added a water quality valuation model for sediment retention. The user now has the option to select avoided dredge cost analysis, avoided water treatment cost analysis or both.  The water quality valuation approach is the same as that used in the Water Purification: Nutrient Retention model.

    - The threshold information for allowed sediment loads (TMDL, dead volume, etc.) are now input in a stand alone table instead of being included in the valuation table. This adjusts the biophysical service output for any social allowance of pollution. Previously, the adjustment was only done in the valuation model.

    - The watersheds and sub-watershed layers are now input as shapefiles instead of rasters.

    - Final outputs are now aggregated to the sub-basin scale. The user must input a sub-basin shapefile. We provide the Hydro 1K dataset as a starting option. See users guide for changes to many file output names.

    - Users are strongly advised not to interpret pixel-scale outputs for hydrological understanding or decision-making of any kind. Pixel outputs should only be used for calibration/validation or model checking.


6. Hydropower Production

    - The watersheds and sub-watershed layers are now input as shapefiles instead of rasters.

    - Final outputs are now aggregated to the sub-basin scale. The user must input a sub-basin shapefile. We provide the Hydro 1K dataset as a starting option. See users guide for changes to many file output names.

    - Users are strongly advised not to interpret pixel-scale outputs for hydrological understanding or decision-making of any kind. Pixel outputs should only be used for calibration/validation or model checking.

    - The calibration constant for each watershed is now input in a stand-alone table instead of being included in the valuation table. This makes running the water scarcity model simpler.


7. Water Purification: Nutrient Retention

    - The threshold information for allowed pollutant levels (TMDL, etc.) are now input in a stand alone table instead of being included in the valuation table. This adjusts the biophysical service output for any social allowance of pollution. Previously, the adjustment was only done in the valuation model.

    - The watersheds and sub-watershed layers are now input as shapefiles instead of rasters.

    - Final outputs are now aggregated to the sub-basin scale. The user must input a sub-basin shapefile. We provide the Hydro 1K dataset as a starting option. See users guide for changes to many file output names.

    - Users are strongly advised not to interpret pixel-scale outputs for hydrological understanding or decision-making of any kind. Pixel outputs should only be used for calibration/validation or model checking.


8. Carbon Storage and Sequestration

    The model now outputs an aggregate sum of the carbon storage.


9. Habitat Quality and Rarity

    This model had an error while running ReclassByACII if the land cover codes were not sorted alphabetically.  This has now been corrected and it sorts the reclass file before running the reclassification

    The model now outputs an aggregate sum of the habitat quality.

10. Pollination

    In this version, the pollination model accepts an additional parameter which indicated the proportion of a crops yield that is attributed to wild pollinators.


