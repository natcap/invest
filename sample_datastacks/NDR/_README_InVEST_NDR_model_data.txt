
InVEST freshwater model sample data information and sources
-----------------------------------------------------------

These sample data are from Kenya, where the Natural Capital Project worked with partners to inform the creation of a water fund for the city of Nairobi. It is a small sub-watershed called Gura, within the larger area covered by the project. Below are brief notes on the sources of these data. 

Please note that these data are provided as an example only, to help with learning InVEST freshwater models. They should not be treated as authoritative, nor used as is within your own projects. This is particularly true for the biophysical table coefficients. 

See the InVEST User Guide for more information about this model and its data requirements: http://releases.naturalcapitalproject.org/invest-userguide/latest/


Digital elevation model - DEM_gura.tif: NASA Shuttle Radar Data Topography Mission (SRTM) 30m elevation data, resampled to the same resolution as the land use map (15m).

Land use/land cover - land_use_gura.tif: The Nature Conservancy (TNC) carried out a detailed update of the Africover land use maps, using satellite imagery, detailed maps from stakeholders and ground truth points. 
- Two additional files are provided for convenient display of the land use map: land_use_gura.lyr (for ArcGIS) and land_use_gura.qlr (for QGIS). These files are not used by the InVEST model.

Nutrient runoff proxy - precipitation_gura.tif: Several partners in Kenya provided rainfall data for meteorological stations within the watersheds. These data were processed into annual average values for each station, then interpolated to create a raster that covers the study area. This is the same as the Annual Water Yield model precipitation input.

Watersheds - watershed_gura.shp or subwatersheds_gura.shp: Derived from the DEM using the InVEST DelineateIt tool.

Biophysical table - biophysical_table_gura.csv: Local values were not available, so these are globally-averaged values from a variety of literature sources. Some of these sources came from the InVEST parameter database (available from https://naturalcapitalproject.stanford.edu/software/invest), or default starting values mentioned in the User Guide. We do NOT recommend using these table values directly for your own projects, use them for testing/example purposes only.

Threshold flow accumulation: A value of 1000 works well.

Borselli k Parameter: Default value is 2

Subsurface Critical Length: A global average value that can be used as an example is 200m.

Subsurface Maximum Retention Efficiency: A global average value that can be used as an example is 0.8 (=80% efficiency.)




