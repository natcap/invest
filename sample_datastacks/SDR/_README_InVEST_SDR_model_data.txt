
InVEST freshwater model sample data information and sources
-----------------------------------------------------------

These sample data are from Kenya, where the Natural Capital Project worked with partners to inform the creation of a water fund for the city of Nairobi. It is a small sub-watershed called Gura, within the larger area covered by the project. Below are brief notes on the sources of these data. 

Please note that these data are provided as an example only, to help with learning InVEST freshwater models. They should not be treated as authoritative, nor used as is within your own projects. This is particularly true for the biophysical table coefficients.  

See the InVEST User Guide for more information about this model and its data requirements: http://releases.naturalcapitalproject.org/invest-userguide/latest/


Digital elevation model - DEM_gura.tif: NASA Shuttle Radar Data Topography Mission (SRTM) 30m elevation data, resampled to the same resolution as the land use map (15m).

Rainfall erosivity index - erosivity_gura.tif: Derived from annual average precipitation using equations from "Rainfall Erosivity in East Africa" by T.R. Moore; Geografiska Annaler. Series A, Physical Geography, Vol. 61, No. 3/4 (1979), pp. 147-156

Soil erodibility - erodibility_gura.tif: Derived from the Soil and Terrain Database (SOTER) for Upper Tana River Catchment, version 1.1. https://data.isric.org/geonetwork/srv/api/records/ce32091e-006d-4438-8e03-cf7b4c500df7

Land use/Land cover - land_use_gura.tif: The Nature Conservancy (TNC) carried out a detailed update of the Africover land use maps, using satellite imagery, detailed maps from stakeholders and ground truth points. 
- Two additional files are provided for convenient display of the land use map: land_use_gura.lyr (for ArcGIS) and land_use_gura.qlr (for QGIS). These files are not used by the InVEST model.

Watersheds - watershed_gura.shp or subwatersheds_gura.shp: Derived from the DEM using the InVEST DelineateIt tool.

Biophysical table - biophysical_table_gura.csv: Local values were not available, so these are globally-averaged values from a variety of literature sources. Some of these sources came from the InVEST parameter database (available from https://naturalcapitalproject.stanford.edu/software/invest.) 
- Note that we do NOT recommend using these table values directly for your own projects, use them for testing/example purposes only.
- This table contains coefficients for 3 models - Annual Water Yield, SDR and NDR. 

Threshold flow accumulation: A value of 1000 works well.

Borselli k Parameter: Default value is 2

Borselli IC0 Parameter: Default value is 0.5

Max SDR value: Default value is 0.8


