
InVEST freshwater model sample data information and sources
-----------------------------------------------------------

These sample data are from Kenya, where the Natural Capital Project worked with partners to inform the creation of a water fund for the city of Nairobi. It is a small sub-watershed called Gura, within the larger area covered by the project. Below are brief notes on the sources of these data. 

Please note that these data are provided as an example only, to help with learning InVEST freshwater models. They should not be treated as authoritative, nor used as is within your own projects. This is particularly true for the biophysical table coefficients. 

See the InVEST User Guide for more information about this model and its data requirements: http://releases.naturalcapitalproject.org/invest-userguide/latest/


Precipitation - precipitation_gura.tif: Several partners in Kenya provided rainfall data for meteorological stations within the watersheds. These data were processed into annual average values for each station, then interpolated to create a raster that covers the study area.

Reference evapotranspiration - reference_ET_gura.tif: Derived from annual average precipitation using the Modified Hargreaves method.

Depth to root restricting layer - depth_to_root_restricting_layer_gura.tif: Derived from the Soil and Terrain Database (SOTER) for Upper Tana River Catchment, version 1.1. https://data.isric.org/geonetwork/srv/api/records/ce32091e-006d-4438-8e03-cf7b4c500df7

Plant available water fraction - plant_available_water_fraction_gura.tif: Derived from the Soil and Terrain Database (SOTER) for Upper Tana River Catchment, version 1.1. https://data.isric.org/geonetwork/srv/api/records/ce32091e-006d-4438-8e03-cf7b4c500df7

Land Use - land_use_gura.tif: The Nature Conservancy (TNC) carried out a detailed update of the Africover land use maps, using satellite imagery, detailed maps from stakeholders and ground truth points. 
- Two additional files are provided for convenient display of the land use map: land_use_gura.lyr (for ArcGIS) and land_use_gura.qlr (for QGIS). These files are not used by the InVEST model.

Watersheds - watershed_gura.shp: Derived from the DEM using the InVEST DelineateIt tool.

Sub-watersheds - subwatersheds_gura.shp: Derived from the DEM using the InVEST DelineateIt tool.

Biophysical table - biophysical_table_gura.csv: Local values were not available, so these are globally-averaged values from a variety of literature sources. As an example of one global source, some Kc values are from:
Allen, R.G., L.S. Pereira, D. Raes, M. Smith (1998). Crop evapotranspiration - Guidelines for computing crop water requirements. Irrigation and drainage paper 56. Rome, Food and Agriculture Organization of the United Nations (FAO).
- Note that we do NOT recommend using these table values directly for your own projects, use them for testing/example purposes only.
- This table contains coefficients for 3 models - Annual Water Yield, SDR and NDR.

Z parameter: A value of 5 is fine for testing purposes.

