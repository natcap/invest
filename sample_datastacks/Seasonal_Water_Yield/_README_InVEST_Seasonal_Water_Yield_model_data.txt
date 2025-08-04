
InVEST freshwater model sample data information and sources
-----------------------------------------------------------

These sample data are from Kenya, where the Natural Capital Project worked with partners to inform the creation of a water fund for the city of Nairobi. It is a small sub-watershed called Gura, within the larger area covered by the project. Below are brief notes on the sources of these data. 

Please note that these data are provided as an example only, to help with learning InVEST freshwater models. They should not be treated as authoritative, nor used as is within your own projects. This is particularly true for the biophysical table coefficients.  

See the InVEST User Guide for more information about this model and its data requirements: http://releases.naturalcapitalproject.org/invest-userguide/latest/


Threshold flow accumulation: A value of 1000 works well.

ET0 directory - ETO_monthly: Derived from CHELSA monthly precipitation and min/max temperature using the Modified Hargreaves method.

Precipitation directory - Precipitation_monthly: CHELSA monthly precipitation, version 1.2; Karger, D.N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R.W., Zimmermann, N.E., Linder, H.P. & Kessler, M. (2017) Climatologies at high resolution for the earth’s land surface areas. Scientific Data 4, 170122. http://chelsa-climate.org/downloads/

Digital elevation model - DEM_gura.tif: NASA Shuttle Radar Data Topography Mission (SRTM) 30m elevation data, resampled to the same resolution as the land use map (15m).

Land use/Land cover - land_use_gura.tif: The Nature Conservancy (TNC) carried out a detailed update of the Africover land use maps, using satellite imagery, detailed maps from stakeholders and ground truth points. 
- Two additional files are provided for convenient display of the land use map: land_use_gura.lyr (for ArcGIS) and land_use_gura.qlr (for QGIS). These files are not used by the InVEST model.

Soil group - soil_group_gura.tif: FutureWater Global Maps of Soil Hydraulic Properties, Version 1.2, 2016; https://www.futurewater.eu/2015/07/soil-hydraulic-properties/

AOI/Watershed - watershed_gura.shp or subwatersheds_gura.shp: Derived from the DEM using the InVEST DelineateIt tool.

Biophysical table - biophysical_table_gura_SWY.csv: Local values were not available, so these are globally-averaged values from a variety of literature sources. 
- We do NOT recommend using these table values directly for your own projects, use them for testing/example purposes only. 

Rain events table - rain_events_gura.csv: Obtained from the IWMI Online Climate Summary Service Portal; International Water Management Institute; http://wcatlas.iwmi.org/

Climate zone table - climate_zone_table_gura.csv: Values for cz_id 1 are the same as in the Rain events table above (from IWMI.) Values for cz_id 2 are based on zone 1 but made up for sample purposes.

Climate zone - climate_zones_gura.tif: Made up for sample purposes.

alpha_m Parameter: Default value is 1/12

beta_i Parameter: Default value is 1

gamma Parameter: Default value is 1






