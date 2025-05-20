# ADR-0004: Remove Wind Energy Raster Outputs

Author: Megan Nissel

Science Lead: Rob Griffin

## Context

The Wind Energy model has three major data inputs required for all runs: a Wind Data Points CSV, containing Weibull parameters for each wind data point; a Bathymetry raster; and a CSV of global wind energy infrastructure parameters. Within the wind data points CSV, each row represents a discrete geographic coordinate point. During the model run, this CSV gets converted to a point vector and then the data are interpolated onto rasters.

When run without the valuation component, the model outputs the following:
- `density_W_per_m2.tif`: a raster representing power density (W/m^2) centered on a pixel.
- `harvested_energy_MWhr_per_yr.tif`: a raster representing the annual harvested energy from a farm centered on that pixel.
- `wind_energy_points.shp`: a vector (with points corresponding to those in the input Wind Energy points CSV) that summarizes the outputs of the two rasters.

When run with the valuation component, the model outputs three additional rasters in addition to the two listed above: `carbon_emissions_tons.tif`, `levelized_cost_price_per_kWh.tif`, and `npv.tif`. These values are not currently summarized in `wind_energy_points.shp`.

Users noticed the raster outputs included data in areas outside of those covered by the input Wind Data, resulting from the model's method of interpolating the vector data to the rasters. This led to a larger discussion around the validity of the interpolated raster results.

## Decision

Based on Rob's own use of the model, and review and evaluation of the problem, the consensus is that the model's current use of interpolation introduces too many potential violations of the constraints of the model (e.g. interpolating over areas that are invlaid due to ocean depth or distance from shore, or are outside of the areas included in the input wind speed data) and requires assumptions that may not be helpful for users. Rob therefore recommended removing the raster outputs entirely and retaining the associated values in the output `wind_energy_points.shp` vector.

As such, we have decided to move forward with removing the rasterized outputs:
- `carbon_emissions_tons.tif`
- `density_W_per_m2.tif`
- `harvested_energy_MWhr_per_yr.tif`
- `levelized_cost_price_per_kWh.tif`
- `npv.tif`

The model will need to be updated so that the valuation component also writes values to `wind_energy_points.shp`.

## Status

## Consequences

Once released, the model will no longer provide the rasterized outputs that it previously provided. Instead, values for each point will appear in `wind_energy_points.shp`. This vector will also contain valuation data if the model's valuation component is run.

## References

GitHub:
  * [Pull Request](https://github.com/natcap/invest/pull/1898)
  * [Discussion: Raster result values returned outside of wind data](https://github.com/natcap/invest/issues/1698)
  * [User's Guide PR](https://github.com/natcap/invest.users-guide/pull/178)
