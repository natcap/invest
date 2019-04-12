"""Tracer script to help with development."""
import logging
import sys

import natcap.invest.urban_heat_island_mitigation

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout,
    )
LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    args = {
        'workspace_dir': 'urban_heat_island_invest_workspace',
        'results_suffix': 'suffix2',
        't_obs_raster_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Tair_Sept.tif",
        'lulc_raster_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\LULC_SFBA.tif",
        'ref_eto_raster_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\ETo_SFBA.tif",
        'aoi_vector_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Draft_Watersheds_SFEI\Draft_Watersheds_SFEI.shp",
        'biophysical_table_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Biophysical table_UHI.csv",
        'urban_park_cooling_distance': 1000.0,
        'uhi_max': 3.5,
        't_air_average_radius': "1001.0",
        'building_vector_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Buildings.shp",
        'energy_consumption_table_path': r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Energy.csv",
        'avg_rel_humidity': '100.0',
        'cc_weight_shade': '0.6',
        'cc_weight_albedo': '0.2',
        'cc_weight_eti': '0.2',
        }
    natcap.invest.urban_heat_island_mitigation.execute(args)


if __name__ == '__main__':
    main()
