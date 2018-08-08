"""Tracer script to help with development."""
import logging
import sys

from osgeo import gdal
import pygeoprocessing
import natcap.invest.urban_flood_risk_mitigation

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
        'workspace_dir': 'urban_invest_workspace',
        'results_suffix': '',
        'aoi_watersheds_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\Draft_Watersheds_SFEI\Draft_Watersheds_SFEI.shp",
        'rainfall_depth': 257.0,
        'lulc_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\LULC_SFBA.tif",
        'soils_hydrological_group_raster_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\SoilHydroGroup_SFBA_reproj_0123.tif",
        'curve_number_table_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\Biophysical_water_SF.csv",
        'flood_prone_areas_vector_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\Flood_areas.shp",
        'built_infrastructure_vector_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\Built_infra_censusblock.shp",
        'infrastructure_damage_loss_table_path': r"D:\Dropbox\Urban InVEST\Stormwater data SF\Damage.csv",
        }
    natcap.invest.urban_flood_risk_mitigation.execute(args)


if __name__ == '__main__':
    main()
