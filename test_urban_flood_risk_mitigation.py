"""Tracer script to help with development."""
import natcap.invest.urban_flood_risk_mitigation


def main():
    """Entry point."""
    args = {
        'workspace_dir': '',
        'results_suffix': '',
        'dem_path': '',
        'aoi_watersheds_path': '',
        'rainfall_depth': '',
        'lulc_path': '',
        'soils_hydrological_group_raster_path': '',
        'curve_number_table_path': '',
        'flood_prone_areas_vector_path': '',
        'built_infrastructure_vector_path': '',
        'infrastructure_damage_loss_table_path': '',
        }
    natcap.invest.urban_flood_risk_mitigation.execute(args)


if __name__ == '__main__':
    main()
