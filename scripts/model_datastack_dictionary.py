# Mapping of model keys to datastacks.
# invest-autotest.py and invest-autovalidate.py use this dict.
# Paths are assumed to be relative to the sample data root.
DATASTACKS = {
    'carbon': ['carbon_willamette.invs.json'],
    'coastal_blue_carbon': ['cbc_galveston_bay.invs.json'],
    'coastal_blue_carbon_preprocessor': ['cbc_pre_galveston_bay.invs.json'],
    'coastal_vulnerability': ['coastal_vuln_wcvi.invs.json'],
    'crop_production_percentile': [
        'crop_production_percentile_demo.invs.json'],
    'crop_production_regression': [
        'crop_production_regression_demo.invs.json'],
    'delineateit': ['delineateit_willamette.invs.json'],
    'finfish_aquaculture': ['atlantic_salmon_british_columbia.invs.json'],
    'fisheries': [
        'blue_crab_galveston_bay.invs.json',
        'dungeness_crab_hood_canal.invs.json',
        'spiny_lobster_belize.invs.json',
        'white_shrimp_galveston_bay.invs.json',
    ],
    'fisheries_hst': ['fisheries_hst_demo.invs.json'],
    'forest_carbon_edge_effect': ['forest_carbon_amazonia.invs.json'],
    'globio': ['globio_demo.invs.json'],
    'habitat_quality': ['habitat_quality_willamette.invs.json'],
    'hra': ['hra_wcvi.invs.json'],
    'hydropower_water_yield': ['annual_water_yield_willamette.invs.json'],
    'ndr': ['ndr_n_p_willamette.invs.json'],
    'pollination': ['pollination_willamette.invs.json'],
    'recreation': ['recreation_andros.invs.json'],
    'routedem': ['routedem_willamette.invs.json'],
    'scenario_generator_proximity': ['scenario_proximity_amazonia.invs.json'],
    'scenic_quality': ['wind_turbines_wcvi.invs.json'],
    'sdr': ['sdr_willamette.invs.json'],
    'seasonal_water_yield': ['swy_willamette.invs.json'],
    'wind_energy': ['wind_energy_new_england.invs.json'],
    'wave_energy': [
        'wave_energy_aquabuoy_wcvi.invs.json',
        'wave_energy_owc_wcvi.invs.json',
        'wave_energy_pelamis_wcvi.invs.json',
        'wave_energy_wavedragon_wcvi.invs.json',
    ],
}