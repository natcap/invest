#!python

import subprocess
import os
import tempfile
import logging
import platform
import pprint
import argparse

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('invest-autovalidate.py')

# Mapping of model keys to datastacks to run through the model's UI.
# Paths are assumed to be relative to the data root.
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


def validate_datastack(modelname, binary, workspace, datastack):
    """Run an InVEST model, checking the error code of the process."""
    # Using a list here allows subprocess to handle escaping of paths.
    command = [binary, '--workspace', workspace, '--datastack', datastack,
               '--dry-run', '--debug', '--headless', '--overwrite', modelname]

    # Subprocess on linux/mac seems to prefer a list of args, but path escaping
    # (by passing the command as a list) seems to work better on Windows.
    if platform.system() != 'Windows':
        command = ' '.join(command)

    try:
        _ = subprocess.check_output(command, shell=True)
    except subprocess.CalledProcessError as error_obj:
        error_code = error_obj.returncode
        error_output = error_obj.output
    else:
        error_code = 0
        error_output = ''
    return (modelname, error_code, error_output)


def main(sampledatadir):
    pairs = []
    for name, datastacks in DATASTACKS.iteritems():
        # if not name.startswith(args.prefix):
        #     continue
        for datastack_index, datastack in enumerate(datastacks):
            pairs.append((name, datastack, datastack_index))

    warnings_list = []
    for modelname, datastack, datastack_index in pairs:
        # datastack = os.path.join(args.cwd, datastack)
        datastack = os.path.join(sampledatadir, datastack)

        workspace = tempfile.mkdtemp()
        _, _, output = validate_datastack(modelname, 'invest', workspace, datastack)
        if output:
            warnings_list.append(output)

    if warnings_list:
        raise ValueError('Validation failed with %s' % pprint.pformat(warnings_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sampledatadir', type=str)
    args = parser.parse_args()
    main(args.sampledatadir)
