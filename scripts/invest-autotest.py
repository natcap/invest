#!python

import subprocess
import string
import multiprocessing
import argparse
import sys
import os
import tempfile
import logging
import platform

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('invest-autotest.py')

# Mapping of model keys to datastacks to run through the model's UI.
# Paths are assumed to be relative to the data root.
DATASTACKS = {
    'carbon': ['Carbon/carbon_willamette.invs.json'],
    'coastal_blue_carbon': ['CoastalBlueCarbon/cbc_galveston_bay.invs.json'],
    'coastal_blue_carbon_preprocessor': ['CoastalBlueCarbon/cbc_pre_galveston_bay.invs.json'],
    'coastal_vulnerability': ['CoastalVulnerability/coastal_vuln_grandbahama.invs.json'],
    'crop_production_percentile': [
        'CropProduction/crop_production_percentile_demo.invs.json'],
    'crop_production_regression': [
        'CropProduction/crop_production_regression_demo.invs.json'],
    'delineateit': ['DelineateIt/delineateit_gura.invs.json'],
    'forest_carbon_edge_effect': ['forest_carbon_edge_effect/forest_carbon_amazonia.invs.json'],
    'globio': ['globio/globio_demo.invs.json'],
    'habitat_quality': ['HabitatQuality/habitat_quality_willamette.invs.json'],
    'hra': ['HabitatRiskAssess/hra_wcvi.invs.json'],
    'annual_water_yield': ['Annual_Water_Yield/annual_water_yield_gura.invs.json'],
    'ndr': ['NDR/ndr_gura.invs.json'],
    'pollination': ['pollination/pollination_willamette.invs.json'],
    'recreation': ['recreation/recreation_andros.invs.json'],
    'routedem': ['RouteDEM/routedem_gura.invs.json'],
    'scenario_generator_proximity': ['scenario_proximity/scenario_proximity_amazonia.invs.json'],
    'scenic_quality': ['ScenicQuality/wind_turbines_wcvi.invs.json'],
    'sdr': ['SDR/sdr_gura.invs.json'],
    'seasonal_water_yield': ['Seasonal_Water_Yield/seasonal_water_yield_gura.invs.json'],
    'stormwater': ['UrbanStormwater/stormwater_datastack.invest.json'],
    'urban_cooling_model': ['UrbanCoolingModel/urban_cooling_model_datastack.invest.json'],
    'urban_flood_risk_mitigation': ['UrbanFloodMitigation/urban_flood_risk_mitigation.invs.json'],
    'wind_energy': ['WindEnergy/wind_energy_new_england.invs.json'],
    'wave_energy': [
        'WaveEnergy/wave_energy_aquabuoy_wcvi.invs.json',
        'WaveEnergy/wave_energy_owc_wcvi.invs.json',
        'WaveEnergy/wave_energy_pelamis_wcvi.invs.json',
        'WaveEnergy/wave_energy_wavedragon_wcvi.invs.json',
    ],
}


def sh(command, capture=True):
    """Execute something on the shell and return the stdout."""
    p = subprocess.Popen(command, shell=True,
                         stderr=subprocess.STDOUT,
                         stdout=subprocess.PIPE)
    p_stdout = p.communicate()[0]
    if capture:
        return p_stdout


def run_model(modelname, binary, workspace, datastack, headless=False):
    """Run an InVEST model, checking the error code of the process."""
    # Using a list here allows subprocess to handle escaping of paths.
    if headless:
        command = [binary, 'run', '--workspace', workspace,
                   '--datastack', datastack, '--headless', modelname]
    else:
        command = [binary, 'quickrun', '--workspace', workspace,
                   modelname, datastack]

    # Subprocess on linux/mac seems to prefer a list of args, but path escaping
    # (by passing the command as a list) seems to work better on Windows.
    if platform.system() != 'Windows':
        command = ' '.join(command)

    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as error_obj:
        error_code = error_obj.returncode
    else:
        error_code = 0
    return (modelname, error_code)


def main(user_args=None):
    """Execute all InVEST models using a pool of available processes."""
    if not user_args:
        user_args = sys.argv[1:]

    # Don't use a default CPU count of less than 1.
    default_cpu_count = max(multiprocessing.cpu_count()-1, 1)
    parser = argparse.ArgumentParser(
        prog='invest-autotest.py',
        description=(
            'Run through each InVEST model to verify it completes. '
            'This script is for testing purposes only.'))
    parser.add_argument(
        '--max-cpus',
        default=default_cpu_count,
        type=int,
        help=('The number of CPUs to use. '
              'Defaults to %s.') % default_cpu_count)
    parser.add_argument(
        '--binary',
        default='invest',
        help=('The path to the InVEST binary to call.  Defaults to whatever '
              'is on the PATH.'))
    parser.add_argument(
        '--cwd',
        default='sample_data',
        help=('The CWD from which to execute the models. '
              'If executing from a checked-out InVEST repo, this will probably '
              'be ./data/invest-sample-data/ or a directory at the same '
              'level. If executing from a built InVEST binary, this will be '
              'the sample_data directory.  Default value: "sample_data"'
             ))
    parser.add_argument(
        '--workspace',
        default=tempfile.mkdtemp(),
        help=('Where the output workspaces for all model runs should be '
              'stored. Default value is a new temporary directory.'))
    parser.add_argument(
        '--prefix',
        default='',
        help=('If provided, only those models that start with this value will '
              'be run.  If not provided, all models will be run.'))
    args = parser.parse_args(user_args)
    LOGGER.debug(args)
    LOGGER.info('Writing all model workspaces to %s', args.workspace)
    LOGGER.info('Running on %s CPUs', args.max_cpus)

    pairs = []
    for name, datastacks in DATASTACKS.items():
        if not name.startswith(args.prefix):
            continue

        for datastack_index, datastack in enumerate(datastacks):
            pairs.append((name, datastack, datastack_index))

    pool = multiprocessing.Pool(processes=args.max_cpus)  # cpu_count()-1
    processes = []
    for modelname, datastack, datastack_index in pairs:
        datastack = os.path.join(args.cwd, datastack)

        for headless in (True, False):
            headless_string = ''
            if headless:
                headless_string = 'headless'
            else:
                headless_string = 'gui'
            workspace = os.path.join(os.path.abspath(args.workspace),
                                     'autorun_%s_%s_%s' % (modelname,
                                                           headless_string,
                                                           datastack_index))
            process = pool.apply_async(run_model, (modelname,
                                                   args.binary,
                                                   workspace,
                                                   datastack,
                                                   headless))
            processes.append((process, datastack, headless, workspace))

    # get() blocks until the result is ready.
    model_results = {}
    for _process, _datastack, _headless, _workspace in processes:
        result = _process.get()
        model_results[(result[0], _datastack, _headless, _workspace)] = result[1:]

    # add 10 for ' (headless)'
    max_width = max([len(key[0])+11 for key in model_results.keys()])
    failures = 0

    datastack_width = max([len(key[1]) for key in model_results.keys()])

    # record all statuses, sorted by the modelname, being sure to start on a
    # new line.
    status_messages = ''
    status_messages += '\n%s %s %s\n' % (
        'MODELNAME'.ljust(max_width+1),
        'EXIT CODE'.ljust(10),  # len('EXIT CODE')+1
        'DATASTACK')
    for (modelname, datastack, headless, _), exitcode in sorted(
            model_results.items(), key=lambda x: x[0]):
        if headless:
            modelname += ' (headless)'
        status_messages += "%s %s %s\n" % (
            modelname.ljust(max_width+1),
            str(exitcode[0]).ljust(10),
            datastack)
        if exitcode[0] > 0:
            failures += 1

    if failures > 0:
        status_messages += '\n********FAILURES********\n'
        status_messages += '%s %s %s %s\n' % (
            'MODELNAME'.ljust(max_width+1),
            'EXIT CODE'.ljust(10),
            'DATASTACK'.ljust(datastack_width),
            'WORKSPACE'
        )
        for (modelname, datastack, headless, workspace), exitcode in sorted(
                [(k, v) for (k, v) in model_results.items()
                 if v[0] != 0],
                key=lambda x: x[0]):
            if headless:
                modelname += ' (headless)'
            status_messages += "%s %s %s %s\n" % (
                modelname.ljust(max_width+1),
                str(exitcode[0]).ljust(10),
                datastack.ljust(datastack_width),
                workspace
            )

    print(status_messages)
    with open(os.path.join(args.workspace, 'model_results.txt'), 'w') as log:
        log.write(status_messages)


if __name__ == '__main__':
    main()
