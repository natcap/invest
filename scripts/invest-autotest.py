#!python

import subprocess
import string
import multiprocessing
import argparse
import sys
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('invest-autotest.py')

# Mapping of model keys to scenarios to run through the model's UI.
# Paths are assumed to be relative to the data root.
SCENARIOS = {
    'carbon': [os.path.join('carbon', 'carbon_willamette.invs.json')],
    'coastal_blue_carbon': [
        os.path.join('CoastalBlueCarbon', 'cbc_galveston_bay.invs.json')],
    'coastal_blue_carbon_preprocessor': [
        os.path.join('CoastalBlueCarbon', 'cbc_pre_galveston_bay.invs.json')],
    'coastal_vulnerability': [
        os.path.join('CoastalProtection', 'coastal_vuln_wcvi.invs.json')],
    'crop_production_percentile': [
        os.path.join('CropProduction', 'sample_user_data',
                     'crop_production_percentile_demo.invs.json')],
    'crop_production_regression': [
        os.path.join('CropProduction', 'sample_user_data',
                     'crop_production_regression_demo.invs.json')],
    'delineateit': [
        os.path.join('Base_Data', 'Freshwater',
                     'delineateit_willamette.invs.json')],
    'finfish_aquaculture': [
        os.path.join('Aquaculture',
                     'atlantic_salmon_british_columbia.invs.json')],
    'fisheries': [
        os.path.join('Fisheries', 'blue_crab_galveston_bay.invs.json'),
        os.path.join('Fisheries', 'dungeness_crab_hood_canal.invs.json'),
        os.path.join('Fisheries', 'spiny_lobster_belize.invs.json'),
        os.path.join('Fisheries', 'white_shrimp_galveston_bay.invs.json'),
    ],
    'fisheries_hst': [
        os.path.join('Fisheries', 'fisheries_hst_demo.invs.json')],
    'forest_carbon_edge_effect': [
        os.path.join('forest_carbon_edge_effect',
                     'forest_carbon_amazonia.invs.json')],
    'globio': [os.path.join('globio', 'globio_demo.invs.json')],
    'habitat_quality': [
        os.path.join('HabitatQuality', 'habitat_quality_willamette.invs.json')],
    'hra': [os.path.join('HabitatRiskAssess', 'hra_wcvi.invs.json')],
    'habitat_risk_assessment_preprocessor': [
        os.path.join('HabitatRiskAssess', 'hra_pre_wcvi.invs.json')],
    'hydropower_water_yield': [
        os.path.join('Hydropower', 'annual_water_yield_willamette.invs.json')],
    'ndr': [
        os.path.join('Base_Data', 'Freshwater',
                     'ndr_n_p_willamette.invs.json')],
    'overlap_analysis': [
        os.path.join('OverlapAnalysis', 'overlap_wcvi.invs.json')],
    'overlap_analysis_mz': [
        os.path.join('OverlapAnalysis', 'overlap_mz_wcvi.invs.json')],
    'pollination': [
        os.path.join('pollination', 'pollination_willamette.invs.json')],
    'recreation': [
        os.path.join('recreation', 'recreation_andros.invs.json')],
    'routedem': [
        os.path.join('Base_Data', 'Freshwater',
                     'routedem_willamette.invs.json')],
    'scenario_generator_proximity': [
        os.path.join('scenario_proximity',
                     'scenario_proximity_amazonia.invs.json')],
    'scenario_generator': [
        os.path.join('ScenarioGenerator', 'scenario_generator_demo.invs.json')],
    'scenic_quality': [
        os.path.join('ScenicQuality', 'wind_turbines_wcvi.invs.json')],
    'sdr': [
        os.path.join('Base_Data', 'Freshwater', 'sdr_willamette.invs.json')],
    'seasonal_water_yield': [
        os.path.join('seasonal_water_yield', 'swy_willamette.invs.json')],
    'wind_energy': [os.path.join('WindEnergy', 'new_england.invs.json')],
    'wave_energy': [
        os.path.join('WaveEnergy', 'wave_energy_aquabuoy_wcvi.invs.json'),
        os.path.join('WaveEnergy', 'wave_energy_owc_wcvi.invs.json'),
        os.path.join('WaveEnergy', 'wave_energy_pelamis_wcvi.invs.json'),
        os.path.join('WaveEnergy', 'wave_energy_wavedragon_wcvi.invs.json'),
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


def run_model(modelname, binary, workspace, scenario, headless=False):
    """Run an InVEST model, checking the error code of the process."""
    # Using a list here allows subprocess to handle escaping of paths.
    command = [binary, modelname, '--quickrun', '--workspace=%s' % workspace,
               '-y', '--scenario=%s' % scenario]
    if headless:
        command.append('--headless')

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
        default='.',
        help=('The CWD from which to execute the models. '
              'If executing from a checked-out InVEST repo, this will probably '
              'be ./data/invest-data/ or a directory at the same '
              'level. If executing from a built InVEST binary, this will be '
              'the current directory (".").  Default value: "."'
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
    for name, scenarios in SCENARIOS.iteritems():
        if not name.startswith(args.prefix):
            continue

        for scenario_index, scenario in enumerate(scenarios):
            pairs.append((name, scenario, scenario_index))

    pool = multiprocessing.Pool(processes=args.max_cpus)  # cpu_count()-1
    processes = []
    for modelname, scenario, scenario_index in pairs:
        scenario = os.path.join(args.cwd, scenario)

        for headless in (True, False):
            headless_string = ''
            if headless:
                headless_string = 'headless'
            else:
                headless_string = 'gui'
            workspace = os.path.join(os.path.abspath(args.workspace),
                                     'autorun_%s_%s_%s' % (modelname,
                                                           headless_string,
                                                           scenario_index))
            process = pool.apply_async(run_model, (modelname,
                                                   args.binary,
                                                   workspace,
                                                   scenario,
                                                   headless))
            processes.append((process, scenario, headless, workspace))

    # get() blocks until the result is ready.
    model_results = {}
    for _process, _scenario, _headless, _workspace in processes:
        result = _process.get()
        model_results[(result[0], _scenario, _headless, _workspace)] = result[1:]

    # add 10 for ' (headless)'
    max_width = max([len(key[0])+11 for key in model_results.keys()])
    failures = 0

    scenario_width = max([len(key[1]) for key in model_results.keys()])

    # record all statuses, sorted by the modelname, being sure to start on a
    # new line.
    status_messages = ''
    status_messages += '\n%s %s %s\n' % (
        string.ljust('MODELNAME', max_width+1),
        string.ljust('EXIT CODE', 10),  # len('EXIT CODE')+1
        'SCENARIO')
    for (modelname, scenario, headless, _), exitcode in sorted(
            model_results.iteritems(), key=lambda x: x[0]):
        if headless:
            modelname += ' (headless)'
        status_messages += "%s %s %s\n" % (
            string.ljust(modelname, max_width+1),
            string.ljust(str(exitcode[0]), 10),
            scenario)
        if exitcode[0] > 0:
            failures += 1

    if failures > 0:
        status_messages += '\n********FAILURES********\n'
        status_messages += '%s %s %s %s\n' % (
            string.ljust('MODELNAME', max_width+1),
            string.ljust('EXIT CODE', 10),
            string.ljust('SCENARIO', scenario_width),
            'WORKSPACE'
        )
        for (modelname, scenario, headless, workspace), exitcode in sorted(
                [(k, v) for (k, v) in model_results.iteritems()
                 if v[0] != 0],
                key=lambda x: x[0]):
            if headless:
                modelname += ' (headless)'
            status_messages += "%s %s %s %s\n" % (
                string.ljust(modelname, max_width+1),
                string.ljust(str(exitcode[0]), 10),
                string.ljust(scenario, scenario_width),
                workspace
            )

    print status_messages
    with open(os.path.join(args.workspace, 'model_results.txt'), 'w') as log:
        log.write(status_messages)


if __name__ == '__main__':
    main()
