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
    'pollination': [os.path.join('Pollination', 'willamette.invs.json')],
}


def sh(command, capture=True):
    """Execute something on the shell and return the stdout."""
    p = subprocess.Popen(command, shell=True,
                         stderr=subprocess.STDOUT,
                         stdout=subprocess.PIPE)
    p_stdout = p.communicate()[0]
    if capture:
        return p_stdout


def run_model(modelname, cwd, binary, workspace, scenario):
    """Run an InVEST model, checking the error code of the process."""
    workspace = os.path.join(workspace, 'autorun_%s' % modelname)
    command = ('{binary} {model} --quickrun '
               '--workspace="{workspace}" '
               '--scenario="{scenario}" ').format(binary=binary,
                                                 model=modelname,
                                                 workspace=workspace,
                                                 scenario=scenario),
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
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
              'be ./data/invest-data/Base_Data or a directory at the same '
              'level. If executing from a built InVEST binary, this will be '
              'the current directory (".").  Default value: "."'
              ))
    parser.add_argument(
        '--workspace',
        default=tempfile.mkdtemp(),
        help=('Where the output workspaces for all model runs should be '
              'stored. Default value is a new temporary directory.'))
    args = parser.parse_args(user_args)
    LOGGER.debug(args)
    LOGGER.info('Writing all model workspaces to %s', args.workspace)
    LOGGER.info('Running on %s CPUs', args.max_cpus)

    pairs = []
    for name, scenarios in SCENARIOS.iteritems():
        for scenario in scenarios:
            pairs.append((name, scenario))

    pool = multiprocessing.Pool(processes=args.max_cpus)  # cpu_count()-1
    processes = []
    for _modelname, _scenario in pairs:
        process = pool.apply_async(run_model, (_modelname,
                                               args.cwd,
                                               args.binary,
                                               args.workspace,
                                               _scenario))
        processes.append((process, _scenario))

    # get() blocks until the result is ready.
    model_results = {}
    for _process, _scenario in processes:
        result = _process.get()
        model_results[(result[0], _scenario)] = result[1:]

    max_width = max([len(key[0]) for key in model_results.keys()])
    failures = 0

    # print all statuses, sorted by the modelname.
    print '%s %s %s' % (string.ljust('MODELNAME', max_width+1),
                        string.ljust('EXIT CODE', 10),  # len('EXIT CODE')+1
                        'SCENARIO')
    for (modelname, scenario), exitcode in sorted(
                model_results.iteritems(), key=lambda x: x[0]):
        print "%s %s %s" % (string.ljust(modelname, max_width+1),
                            string.ljust(str(exitcode[0]), 10),
                            scenario)
        if exitcode[0] > 0:
            failures += 1

    if failures > 0:
        print '\n********FAILURES********'
        print '%s %s %s' % (string.ljust('MODELNAME', max_width+1),
                            string.ljust('EXIT CODE', 10),
                            'SCENARIO')
        for (modelname, scenario), exitcode in sorted(
                [(k, v) for (k, v) in model_results.iteritems() if v != 0],
                key=lambda x: x[0]):
            print "%s %s %s" % (string.ljust(modelname, max_width+1),
                                string.ljust(str(exitcode[0]), 10),
                                scenario)

if __name__ == '__main__':
    main()
