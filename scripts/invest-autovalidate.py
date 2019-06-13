#!python

import subprocess
import os
import tempfile
import logging
import platform
import pprint
import argparse

from model_datastack_dictionary import DATASTACKS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('invest-autovalidate.py')


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
