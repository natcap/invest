# encoding=UTF-8
"""convert-requirements-to-conda-yml.py"""

import sys
import argparse
import requests
import pkg_resources

FEEDSTOCK_URL = 'https://github.com/conda-forge/{package}-feedstock'
YML_TEMPLATE = """name: invest-env
channels:
- conda-forge
- default
dependencies:
{conda_dependencies}
{pip_dependencies}
"""

SCM_MAP = {
    'hg': 'mercurial',
    'git': 'git',
}

def ui(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description=(
        'Convert a ser of pip requirements.txt files into an environment '
        'file for use by `conda create`.'
    ), prog=__file__)

    parser.add_argument('req', nargs='+',
                        help='A requirements.txt file to analyze')

    args = parser.parse_args(args)
    main(*args.req)


def main(*requirements_files):
    pip_requirements = set([])
    conda_requirements = set(['python=2.7'])
    for requirement_file in requirements_files:
        for line in open(requirement_file):
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue

            if line.startswith(tuple(SCM_MAP.keys())):
                pip_requirements.add(line)
                conda_requirements.add(SCM_MAP[line.split('+')[0]])
                continue

            requirement = pkg_resources.Requirement.parse(line)
            conda_forge_url = FEEDSTOCK_URL.format(
                package=requirement.project_name.lower())
            if (requests.get(conda_forge_url).status_code == 200 and not
                    line.endswith('# pip-only')):
                conda_requirements.add(line)
            else:
                pip_requirements.add(line)

    conda_deps_string = '\n'.join(['- %s' % dep for dep in
                                   sorted(conda_requirements,
                                          key=lambda x: x.lower())])
    pip_deps_string = '- pip:\n' + '\n'.join(['  - %s' % dep for dep in
                                              sorted(pip_requirements,
                                                     key=lambda x: x.lower())])
    print YML_TEMPLATE.format(
        conda_dependencies=conda_deps_string,
        pip_dependencies=pip_deps_string)


if __name__ == '__main__':
    ui()

# TODO: document
# TODO: resolve dependencies by calling conda?
