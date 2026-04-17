# encoding=UTF-8
"""convert-requirements-to-conda-yml.py"""

import argparse
import platform
import sys

YML_TEMPLATE = """channels:
- conda-forge
- nodefaults
dependencies:
{conda_dependencies}
{pip_dependencies}
"""


def build_environment_from_requirements(cli_args):
    """Build a conda environment.yml from requirements.txt files.

    This script assumes a couple of rules about what should be installed by
    conda and what should be installed by pip:

        1. If the requirement is installed from git or hg, it should be
           installed by pip.
        2. If the requirement is followed by the commend '# pip-only', it
           should be installed by pip.
        3. If the requirement is available on conda-forge, it should be
           installed by conda.
        4. Otherwise, it should be installed by pip.


    Arguments:
        cli_args (list): A list of command-line arguments.

    Returns:
        ``None``
    """
    parser = argparse.ArgumentParser(description=(
        'Convert a set of pip requirements.txt files into an environment '
        'file for use by `conda create`.'
    ), prog=__file__)

    parser.add_argument('req', nargs='+',
                        help='A requirements.txt file to analyze')

    args = parser.parse_args(cli_args)
    requirements_files = args.req

    pip_requirements = set()
    conda_requirements = set()
    for requirement_file in requirements_files:
        with open(requirement_file) as file:
            for line in file:
                line = line.strip()

                # Blank line or comment
                if len(line) == 0 or line.startswith('#'):
                    continue

                if line.endswith('# pip-only'):
                    # Conda prefers that we explicitly include pip as a
                    # requirement if we're using pip.
                    conda_requirements.add('pip')

                    pip_requirements.add(line)

                    # If an scm needs to be installed for pip to clone to a
                    # revision, add it to the conda package list.
                    #
                    # Bazaar (bzr, which pip supports) is not listed;
                    # deprecated as of 2016 and not available on conda-forge.
                    install_compiler = False
                    for prefix, scm_conda_pkg in [("git+", "git"),
                                                  ("hg+", "mercurial"),
                                                  ("svn+", "subversion")]:
                        if line.startswith(prefix):
                            conda_requirements.add(scm_conda_pkg)
                            install_compiler = True
                            break  # The line can only match 1 prefix

                    # It's less common (like for pygeoprocessing) to have linux
                    # wheels.  Install a compiler if we're on linux, to be
                    # safe.
                    if platform.system() == 'Linux' or install_compiler:
                        # Always make the compiler available
                        # cxx-compiler works on all OSes.
                        # NOTE: do not use this as a dependency in a
                        # conda-forge package recipe!
                        # https://anaconda.org/conda-forge/cxx-compiler
                        conda_requirements.add('cxx-compiler')

                else:
                    conda_requirements.add(line)

    conda_deps_string = '\n'.join(
        [f'- {dep}' for dep in sorted(conda_requirements, key=str.casefold)])
    if pip_requirements:
        pip_deps_string = '- pip:\n' + '\n'.join(
            ['  - %s' % dep for dep in sorted(pip_requirements, key=str.casefold)])
    else:
        pip_deps_string = ''
    print(YML_TEMPLATE.format(
        conda_dependencies=conda_deps_string,
        pip_dependencies=pip_deps_string))


if __name__ == '__main__':
    build_environment_from_requirements(sys.argv[1:])

# TODO: resolve dependencies by calling conda?
