import argparse
import distutils
import distutils.ccompiler
import getpass
import glob
import imp
import importlib
import inspect
import json
import os
import pkgutil
import platform
import shutil
import site
import socket
import subprocess
import sys
import tarfile
import textwrap
import time
import warnings
import zipfile
from types import DictType

import pkg_resources
import paver.svn
import paver.path
import paver.virtual
from paver.easy import task, cmdopts, consume_args, might_call,\
    dry, sh, call_task, BuildFailure, no_help, Bunch
import virtualenv
import yaml


# Pip 6.0 introduced the --no-use-wheel option.  Pip 7.0.0 deprecated
# --no-use-wheel in favor of --no-binary.  Stable versions of Fedora
# currently use pip 6.x
# virtualenv 13.0.0 upgraded pip to 7.0.0, but the older flags still work for
# now.
try:
    pkg_resources.require('pip>=7.0.0')
    pkg_resources.require('virtualenv>=13.0.0')
    NO_WHEEL_SUBPROCESS = "'--no-binary', ':all:'"
    NO_WHEEL_SH = '--no-binary :all:'
except pkg_resources.VersionConflict:
    NO_WHEEL_SUBPROCESS = "'--no-use-wheel'"
    NO_WHEEL_SH = '--no-use-wheel'


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.

    Taken from http://stackoverflow.com/a/22254892/299084
    """
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


TERM_IS_COLOR = supports_color()


def _colorize(color_pattern, msg):
    """
    Apply the color pattern (likely an ANSI color escape code sequence)
    to the message if the current terminal supports color.  If the terminal
    does not support color, return the messge.
    """
    if TERM_IS_COLOR:
        return color_pattern % msg
    return msg

def green(msg):
    """
    Return a string that is formatted as ANSI green.
    If the terminal does not support color, the input message is returned.
    """
    return _colorize('\033[92m%s\033[0m', msg)

def yellow(msg):
    """
    Return a string that is formatted as ANSI yellow.
    If the terminal does not support color, the input message is returned.
    """
    return _colorize('\033[93m%s\033[0m', msg)

def red(msg):
    """
    Return a string that is formatted as ANSI red.
    If the terminal does not support color, the input message is returned.
    """
    return _colorize('\033[91m%s\033[0m', msg)

def bold(message):
    """
    Return a string formatted as ANSI bold.
    If the terminal does not support color, the input message is returned.
    """
    return _colorize("\033[1m%s\033[0m", message)


ERROR = red('ERROR:')
WARNING = yellow('WARNING:')
OK = green('OK')


def _import_namespace_pkg(modname, print_msg=True):
    """
    Import a package within the natcap namespace and print helpful
    debug messages as packages are discovered.

    Parameters:
        modname (string): The natcap subpackage name.
        print_msg=True (bool): Whether to print messages about the import
            state.

    Returns:
        Either 'egg' or 'dir' if the package is importable.

    Raises:
        ImportError: If the package cannot be imported.
    """
    module = importlib.import_module('natcap.%s' % modname)
    try:
        version = module.__version__
    except AttributeError:
        packagename = 'natcap.%s' % modname
        version = pkg_resources.require(packagename)[0].version

    is_egg = reduce(
        lambda x, y: x or y,
        [p.endswith('.egg') for p in module.__file__.split(os.sep)])

    if len(module.__path__) > 1:
        module_path = module.__path__
    else:
        module_path = module.__path__[0]

    if not is_egg:
        return_type = 'dir'
        message = '{warn} natcap.{mod}=={ver} ({dir}) not an egg.'.format(
            warn=WARNING, mod=modname, ver=version, dir=module_path)
    else:
        return_type = 'egg'
        message = "natcap.{mod}=={ver} installed as egg ({dir})".format(
            mod=modname, ver=version, dir=module_path)

    if print_msg:
        print message

    return (module, return_type)


def is_exe(fpath):
    """
    Check whether a file is executable and that it exists.

    Parameters:
        fpath (string): The filepath to check.

    Returns:
        A boolean.
    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def find_executable(program):
    """
    Locate the provided program.

    Parameters:
        program (string): Either the absolute path to an executable or an exe
            name (e.g. python, git).  On Windows systems, if the program name
            does not already include '.exe', it will be appended to the
            program name provided.

    Returns:
        The absolute path to the executable if it can be found.  Raises
        EnvironmentError if not.

    Raises:
        EnvironmentError: When the program cannot be found.

    """
    if platform.system() == 'Windows' and not program.endswith('.exe'):
        program += '.exe'

    fpath, fname = os.path.split(program)
    if fpath:  # fpath is not '' when an absolute path is given.
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    raise EnvironmentError(
        "Executable not found: {program}".format(
            program=program))


def user_os_installer():
    """
    Determine the operating system installer.

    On Linux, this will be either "deb" or "rpm" depending on the presence
    of /usr/bin/rpm.

    Returns:
        One of "rpm", "deb", "dmg", "nsis".  Returns 'UNKNOWN' if the installer
        type could not be determined.

    """
    if platform.system() == 'Linux':
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            rpm_path = os.path.join(path, 'rpm')
            if is_exe(rpm_path):
                # https://ask.fedoraproject.org/en/question/49738/how-to-check-if-system-is-rpm-or-debian-based/?answer=49850#post-id-49850
                # -q -f /path/to/rpm checks to see if RPM owns RPM.
                # If it's not owned by RPM, we can assume it's owned by apt/dpkg.
                exit_code = subprocess.call([rpm_path, '-q', '-f', rpm_path])
                if exit_code == 0:
                    return 'rpm'
                else:
                    break
        return 'deb'

    if platform.system() == 'Darwin':
        return 'dmg'

    if platform.system() == 'Windows':
        return 'nsis'

    return 'UNKNOWN'

# the options object is used for global paver configuration.  It contains
# default values for all tasks, which makes our handling of parameters much
# easier.
_ENVNAME = 'release_env'
_PYTHON = sys.executable
paver.easy.options(
    dry_run=False,
    build=Bunch(
        force_dev=False,
        skip_data=False,
        skip_installer=False,
        skip_bin=False,
        python=_PYTHON,
        envname=_ENVNAME,
        skip_python=False
    ),
    build_installer=Bunch(
        force_dev=False,
        insttype=user_os_installer(),
        arch=platform.machine(),
        bindir=os.path.join('dist', 'invest_dist')
    ),
    collect_release_files=Bunch(
        python=_PYTHON
    ),
    version=Bunch(
        json=False,
        save=False
    ),
    env=Bunch(
        system_site_packages=False,
        clear=False,
        envname=_ENVNAME,
        with_invest=False,
        requirements='',
        bootstrap_file='bootstrap.py',
        dev=False,
        with_pygeoprocessing=False,
        compiler=None
    ),
    build_docs=Bunch(
        force_dev=False,
        skip_api=False,
        skip_guide=False,
        python=_PYTHON
    ),
    build_data=Bunch(
        force_dev=False
    ),
    build_bin=Bunch(
        force_dev=False,
        python=_PYTHON
    ),
    jenkins_installer=Bunch(
        nodata='false',
        nobin='false',
        nodocs='false',
        noinstaller='false',
        nopush='false'
    ),
    clean=Bunch(),
    virtualenv=Bunch(),
    dev_env=Bunch(
        envname='test_env',
        noinvest=False,
        compiler=None
    ),
    check=Bunch(
        fix_namespace=False,
        allow_errors=False
    )
)


class Repository(object):
    """Abstract class representing a version-controlled repository."""
    tip = ''  # The string representing the latest revision
    statedir = ''  # Where the SCM stores its data, relative to repo root
    cmd = ''  # The command-line exe to call.

    def __init__(self, local_path, remote_url):
        """Initialize the Repository instance

        Parameters:
            local_path (string): The filepath on disk to the repo
                (relative to pavement.py)
            remote_url (string): The string URL to use to identify the repo.
                Used for clones, updates.

        Returns:
            An instance of Repository.
        """
        self.local_path = local_path
        self.remote_url = remote_url

    def get(self, rev):
        """Update to the target revision.

        Parameters:
            rev (string): The revision to update the repo to.

        Returns:
            None
        """
        if not self.ischeckedout():
            self.clone()
        else:
            print 'Repository %s exists' % self.local_path

        # If we're already updated to the correct rev, return.
        if self.at_known_rev():
            print 'Repo %s is in a known state' % self.local_path
            return

        # Try to update to the correct revision.  If we can't pull, then
        # update.
        try:
            self.update(rev)
        except BuildFailure:
            print 'Repo %s not found, falling back to fresh clone and update' % self.local_path
            # When the current repo can't be updated because it doesn't know
            # the change we want to update to
            self.clone(rev)

    def ischeckedout(self):
        """Identify whether the repository is checked out on disk."""
        return os.path.exists(os.path.join(self.local_path, self.statedir))

    def clone(self, rev=None):
        """Clone the repository from the remote URL.

        This method is to be overridden in a subclass with the SCM-specific
        commands.

        Parameters:
            rev=None (string or None): The revision to update to.  If None,
                the revision will be fetched from versions.json.

        Returns:
            None

        Raises:
            NotImplementedError: When the method is not overridden in a subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError

    def update(self, rev=None):
        """Update the repository to a revision.

        This method is to be overridden in a subclass with the SCM-specific
        commands.

        Parameters:
            rev=None (string or None): The revision to update to.  If None,
                the revision will be fetched from versions.json.

        Returns:
            None

        Raises:
            NotImplementedError: When the method is not overridden in a subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError

    def tracked_version(self):
        """Get this repository's expected version from versions.json.

        Returns:
            A string representation of the tracked version.
        """
        tracked_rev = json.load(open('versions.json'))[self.local_path]
        if type(tracked_rev) is DictType:
            user_os = platform.system()
            return tracked_rev[user_os]
        elif tracked_rev.startswith('REQUIREMENTS_TXT'):
            pkgname = tracked_rev.split(':')[1]
            version =_parse_version_requirement(pkgname)
            if version is None:
                raise ValueError((
                    'Versions.json requirement string must have the '
                    'format "REQUIREMENTS_TXT:<pkgname>".  Example: '
                    '"REQUIREMENTS_TXT:pygeoprocessing".  %s found.'
                    ) % tracked_rev)
            tracked_rev = version
        return tracked_rev

    def at_known_rev(self):
        """Identify whether the Repository is at the expected revision.

        Returns:
            A boolean.
        """
        if not self.ischeckedout():
            return False

        tracked_version = self.format_rev(self.tracked_version())
        return self.current_rev() == tracked_version

    def format_rev(self, rev):
        """Get the uniquely-identifiable commit ID of `rev`.

        This is particularly useful for SCMs that have multiple ways of
        identifying commits.

        Parameters:
            rev (string): The revision to identify.

        Returns:
            The string id of the commit.

        Raises:
            NotImplementedError: When the method is not overridden in a subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError

    def current_rev(self):
        """Fetch the current revision of the repository on disk.

        This method should be overridden in a subclass with the SCM-specific
        command(s).

        Raises:
            NotImplementedError: When the method is not overridden in a subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError


class DVCSRepository(Repository):
    """Abstract class for distributed revision control system repositories."""
    tip = ''
    statedir = ''
    cmd = ''

    def pull(self):
        """Pull new changes from the remote.

        This should be overridden in SCM-specific subclasses.

        Returns:
            None
        """
        raise NotImplementedError

    def pull_and_retry(self, shell_string, cwd=None):
        """Run a command, pulling new changes if needed.

        Parameters:
            shell_string (string): The formatted command to run.
            cwd=None(string or None): The directory from which to execute
                the command.  If None, the current working directory will be
                used.

        Returns:
            None

        Raises:
            BuildFailure: Raised when the shell command fails after pulling.
        """
        for check_again in [True, False]:
            try:
                return sh(shell_string, cwd=cwd, capture=True).rstrip()
            except BuildFailure as failure:
                if check_again is True:
                    # Pull and try to run again
                    # Assumes that self.pull is implemented in the DVCS
                    # subclass.
                    self.pull()
                else:
                    raise failure


class HgRepository(DVCSRepository):
    tip = 'tip'
    statedir = '.hg'
    cmd = 'hg'

    def clone(self, rev=None):
        if rev is None:
            rev = self.tracked_version(convert=False)
        sh('hg clone %(url)s %(dest)s -u %(rev)s' % {'url': self.remote_url,
                                                     'dest': self.local_path,
                                                     'rev': rev})

    def pull(self):
        sh('hg pull -R %(dest)s %(url)s' % {'dest': self.local_path,
                                            'url': self.remote_url})

    def update(self, rev):
        update_string = 'hg update -R {dest} -r {rev}'.format(
            dest=self.local_path, rev=rev)
        return self.pull_and_retry(update_string)

    def _format_log(self, template='', rev='.'):
        log_string = 'hg log -R {dest} -r {rev} --template="{template}"'.format(
            dest=self.local_path, rev=rev, template=template)
        return self.pull_and_retry(log_string)

    def format_rev(self, rev):
        return self._format_log('{node}', rev=rev)

    def current_rev(self, convert=True):
        return self._format_log('{node}')

    def tracked_version(self, convert=True):
        json_version = Repository.tracked_version(self)
        if not convert or not os.path.exists(self.local_path):
            return json_version
        return self._format_log(template='{node}', rev=json_version)


class SVNRepository(Repository):
    tip = 'HEAD'
    statedir = '.svn'
    cmd = 'svn'

    def at_known_rev(self):
        """Determine repo status from `svn status`

        Overridden from Repository.at_known_rev(...).  SVN info does not
        correctly report the status of the repository in the version number,
        so we must parse the output of `svn status` to see if a checkout or
        update was interrupted.

        Returns True if the repository is up-to-date.  False if not."""
        # Parse the output of SVN status.
        repo_status = sh('svn status', cwd=self.local_path, capture=True)
        for line in repo_status:
            # If the line is empty, skip it.
            if line.strip() == '':
                continue

            if line.split()[0] in ['!', 'L']:
                print 'Checkout or update incomplete!  Repo NOT at known rev.'
                return False

        print 'Status ok.'
        return Repository.at_known_rev(self)

    def _cleanup_and_retry(self, cmd, *args, **kwargs):
        """Run SVN cleanup."""
        for retry in [True, False]:
            try:
                cmd(*args, **kwargs)
            except BuildFailure as failure:
                if retry and self.ischeckedout():
                    # We should only retry if the repo is checked out.
                    print 'Cleaning up SVN repository %s' % self.local_path
                    sh('svn cleanup', cwd=self.local_path)
                    # Now we'll try the original command again!
                else:
                    # If there was a failure before the repo is checked out,
                    # then the issue is probably identified in stderr.
                    raise failure

    def clone(self, rev=None):
        if rev is None:
            rev = self.tracked_version()
        self._cleanup_and_retry(paver.svn.checkout, self.remote_url,
                                self.local_path, revision=rev)

    def update(self, rev):
        # check that the repository URL hasn't changed.  If it has, update to
        # the new URL
        local_copy_info = paver.svn.info(self.local_path)
        if local_copy_info.repository_root != self.remote_url:
            sh('svn switch --relocate {orig_url} {new_url}'.format(
                orig_url=local_copy_info.repository_root,
                new_url=self.remote_url), cwd=self.local_path)

        self._cleanup_and_retry(paver.svn.update, self.local_path, rev)

    def current_rev(self):
        try:
            return paver.svn.info(self.local_path).revision
        except AttributeError:
            # happens when we're in a dry run
            # In this case, paver.svn.info() returns an empty Bunch object.
            # Returning 'Unknown' for now until we implement something more
            # stable.
            warnings.warn('SVN version info does not work when in a dry run')
            return 'Unknown'

    def format_rev(self, rev):
        return rev


class GitRepository(DVCSRepository):
    tip = 'master'
    statedir = '.git'
    cmd = 'git'

    def clone(self, rev=None):
        sh('git clone {url} {dest}'.format(**{'url': self.remote_url,
                                              'dest': self.local_path}))
        if rev is None:
            rev = self.tracked_version()
            self.update(rev)

    def pull(self):
        sh('git fetch %(url)s' % {'url': self.remote_url}, cwd=self.local_path)

    def update(self, rev):
        update_string = 'git checkout {rev}'.format(rev=rev)
        self.pull_and_retry(update_string, cwd=self.local_path)

    def current_rev(self):
        rev_cmd_string = 'git rev-parse --verify HEAD'
        return self.pull_and_retry(rev_cmd_string, cwd=self.local_path)

    def format_rev(self, rev):
        log_string = 'git log --format=format:%H -1 {rev}'.format(rev=rev)
        return self.pull_and_retry(log_string, cwd=self.local_path)


REPOS_DICT = {
    'users-guide': HgRepository('doc/users-guide', 'https://bitbucket.org/natcap/invest.users-guide'),
    'invest-data': SVNRepository('data/invest-data', 'svn://scm.naturalcapitalproject.org/svn/invest-sample-data'),
    'test-data': SVNRepository('data/invest-test-data', 'svn://scm.naturalcapitalproject.org/svn/invest-test-data'),
    'invest-2': HgRepository('src/invest-natcap.default', 'http://bitbucket.org/natcap/invest.arcgis'),
    'pyinstaller': GitRepository('src/pyinstaller', 'https://github.com/pyinstaller/pyinstaller.git'),
    'pygeoprocessing': HgRepository('src/pygeoprocessing', 'https://bitbucket.org/richpsharp/pygeoprocessing'),
}
REPOS = REPOS_DICT.values()


def _invest_version(python_exe=None):
    """
    Load the InVEST version string and return it.

    Fetches the string from natcap.invest if the package is installed and
    is able to be imported.  Otherwise, fetches the version string from
    the natcap.invest source.

    Parameters:
        python_exe=None (string): The path to the python interpreter to use.
            If None, the PATH python will be used.

    Returns:
        The version string.
    """

    try:
        import natcap.versioner
        print 'Retrieved version from natcap.versioner'
        return natcap.versioner.parse_version()
    except ImportError:
        print 'natcap.versioner not available'

    try:
        import natcap.invest as invest
        print 'Retrieved version from natcap.invest'
        return invest.__version__
    except ImportError:
        print 'natcap.invest not available'

    if python_exe is None:
        python_exe = 'python'
    else:
        python_exe = os.path.abspath(python_exe)

    # try to get the version from setup.py
    invest_version = sh('{python} setup.py --version'.format(
        python=python_exe), capture=True).rstrip()

    invest_version_strings = invest_version.split(os.linesep)
    if len(invest_version_strings) > 1:

        print 'Retrieved version from setup.py --version'
        # leave out the PEP440 warning strings if present.
        if platform.system() == 'Windows':
            return invest_version_strings[0]
        else:
            return invest_version_strings[-1]

    if invest_version != '':
        # In case the version wasn't imported for some reason.
        return invest_version

    # try to get it from the designated python
    invest_version = sh(
        '{python} -c "import natcap.invest; print natcap.invest.__version__"'.format(
            python=python_exe),
        capture=True).rstrip()
    print 'Retrieved version from site-packages'
    return invest_version


def _repo_is_valid(repo, options):
    # repo is a repository object
    # options is the Options object passed in when using the @cmdopts
    # decorator.
    try:
        options.force_dev
    except AttributeError:
        # options.force_dev not specified as a cmd opt, defaulting to False.
        options.force_dev = False

    if not os.path.exists(repo.local_path):
        print "WARNING: Repository %s has not been cloned." % repo.local_path
        print "To clone, run this command:"
        print "    paver fetch %s" % repo.local_path
        return False

    if not repo.at_known_rev() and not options.force_dev:
        current_rev = repo.current_rev()
        print 'ERROR: Revision mismatch in repo %s' % repo.local_path
        print '*****  Repository at rev %s' % current_rev
        print '*****  Expected rev: %s' % repo.tracked_version()
        print '*****  To override, use the --force-dev flag.'
        return False
    return True


@task
@cmdopts([
    ('json', '', 'Export to json'),
    ('save', '', 'Write json to versions.json')
])
def version(options):
    """
    Display the versions of nested repositories and exit.  UNIMPLEMENTED
    """
    # If --json and --save are both specified, raise an error.
    # These options should be mutually exclusive.
    if options.json and options.save:
        raise BuildFailure("ERROR: --json and --save are mutually exclusive")

    # print the version information.
    data = dict((repo.local_path, repo.current_rev() if os.path.exists(
        repo.local_path) else None) for repo in REPOS)
    json_string = json.dumps(data, sort_keys=True, indent=4)
    try:
        options.json
        print json_string
        return
    except AttributeError:
        pass

    try:
        options.save
        open('versions.json', 'w').write(json_string)
        return
    except AttributeError:
        pass

    # print a formatted table of repository versions and whether the repo is at
    # the known version.
    # Columns:
    # local_path | repo_type | rev_matches
    repo_col_width = max(map(lambda x: len(x.local_path), REPOS)) + 4
    fmt_string = "%(path)-" + str(repo_col_width) + "s %(type)-10s %(is_tracked)-10s"
    data = []
    for repo in sorted(REPOS, key=lambda x: x.local_path):
        if not os.path.exists(repo.local_path):
            at_known_rev = 'not cloned'
        else:
            try:
                at_known_rev = repo.at_known_rev()
                if not at_known_rev:
                    at_known_rev = 'MODIFIED'
            except KeyError:
                at_known_rev = 'UNTRACKED'

        data.append({
            "path": repo.local_path,
            "type": repo.cmd,
            "is_tracked": at_known_rev,
        })

    this_repo_rev = sh('hg log -r . --template="{node}"', capture=True)
    this_repo_branch = sh('hg branch', capture=True)
    local_version = _get_local_version()
    print
    print '*** THIS REPO ***'
    print 'Pretty: %s' % local_version
    print 'Rev:    %s' % this_repo_rev
    print 'Branch: %s' % this_repo_branch

    print
    print '*** SUBREPOS ***'
    headers = {"path": 'REPO PATH', "type": 'REPO TYPE', "is_tracked": 'AT TRACKED REV'}
    print fmt_string % headers
    for repo_data in data:
        print fmt_string % repo_data


@task
@cmdopts([
    ('envname=', 'e', 'The name of the environment to use'),
    ('noinvest', '', 'Skip installing InVEST'),
    ('compiler=', 'c', ('The compiler to use.  Must be a valid distutils '
                      'compiler string. See `python setup.py build '
                      '--help-compiler` for available compiler strings.'))
])
def dev_env(options):
    """
    Set up a development environment with common parameters.


    Setup a development environment with:
        * access to system-site-packages
        * InVEST installed
        * pygeoprocessing installed

    Saved to test_env, or the envname of choice.  If an env of the same name
    exists, clear out the existing env.
    """
    call_task('env', options={
        'system_site_packages': True,
        'clear': True,
        'with_invest': not options.dev_env.noinvest,
        'with_pygeoprocessing': True,
        'envname': options.dev_env.envname,
        'compiler': options.dev_env.compiler,
        'dev': True,
    })


def _read_requirements_dict():
    """Read requirments.txt into a dict.

    Returns:
        A dict mapping {projectname: requirement}.

    Example:
        >>> _read_requirements_dict()
        {'pygeoprocessing': 'pygeoprocessing>=0.3.0a12', ...}
    """
    reqs = {}
    for line in open('requirements.txt'):
        line = line.strip()
        parsed_req = pkg_resources.Requirement.parse(line)
        reqs[parsed_req.project_name] = line
    return reqs


def _parse_version_requirement(pkg_name):
    """Parse a version requirement from requirements.txt.

    Returns the first parsed version that meets the >= requirement.

    Parameters:
        pkg_name (string): The string package name to search for.

    Returns:
        The string version or None if no >= version requirement can be parsed.

    """
    for line in open('requirements.txt'):
        if line.startswith(pkg_name):
            # Assume that we only have one version spec to deal with.
            version_specs = pkg_resources.Requirement.parse(line).specs
            for requirement_type, version in version_specs:
                if requirement_type == '>=':
                    return version


@task
@cmdopts([
    ('system-site-packages', '', ('Give the virtual environment access '
                                  'to the global site-packages')),
    ('clear', '', 'Clear out the non-root install and start from scratch.'),
    ('envname=', 'e', ('The name of the environment to use')),
    ('with-invest', '', 'Install the current version of InVEST into the env'),
    ('with-pygeoprocessing', '', ('Install the current version of '
                                  'pygeoprocessing into the env')),
    ('requirements=', 'r', 'Install requirements from a file'),
    ('dev', 'd', ('Install InVEST namespace packages as flat eggs instead of '
                  'in a single folder hierarchy.  Better for development, '
                  'not so great for pyinstaller build')),
    ('compiler=', 'c', ('The compiler to use.  Must be a valid distutils '
                      'compiler string. See `python setup.py build '
                      '--help-compiler` for available compiler strings.'))
])
def env(options):
    """
    Set up a virtualenv for the project.
    """
    # paver provides paver.virtual.bootstrap(), but this does not afford the
    # degree of control that we want and need with installing needed packages.
    # We therefore make our own bootstrapping function calls here.
    install_string = """
import os, subprocess, platform
def after_install(options, home_dir):
    etc = join(home_dir, 'etc')
    if not os.path.exists(etc):
        os.makedirs(etc)
    if platform.system() == 'Windows':
        bindir = 'Scripts'
        distutils_dir = os.path.join(home_dir, 'Lib', 'distutils')
    else:
        bindir = 'bin'
        distutils_dir = os.path.join(home_dir, 'lib', 'python27', 'distutils')
    distutils_cfg = os.path.join(distutils_dir, 'distutils.cfg')

"""

    # If the user has a distutils.cfg file defined in their global distutils
    # installation, copy that over.
    source_file = os.path.join(distutils.__path__[0],
                               'distutils.cfg')
    install_string += (
        "    if os.path.exists('{src_distutils_cfg}'):\n"
        "       if not os.path.exists(distutils_dir):\n"
        "           os.makedirs(distutils_dir)\n"
        "       shutil.copyfile('{src_distutils_cfg}', distutils_cfg)\n"
    ).format(src_distutils_cfg=source_file)

    # Track preinstalled packages so we don't install them twice.
    preinstalled_pkgs = set([])

    if options.env.compiler:
        _valid_compilers = distutils.ccompiler.compiler_class.keys()
        if options.env.compiler not in _valid_compilers:
            raise BuildFailure('Invalid compiler: %s not in %s' % (
                options.env.compiler, _valid_compilers))
        print 'Preferring user-defined compiler %s' % str(options.env.compiler)
        # If the user defined a compiler to use, customize the available pip
        # options to pass the compiler flag through to the setup.py build_ext
        # command that precedes the install.
        compiler_string = ("'--global-option', 'build_ext', '--global-option', "
                           "'--compiler={compiler}', ").format(
                               compiler=options.env.compiler)
    else:
        # If the user didn't specify the compiler as a command-line option,
        # we'll default to whatever pip thinks is best.
        compiler_string = ''

    if options.env.with_pygeoprocessing:
        # Verify that natcap.versioner is present and importable.
        # pygeoprocessing won't install properly unless this is present.
        _import_namespace_pkg('versioner')

        # Check and update the pygeoprocessing repo if needed.
        call_task('check_repo', options={
            'force-dev': False,
            'repo': 'src/pygeoprocessing',
            'fetch': True,
        })

        try:
            # Determine the required pygeoprocessing and only install it to the
            # env if the system version isn't suitable.
            pygeo_version = REPOS_DICT['pygeoprocessing'].tracked_version(
                convert=False)
            pkg_resources.require('pygeoprocessing>=%s' % pygeo_version)
        except (pkg_resources.DistributionNotFound,
                pkg_resources.VersionConflict) as (required_pkg, found_pkg):
            print yellow(('Unsuitable pygeoprocessing %s found, but %s '
                          'required. Installing the correct version to the '
                          'dev_env.') % (found_pkg, required_pkg))
        # install with --no-deps (will otherwise try to install numpy, gdal,
        # etc.), and -I to ignore any existing pygeoprocessing install (as
        # might exist in system-site-packages).
        # Installing as egg grants pygeoprocessing greater precendence in the
        # import order.  If I install as a wheel, the system install of
        # pygeoprocessing takes precedence.  I believe this to be a bug in
        # pygeoprocessing (poster, for example, does not have this issue!).
        install_string += (
            "    subprocess.call([join(home_dir, bindir, 'pip'), 'install', "
            "'--no-deps', '-I', '--egg', {compiler_flags} "
            "'./src/pygeoprocessing'])\n"
        ).format(compiler_flags=compiler_string)
        preinstalled_pkgs.add('pygeoprocessing')
    else:
        print 'Skipping the installation of pygeoprocessing per user input.'

    requirements_files = ['requirements.txt']
    if options.env.requirements not in [None, '']:
        requirements_files.append(options.env.requirements)

    # extra parameter strings needed for installing certain packages
    # Always install nose, natcap.versioner to the env over whatever else
    # is there.
    pkg_pip_params = {
        'nose': ['-I'],
        'natcap.versioner': ['-I'],
        # Pygeoprocessing wheels are compiled against specific versions of
        # numpy.  Sometimes the wheel on PyPI is incompatible with the locally
        # installed numpy.  Force compilation from source to avoid this issue.
        'pygeoprocessing': NO_WHEEL_SH.split(),
    }
    if options.env.dev:
        # I only want to install natcap namespace packages as flat wheels if
        # we're in a development environment (not a build environment).
        # Pyinstaller seems to work best with namespace packages that are all
        # in a single source tree, though python will happily import multiple
        # eggs from different places.
        pkg_pip_params['natcap.versioner'] += ['--egg', '--no-use-wheel']

    def _format_params(param_list):
        """
        Convert a list of string parameters to a string suitable for adding to
        the environment bootstrap file.

        Returns:
            A string
        """
        params_as_strings = ["'{param}'".format(param=x) for x in param_list]
        extra_params = ", %s" % ', '.join(params_as_strings)
        return extra_params

    pip_template = "    subprocess.call([join(home_dir, bindir, 'pip'), 'install', '{pkgname}' {extra_params}])\n"
    for reqs_file in requirements_files:
        for requirement in pkg_resources.parse_requirements(open(reqs_file).read()):
            projectname = requirement.project_name  # project name w/o version req
            if projectname in preinstalled_pkgs:
                print ('Requirement %s from requirements.txt already '
                       'handled by bootstrap script') % projectname
                continue
            try:
                install_params = pkg_pip_params[projectname]
                extra_params = _format_params(install_params)
            except KeyError:
                # No extra parameters needed for this package.
                extra_params = ''

            install_string += pip_template.format(pkgname=requirement, extra_params=extra_params)

    if options.env.with_invest:
        # Build an sdist and install it as an egg.  Works better with
        # pyinstaller, it would seem.  Also, namespace packages complicate
        # imports, so installing all natcap pkgs as eggs seems to work as
        # expected.
        install_string += (
            "    subprocess.call([join(home_dir, bindir, 'python'), 'setup.py', 'egg_info', 'sdist', '--formats=gztar'])\n"
            "    version = subprocess.check_output([join(home_dir, bindir, 'python'), 'setup.py', '--version'])\n"
            "    version = version.rstrip()  # clean trailing whitespace\n"
            "    invest_sdist = join('dist', 'natcap.invest-{version}.tar.gz'.format(version=version))\n"
            "    # Sometimes, don't know why, sdist ends up with - instead of + as local ver. separator.\n"
            "    if not os.path.exists(invest_sdist):\n"
            "        invest_sdist = invest_sdist.replace('+', '-')\n"
        )

        if not options.env.dev:
            install_string += (
                # Recent versions of pip build wheels by default before
                # installing, but wheel has a bug preventing builds for
                # namespace packages.
                # Therefore, skip wheel builds for invest.
                # Pyinstaller also doesn't handle namespace packages all that
                # well, so --egg --no-use-wheel doesn't really work in a
                # release environment either.
                "    subprocess.call([join(home_dir, bindir, 'pip'), 'install'"
                ", {no_wheel_flag}, {compiler_flags}"
                " invest_sdist])\n"
            ).format(no_wheel_flag=NO_WHEEL_SUBPROCESS,
                     compiler_flags=compiler_string)
        else:
            install_string += (
                # If we're in a development environment, it's ok (even
                # preferable) to install natcap namespace packages as flat
                # eggs.
                "    subprocess.call([join(home_dir, bindir, 'pip'), "
                "'install', '--egg', {no_wheel_flag}, {compiler_flags}"
                " invest_sdist])\n"
            ).format(no_wheel_flag=NO_WHEEL_SUBPROCESS,
                     compiler_flags=compiler_string)
    else:
        print 'Skipping the installation of natcap.invest per user input'

    output = virtualenv.create_bootstrap_script(textwrap.dedent(install_string))
    open(options.env.bootstrap_file, 'w').write(output)

    # Built the bootstrap env via a subprocess call.
    # Calling via the shell so that virtualenv has access to environment
    # vars as needed.
    try:
        pkg_resources.require('virtualenv>=13.0.0')
        no_wheel_flag = '--no-wheel'
    except pkg_resources.VersionConflict:
        # Early versions of virtualenv don't ship wheel, so there's no flag for
        # us to provide.
        no_wheel_flag = ''

    bootstrap_cmd = "%(python)s %(bootstrap_file)s %(site-pkgs)s %(clear)s %(no-wheel)s %(env_name)s"
    bootstrap_opts = {
        "python": sys.executable,
        "bootstrap_file": options.env.bootstrap_file,
        "env_name": options.env.envname,
        "site-pkgs": '--system-site-packages' if options.env.system_site_packages else '',
        "clear": '--clear' if options.env.clear else '',
        "no-wheel": no_wheel_flag,  # exclude wheel.  It has a bug preventing namespace pkgs from compiling
    }
    sh(bootstrap_cmd % bootstrap_opts)

    # Virtualenv appears to partially copy over distutills into the new env.
    # Remove what was copied over so we din't confuse pyinstaller.
    if platform.system() == 'Windows':
        init_file = os.path.join(options.env.envname, 'Lib', 'site-packages', 'natcap', '__init__.py')
    else:
        init_file = os.path.join(options.env.envname, 'lib', 'python2.7', 'site-packages', 'natcap', '__init__.py')

    if options.with_invest and not options.env.dev:
        # writing this import appears to help pyinstaller find the __path__
        # attribute from a package.  Only write it if InVEST is installed and
        # is installed as a package (not a dev environment)
        init_string = "import pkg_resources\npkg_resources.declare_namespace(__name__)\n"
        with open(init_file, 'w') as namespace_init:
            namespace_init.write(init_string)

    print '*** To activate the env, run:'
    if platform.system() == 'Windows':
        print r'    call .\%s\Scripts\activate' % options.env.envname
    else:  # assume all POSIX systems behave the same way
        print '    source %s/bin/activate' % options.env.envname


@task
@consume_args  # when consuming args, it's a list of str arguments.
def fetch(args, options):
    """
    Clone repositories the correct locations.
    """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('repo', metavar='REPO[@rev]', nargs='*',
                            help=('The repository to fetch.  Optionally, the '
                                  'revision to update to can be specified by '
                                  'using the "@" symbol.  Example: '
                                  ' `paver fetch data/invest-data@27`. '
                                  'If no repos are specified, all known repos '
                                  'will be fetched.  Specifying an argument of'
                                  ' "*" will also cause all repos to be '
                                  'fetched.'))

    # figure out which repos/revs we're hoping to update.
    # None is our internal, temp keyword representing the LATEST possible
    # rev.
    user_repo_revs = {}  # repo -> version
    parsed_args = arg_parser.parse_args(args[:])
    for user_repo in parsed_args.repo:
        repo_parts = user_repo.split('@')
        if len(repo_parts) == 1:
            repo_name = repo_parts[0]
            repo_rev = None
        elif len(repo_parts) == 2:
            repo_name, repo_rev = repo_parts
        else:
            raise BuildFailure("Can't parse repo/rev {repo}".format(user_repo))

        # if the repo name ends with a trailing / (or \\ on Windows), trim it
        # Improves comparison of repo strings.  Tab-completion likes to add the
        # trailing slash.
        if repo_name.endswith(os.sep):
            repo_name = repo_name[:-1]
        user_repo_revs[repo_name] = repo_rev

    # We include all repos if EITHER the user has not provided any arguments at
    # all OR the one argument present is a *
    include_all_repos = (parsed_args.repo == [] or parsed_args.repo == ['*'])

    # determine which known repos the user wants to operate on.
    # example: `src` would represent all repos under src/
    # example: `data` would represent all repos under data/
    # example: `src/pyinstaller` would represent the pyinstaller repo
    desired_repo_revs = {}
    known_repos = dict((repo.local_path, repo) for repo in REPOS)
    for known_repo_path, repo_obj in known_repos.iteritems():
        if include_all_repos:
            # If no repos were specified as input to this function, fetch them
            # all!  Use the version in versions.json.
            desired_repo_revs[repo_obj] = repo_obj.tracked_version()
        else:
            for user_repo, user_rev in user_repo_revs.iteritems():
                if user_repo in known_repo_path:
                    if known_repo_path in desired_repo_revs:
                        raise BuildFailure('The same repo has been selected '
                                           'twice')
                    else:
                        desired_repo_revs[repo_obj] = user_rev

    for user_requested_repo, target_rev in desired_repo_revs.iteritems():
        print 'Fetching {path}'.format(path=user_requested_repo.local_path)

        # If the user did not define a target rev, we use the one on disk.
        if target_rev is None:
            try:
                target_rev = user_requested_repo.tracked_version()
            except KeyError:
                repo_path = user_requested_repo.local_path
                raise BuildFailure(('Repo not tracked in versions.json: '
                                   '{repo}').format(repo=repo_path))

        if options.dry_run:
            print 'Fetching {path}'.format(user_requested_repo.local_path)
            continue
        else:
            user_requested_repo.get(target_rev)


@task
@consume_args
def push(args):
    """Push a file or files to a remote server.

    Usage:
        paver push [--private-key=KEYFILE] [--password] [--makedirs] [user@]hostname[:target_dir] file1, file2, ...

    Uses pythonic paramiko-based SCP to copy files to the remote server.

    if --private-key=KEYFILE is provided, KEYFILE must be the path to the private
    key file to use.  If this file cannot be found, BuildFailure will be raised.

    If --password is provided at the command line, the user will be prompted
    for a password.  This is sometimes required when the remote's private key
    requires a password to decrypt.

    If --makedirs is provided, intermediate directories will be created as needed.

    If a target username is not provided ([user@]...), the current user's username
    used for the transfer.

    If a target directory is not provided (hostname[:target_dir]), the current
    directory of the target user is used.
    """
    print args
    import paramiko

    paramiko.util.log_to_file('paramiko-log.txt')

    from paramiko import SSHClient
    ssh = SSHClient()
    ssh.load_system_host_keys()

    # Automatically add host key if needed
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Clean out all of the user-configurable options flags.
    config_opts = []
    for argument in args[:]:  # operate on a copy of args
        if argument.startswith('--'):
            config_opts.append(argument)
            args.remove(argument)

    use_password = '--password' in config_opts

    # check if the user specified a private key to use
    private_key = None
    for argument in config_opts:
        if argument.startswith('--private-key'):
            private_key_file = argument.split('=')[1]
            if not os.path.exists(private_key_file):
                raise BuildFailure(
                    'Cannot fild private key file %s' % private_key_file)
            print 'Using private key %s' % private_key_file
            private_key = paramiko.RSAKey.from_private_key_file(private_key_file)
            break
    try:
        destination_config = args[0]
    except IndexError:
        raise BuildFailure("ERROR: destination config must be provided")

    files_to_push = args[1:]
    if len(files_to_push) == 0:
        raise BuildFailure("ERROR: At least one file must be given")

    def _fix_path(path):
        """Fix up a windows path to work on linux"""
        # destination OS is linux, so adjust windows filepaths to match
        if platform.system() == 'Windows':
            return path.replace(os.sep, '/')
        return path

    # ASSUME WE'RE ONLY DOING ONE HOST PER PUSH
    # split apart the configuration string.
    # format:
    #    [user@]hostname[:directory]
    if '@' in destination_config:
        username = destination_config.split('@')[0]
        destination_config = destination_config.replace(username + '@', '')
    else:
        username = getpass.getuser().strip()

    if ':' in destination_config:
        target_dir = destination_config.split(':')[-1]
        destination_config = destination_config.replace(':' + target_dir, '')
        target_dir = _fix_path(target_dir)
    else:
        # just use the SCP default
        target_dir = None
    print 'Target dir: %s' % target_dir
    print 'Dest config: %s' % destination_config

    # hostname is whatever remains of the dest config.
    hostname = destination_config.strip()
    print 'Hostname: %s' % hostname

    # start up the SSH connection
    if use_password:
        password = getpass.getpass()
    else:
        password = None

    try:
        ssh.connect(hostname, 22, username=username, password=password, pkey=private_key)
    except paramiko.BadAuthenticationType:
        raise BuildFailure('ERROR: incorrect password or bad SSH key')
    except paramiko.PasswordRequiredException:
        raise BuildFailure('ERROR: password required to decrypt private key on remote.  Use --password flag')
    except socket.error as other_error:
        raise BuildFailure(other_error)

    # Make folders on remote if needed.
    if target_dir is not None and '--makedirs' in config_opts:
        ssh.exec_command('if [ ! -d "{dir}" ]\nthen\nmkdir -p -v {dir}\nfi'.format(
            dir=target_dir))
    else:
        print 'Skipping creation of folders on remote'

    def _sftp_callback(bytes_transferred, total_bytes):
        try:
            current_time = time.time()
            if current_time - _sftp_callback.last_time > 2:
                tx_ratio = bytes_transferred / float(total_bytes)
                tx_ratio = round(tx_ratio*100, 2)

                print 'SFTP copied {transf} out of {total} ({ratio} %)'.format(
                    transf=bytes_transferred, total=total_bytes, ratio=tx_ratio)
                _sftp_callback.last_time = current_time
        except AttributeError:
            _sftp_callback.last_time = time.time()

    print 'Opening SCP connection'
    sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
    for transfer_file in files_to_push:
        file_basename = os.path.basename(transfer_file)
        if target_dir is not None:
            target_filename = os.path.join(target_dir, file_basename)
        else:
            target_filename = file_basename

        target_filename = _fix_path(target_filename)  # convert windows to linux paths
        print 'Transferring %s -> %s ' % (os.path.basename(transfer_file), target_filename)
        for repeat in [True, True, False]:
            try:
                sftp.put(transfer_file, target_filename, callback=_sftp_callback)
            except IOError as filesize_inconsistency:
                # IOError raised when the file on the other end reports a
                # different filesize than what we sent.
                if not repeat:
                    raise filesize_inconsistency


    print 'Closing down SCP'
    sftp.close()

    print 'Closing down SSH'
    ssh.close()


@task
def clean(options):
    """
    Remove files and folders known to be generated by build scripts.
    """

    folders_to_rm = ['build', 'dist', 'tmp', 'bin', 'test',
                     options.env.envname,
                     'installer/darwin/temp',
                     'invest-3-x86',
                     'exe/dist',
                     'exe/build',
                     'api_env',
                     'natcap.invest.egg-info',
                     'release_env',
                     'test_env',
                     'invest-bin',
                     ]
    files_to_rm = [
        options.env.bootstrap_file,
        'installer/linux/*.deb',
        'installer/linux/*.rpm',
        'installer/darwin/*.dmg',
        'installer/windows/*.exe',
    ]

    for folder in folders_to_rm:
        for globbed_dir in glob.glob(folder):
            paver.path.path(globbed_dir).rmtree()

    for filename in files_to_rm:
        for globbed_file in glob.glob(filename):
            paver.path.path(globbed_file).remove()

    # clean out all python package repos in src/
    for repodir in map(lambda x: x.local_path, REPOS):
        if repodir.startswith('src'):
            if os.path.exists(os.path.join(repodir, 'setup.py')):
                # use setup.py for package directories.
                sh(sys.executable + ' setup.py clean', cwd=repodir)
        elif repodir.startswith('doc') and os.path.exists(repodir):
            sh('make clean', cwd=repodir)


@task
@might_call('zip')
@cmdopts([
    ('force-dev', '', 'Zip subrepos even if their version does not match the known state'),
])
def zip_source(options):
    """
    Create a zip archive of all source repositories for this project.

    Creates a standalone zip file that, when extracted, will contain all source code
    needed to create documentation and functional binaries from the various projects
    managed by this project.  If there's a source repo in this repository, its source
    is in this archive.

    If --force-dev is provided, the state of the contained subrepositories/subprojects
    is allowed to differ from the revision noted in versions.json.  If the state of
    the subproject/subrepository does not match the state noted in versions.json and
    --force-dev is NOT provided, an error will be raised.

    The output file is written to dist/InVEST-source-{version}.zip
    The version used is compiled from the state of the repository.
    """

    version = _get_local_version()

    source_dir = os.path.join('tmp', 'source')
    invest_bin_zip = os.path.join('tmp', 'invest-bin.zip')
    invest_dir = os.path.join('tmp', 'source', 'invest-bin')
    dist_dir = 'dist'
    try:
        dry('mkdir -p %s' % dist_dir, os.makedirs, source_dir)
    except OSError:
        # Folder already exists.  Skipping.
        pass
    sh('hg archive %s' % invest_bin_zip)

    def _unzip(zip_uri, dest_dir):
        def _unzip_func():
            zip = zipfile.ZipFile(zip_uri)
            zip.extractall(dest_dir)
        dry('unzip -o %s -d %s' % (zip_uri, dest_dir), _unzip_func)

    _unzip(invest_bin_zip, source_dir)

    for dirname in map(lambda x: x.local_path, REPOS):
        if not dirname[0:3] in ['doc', 'src']:
            continue

        if dirname.startswith('src'):
            source_dir = os.path.join(invest_dir, 'src')
        elif dirname.startswith('doc'):
            source_dir = os.path.join(invest_dir, 'doc')

        projectname = dirname[4:]  # remove the / as well.
        unzipped_dir = os.path.join(source_dir, projectname)
        print unzipped_dir
        try:
            dry('rm -r %s' % unzipped_dir, shutil.rmtree, unzipped_dir)
        except OSError:
            # when the source dir doesn't exist, that's ok.
            pass

        sh('hg archive -R %(repo)s tmp/%(zipname)s.zip' % {
            'repo': dirname, 'zipname': projectname})

        zipfile_name = projectname + '.zip'
        _unzip(os.path.join('tmp', zipfile_name), source_dir)

    # leave off the .zip filename here.  shutil.make_archive adds it based on
    # the format of the archive.
    archive_name = os.path.abspath(os.path.join('dist', 'InVEST-source-%s' % version))
    call_task('zip', args=[archive_name, source_dir])


@task
@might_call('zip')
@cmdopts([
    ('force-dev', '', ('Allow docs to build even if repo version does not '
                       'match the known state')),
    ('skip-api', '', 'Skip building the API docs'),
    ('skip-guide', '', "Skip building the User's Guide"),
    ('python=', '', 'The python interpreter to use'),
])
def build_docs(options):
    """
    Build the sphinx user's guide for InVEST.

    Builds the sphinx user's guide in HTML, latex and PDF formats.
    Compilation of the guides uses sphinx and requires that all needed
    libraries are installed for compiling html, latex and pdf.

    Requires make and sed.
    """

    invest_version = _invest_version(options.build_docs.python)
    archive_template = os.path.join('dist', 'invest-%s-%s' % (invest_version, '%s'))

    print 'Using this template for the archive name: %s' % archive_template

    # If the user has not provided the skip-guide flag, build the User's guide.
    skip_guide = getattr(options, 'skip_guide', False)
    if not skip_guide:
        call_task('check_repo', options={
            'force_dev': options.build_docs.force_dev,
            'repo': REPOS_DICT['users-guide'].local_path,
            'fetch': True,
        })

        guide_dir = os.path.join('doc', 'users-guide')
        latex_dir = os.path.join(guide_dir, 'build', 'latex')
        sh('make html', cwd=guide_dir)
        sh('make latex', cwd=guide_dir)
        sh('make all-pdf > latex-warnings.log', cwd=latex_dir)

        archive_name = archive_template % 'userguide'
        build_dir = os.path.join(guide_dir, 'build', 'html')
        call_task('zip', args=[archive_name, build_dir, 'userguide'])
    else:
        print "Skipping the User's Guide"

    skip_api = getattr(options, 'skip_api', False)
    if not skip_api:
        sh('{python} setup.py build_sphinx'.format(python=options.build_docs.python))
        archive_name = archive_template % 'apidocs'
        call_task('zip', args=[archive_name, 'build/sphinx/html', 'apidocs'])
    else:
        print "Skipping the API docs"


@task
@no_help  # users should use `paver version` to see the repo states.
@might_call(['fetch'])
@cmdopts([
    ('force-dev', '', 'Allow a development version of the repo'),
    ('repo', '', 'The repo to check'),
    ('fetch', '', 'Fetch the repo if needed'),
])
def check_repo(options):

    # determine the current repo_object
    repo_path = options.check_repo.repo
    repo = None
    for possible_repo in REPOS:
        if possible_repo.local_path == repo_path:
            repo = possible_repo

    if repo is None:
        raise BuildFailure('Repo %s is invalid' % repo_path)

    if not repo.ischeckedout() and not options.check_repo.fetch:
        print (
            'Repo %s is not checked out. '
            'Use `paver fetch %s`.' % (
                repo.local_path, repo.local_path))
        return

    if options.check_repo.fetch:
        call_task('fetch', args=[repo_path])

    if not repo.ischeckedout():
        return

    tracked_rev = repo.format_rev(repo.tracked_version())
    current_rev = repo.current_rev()
    if tracked_rev != current_rev:
        if not options.check_repo.force_dev:
            raise BuildFailure(
                ('ERROR: %(local_path)s at rev %(cur_rev)s, '
                    'but expected to be at rev %(exp_rev)s') % {
                    'local_path': repo.local_path,
                    'cur_rev': current_rev,
                    'exp_rev': tracked_rev
                })
        else:
            print 'WARNING: %s revision differs, but --force-dev provided' % repo.local_path
    print 'Repo %s is at rev %s' % (repo.local_path, tracked_rev)



@task
@cmdopts([
    ('fix-namespace', '', 'Fix issues with the natcap namespace if found'),
    ('allow-errors', '', 'Errors will be printed, but the task will not fail'),
])
def check(options):
    """
    Perform reasonable checks to verify the build environment.


    This task checks for the presence of required binaries, python packages
    and for known issues with the natcap python namespace.
    """
    # verify required programs exist
    errors_found = False
    programs = [
        ('hg', 'everything'),
        ('git', 'binaries'),
        ('svn', 'testing, installers'),
        ('make', 'documentation'),
        ('pdflatex', 'documentation'),
        ('pandoc', 'documentation'),
    ]
    if platform.system() == 'Linux':
        programs.append(('fpm', 'installers'))

    print bold("Checking binaries")
    for program, build_steps in programs:
        # Inspired by this SO post: http://stackoverflow.com/a/855764/299084

        try:
            path_to_exe = find_executable(program)
        except EnvironmentError as exception_msg:
            errors_found = True
            print "{error} {exe} not found. Required for {step}".format(
                error=ERROR, exe=program, step=build_steps)
        else:
            found_exe = True
            print "Found %-14s: %s" % (program, path_to_exe)

    required = 'required'
    suggested = 'suggested'
    lib_needed = 'lib_needed'
    install_managed = 'install_managed'

    # (requirement, level, version_getter, special_install_message)
    # requirement: This is the setuptools package requirement string.
    # level: one of required, suggested, lib_needed.
    # version_getter: some packages are imported by a different name.
    #    This is that name.  If None, default to the requirement's distname.
    # special_install_message: A special installation message if needed.
    #    If None, no special message will be shown after the conflict report.
    #    This is only for use by required packages.
    #
    # NOTE: This list is for tracking packages with special notes, warnings or
    # import conditions ONLY.  Version requirements should be tracked in
    # versions.json ONLY.
    print bold("\nChecking python packages")
    requirements = [
        # requirement, level, version_getter, special_install_message
        ('setuptools', required, None, None),  # 8.0 implements pep440
        ('virtualenv', required, None, None),
        ('pip', required, None, None),
        ('numpy', lib_needed,  None, None),
        ('scipy', lib_needed,  None, None),
        ('paramiko', suggested, None, None),
        ('pycrypto', suggested, 'Crypto', None),
        ('h5py', lib_needed,  None, None),
        ('gdal', lib_needed,  'osgeo.gdal', None),
        ('shapely', lib_needed,  None, None),
        ('poster', lib_needed,  None, None),
        ('pyyaml', required, 'yaml', None),
        ('pygeoprocessing', install_managed, None, None),
        ('PyQt4', lib_needed, 'PyQt4', None),
    ]

    try:
        # poster stores its version in a triple of ints
        import poster
        poster.__version__ = '.'.join([str(i) for i in poster.version])
    except ImportError:
        # If the package can't be found, this will be caught by pkg_resources
        # below, and the error message will be formatted there.
        pass

    try:
        # PyQt version string is also stored in an awkward place.
        import PyQt4
        from PyQt4.Qt import PYQT_VERSION_STR
        PyQt4.__version__ = PYQT_VERSION_STR
    except ImportError:
        # If the package can't be found, this will be caught by pkg_resources
        # below, and the error message will be formatted there.
        pass

    # pywin32 is required for pyinstaller builds
    if platform.system() == 'Windows':
        # Wheel has an issue with namespace packages on windows.
        # See https://bitbucket.org/pypa/wheel/issues/91
        # I've implemented cgohlke's fix and pushed it to my fork of wheel.
        # To install a working wheel package, do this on your windows install:
        #   pip install hg+https://bitbucket.org/jdouglass/wheel@default
        #
        # This requires that you have command-line hg installed.
        # Setuptools >= 8.0 is required.  Local version notation (+...)
        # will not work with setuptools < 8.0.
        requirements.append(('wheel>=0.25.0+natcap.1', required, None, (
            'pip install --upgrade hg+https://bitbucket.org/jdouglass/wheel'
        )))

        # paver has a restriction within @paver.virtual.virtualenv where it
        # (unnecessarily) always treats the activation of a virtualenv like
        # it's on a POSIX system.  I've submitted a PR to fix this to the
        # upstream paver repo (https://github.com/paver/paver/pull/153),
        # which was merged, but an official release of paver that includes this
        # version has not been made just yet.
        requirements.append(('paver==1.2.4+natcap.1', required, None, (
            'pip install --upgrade '
            'git+https://github.com/phargogh/paver@natcap-version'
        )))

        # Don't need to try/except this ... paver is imported at the top of
        # this file so we know that paver exists.  If it doesn't have a version
        # module, the ImportError should definitely be raised.
        from paver import version
        paver.__version__ = version.VERSION

        try:
            requirements.append(('pywin32', required, 'pywin', None))

            # Get the pywin32 version here, as demonstrated by
            # http://stackoverflow.com/a/5071777.  If we can't import pywin,
            # the __version__ attribute (below) will never be reached.
            import pywin
            import win32api
            fixed_file_info = win32api.GetFileVersionInfo(
                win32api.__file__, '\\')
            pywin.__version__ = fixed_file_info['FileVersionLS'] >> 16
        except ImportError:
            pass
    else:
        # Non-windows OSes also require wheel,just not a special installation
        # of it.
        requirements.append(('wheel', required, None, None))

    # Compare the above-defined requirements with those in requirements.txt
    # The resulting set should be the union of the two.  Package verison
    # requirements should be stored in requirements.txt.
    existing_reqs = set([pkg_resources.Requirement.parse(r[0]).project_name
                         for r in requirements])
    requirements_txt_dict = _read_requirements_dict()
    for reqname, req in requirements_txt_dict.iteritems():
        if reqname not in existing_reqs:
            requirements.append((req, required, None, None))

    warnings_found = False
    for requirement, severity, import_name, install_msg in sorted(
        requirements, key=lambda x: x[0].lower()):
        # We handle natcap namespace packages specially below.
        if requirement.startswith('natcap'):
            continue

        try:
            # If we have a required package version (defined in
            # requirements.txt), use that string.
            requirement = requirements_txt_dict[requirement]
        except KeyError:
            pass

        try:
            pkg_req = pkg_resources.Requirement.parse(requirement)
            if import_name is None:
                import_name = pkg_req.project_name

            try:
                pkg_resources.require(requirement)
            except pkg_resources.DistributionNotFound as missing_req:
                # Some packages (ahem ... PyQt4) are actually importable, but
                # cannot be found by pkg_resources.  We handle this case here
                # by attempting to import the 'missing' package, and raising
                # the DistributionNotFound if we can't import it.
                try:
                    importlib.import_module(import_name)
                except ImportError:
                    raise missing_req

            pkg = __import__(import_name)
            print "Python package {ok}: {pkg} {ver} (meets {req})".format(
                ok=OK,
                pkg=pkg_req.project_name,
                ver=pkg.__version__,
                req=requirement)
        except AttributeError as error:
            print 'Could not define module ', pkg
            raise error
        except (pkg_resources.VersionConflict,
                pkg_resources.DistributionNotFound) as conflict:
            if not hasattr(conflict, 'report'):
                # Setuptools introduced report() in v6.1
                print ('{error} Setuptools is very out of date. '
                    'Upgrade and try again'.format(error=ERROR))
                if not options.check.allow_errors:
                    raise BuildFailure('Setuptools is very out of date. '
                                    'Upgrade and try again')

            if severity == required:
                if install_msg is None:
                    fmt_install_msg = ''
                else:
                    fmt_install_msg = '\nInstall this package via:\n    ' + install_msg
                print 'Python package {error} {report} {msg}'.format(error=ERROR,
                                                      report=conflict.report(),
                                                      msg=fmt_install_msg)
                errors_found = True
            elif severity == lib_needed:
                if isinstance(conflict, pkg_resources.DistributionNotFound):
                    print (
                        '{warning} {report}  This library requires appropriate '
                        'headers to compile the python '
                        'package.').format(warning=WARNING,
                                           report=conflict.report())
                else:
                    print ('{warning} {report}.  You may need to upgrade your '
                           'development headers along with the python '
                           'package.').format(warning=WARNING,
                                              report=conflict.report())
                warnings_found = True
            elif severity == 'install_managed':
                print ('{warning} {pkg} is required, but will be '
                       'installed automatically if needed for paver.').format(
                            warning=WARNING,
                            pkg=requirement)
            else:  # severity is 'suggested'
                print '{warning} {report}'.format(warning=WARNING,
                                                  report=conflict.report())
                warnings_found = True

        except ImportError:
            print '{error} Package not found: {req}'.format(error=ERROR,
                                                            req=requirement)

    # Build in a check for the package setup the natcap namespace, in case the
    # user has globally installed natcap namespace packages.
    # Known problems:
    #   * User has some packages installed to site-packages/natcap,
    #     others installed to site-packages/natcap.pkgname.egg
    #     Will cause errors when importing eggs, as natcap dir is
    #     found first, eggs are skipped.
    #   * User has packages installed as eggs.  Pyinstaller doesn't like this,
    #     for some reason, so warn the user about builds.  Development
    #     should be fine, though.
    #   * User has any packages installed to site-packages/natcap/.  This will
    #     cause problems with importing packages installed to virtualenv.
    #
    # SO:
    #  * If a package is installed to the global site-packages, it better be an
    #    egg.
    noneggs = []
    eggs = []
    try:
        print ""
        print bold("Checking natcap namespace")
        if options.check.fix_namespace:
            print yellow('--fix-namespace provided; Fixing issues as they are'
                         ' encountered')

        import natcap

        for importer, modname, ispkg in pkgutil.iter_modules(natcap.__path__):
            module, pkg_type = _import_namespace_pkg(modname)

            if pkg_type == 'egg':
                eggs.append(modname)
            else:
                noneggs.append(modname)

        if len(noneggs) > 0:
            if options.check.fix_namespace:
                for package in noneggs:
                    print yellow('Reinstalling natcap.%s as egg' % package)
                    sh('pip uninstall -y natcap.{package} > natcap.{package}.log'.format(package=package))
                    sh(('pip install --egg {no_wheel} '
                        'natcap.{package} > natcap.{package}.log').format(
                            package=package, no_wheel=NO_WHEEL_SH))
                    print green('Package natcap.%s reinstalled successfully' % package)
            else:
                pip_inst_template = \
                    yellow("    pip install --egg {no_wheel} natcap.%s").format(
                        no_wheel=NO_WHEEL_SH)
                namespace_msg = (
                    "\n"
                    "Natcap namespace issues:\n"
                    "You appear to have natcap packages installed to your global \n"
                    "site-packages that have not been installed as eggs.\n"
                    "This will cause import issues when trying to build binaries\n"
                    "with pyinstaller, but should work well for development.\n"
                    "By contrast, eggs should work well for development.\n"
                    "For best results, install these packages as eggs like so:\n")
                namespace_msg += "\n".join([pip_inst_template % n for n in noneggs])
                namespace_msg += "\n\nOr run 'paver check --fix-namespace'\n"
                print namespace_msg
                warnings_found = True
        elif len(noneggs) == 0 and len(eggs) == 0:
            base_warning = 'WARNING: namespace artifacts found.'
            if options.check.fix_namespace:
                base_warning += ' Attempting to repair'
            print yellow(base_warning)

            # locate the problematic namespace artifacts.
            namespace_artifacts = []
            namespace_packages_with_artifacts = set([])
            for site_pkgs in site.getsitepackages():
                for namespace_item in glob.glob(os.path.join(
                        site_pkgs, '*natcap*')):
                    namespace_artifacts.append(namespace_item)

                    # Namespace items are usually formatted like this:
                    # natcap.subpackage-version.something OR
                    # natcap.subpackage-version-something
                    # Track the package so we can print it to the user.
                    namespace_packages_with_artifacts.add(
                        os.path.basename(namespace_item).split('-')[0])

            if options.check.fix_namespace:
                for namespace_artifact in namespace_artifacts:
                    print yellow('Removing %s' % namespace_artifact)

                    if os.path.isdir(namespace_artifact):
                        shutil.rmtree(namespace_artifact)
                    else:
                        os.remove(namespace_artifact)
                for ns_item in sorted(namespace_packages_with_artifacts):
                    print yellow('Namespace artifacts from %s cleaned up; you '
                                 'may need to reinstall') % ns_item
            else:
                warnings_found = True
                warn = ('{warning} The natcap namespace is importable, but the '
                        'source could not be found.\n'
                        'This can happen with incomplete uninstallations. '
                        'These artifacts were found\n'
                        'and should be removed:\n').format(warning=WARNING)
                for artifact in namespace_artifacts:
                    warn += ' * %s\n' % artifact
                warn += ("\nUse 'paver check --fix-namespace' to automatically "
                         "remove these files")
                print warn

    except ImportError:
        print 'No natcap installations found.'


    # Check if we need to have versioner installed for setup.py
    setup_file = os.path.join(os.path.dirname(__file__), 'setup.py')
    setup_uses_versioner = False
    with open(setup_file) as setup:
        for line in setup:
            if 'import natcap.versioner' in line:
                setup_uses_versioner = True
                break

    if setup_uses_versioner:
        try:
            # If 'versioner' is in eggs, we've already proven that we can
            # import it, so no need to import again.
            if 'versioner' not in eggs:
                _, _ = _import_namespace_pkg('versioner')
        except ImportError:
            if options.check.fix_namespace:
                print yellow('natcap.versioner required by setup.py but '
                                'not found.  Installing.')
                # Install natcap.versioner
                sh('pip install --egg {no_wheel} natcap.versioner > natcap.versioner.log'.format(no_wheel=NO_WHEEL_SH))

                # Verify that versioner installed properly.  Must import in new
                # process to verify. _import_namespace_pkg allows for pretty
                # printing.
                try:
                    sh('python -c "'
                        'import pavement;'
                        'pavement._import_namespace_pkg(\'versioner\')'
                        '"')
                    print green('natcap.versioner successfully installed as egg')
                except BuildFailure:
                    # An exception was raised or some other error encountered.
                    errors_found = True
                    print red('Installation failed: natcap.versioner')
            else:
                warnings_found = True
                print ('{warning} natcap.versioner required by setup.py but not '
                    'installed.  To fix:').format(warning=WARNING)
                print '    pip install --egg {no_wheel} natcap.versioner'.format(no_wheel=NO_WHEEL_SH)
                print 'Or use paver check --fix-namespace'

    if errors_found:
        error_string = (' Programs missing and/or package '
                        'requirements not met')
        if options.check.allow_errors:
            print red('CRITICAL:') + error_string
            print red('CRITICAL:') + ' Ignoring errors per user request'
        else:
            raise BuildFailure(ERROR + error_string)
    elif warnings_found:
        print "\033[93mWarnings found; Builds may not work as expected\033[0m"
    else:
        print green("All's well.")


@task
@cmdopts([
    ('force-dev', '', 'Zip data folders even if repo version does not match the known state')
])
def build_data(options):
    """
    Build data zipfiles for sample data.

    Expects that sample data zipfiles are provided in the invest-data repo.
    Data files should be stored in one directory per model, where the directory
    name matches the model name.  This creates one zipfile per folder, where
    the zipfile name matches the folder name.

    options:
        --force-dev : Provide this option if you know that the invest-data version
                      does not match the version tracked in versions.json.  If the
                      versions do not match and the flag is not provided, the task
                      will print an error and quit.
    """

    data_repo = REPOS_DICT['invest-data']
    call_task('check_repo', options={
        'force_dev': options.build_data.force_dev,
        'repo': data_repo.local_path,
        'fetch': True,
    })

    dist_dir = 'dist'
    if not os.path.exists(dist_dir):
        dry('mkdir %s' % dist_dir, os.makedirs, dist_dir)

    data_folders = os.listdir(data_repo.local_path)
    for data_dirname in data_folders:
        out_zipfile = os.path.abspath(os.path.join(
            dist_dir, os.path.basename(data_dirname) + ".zip"))

        # Only zip up directories in the data repository.
        if not os.path.isdir(os.path.join(data_repo.local_path, data_dirname)):
            continue

        # Don't zip up .svn folders in the data repo.
        if data_dirname == data_repo.statedir:
            continue

        # We don't want Base_Data to be a big ol' zipfile, so we ignore it
        # for now and add its subdirectories (Freshwater, Marine,
        # Terrestrial) as their own zipfiles.
        if data_dirname == 'Base_Data':
            for basedata_subdir in os.listdir(os.path.join(data_repo.local_path, data_dirname)):
                data_folders.append(os.path.join(data_dirname, basedata_subdir))
            continue

        dry('zip -r %s %s' % (out_zipfile, data_dirname),
            shutil.make_archive, **{
                'base_name': os.path.splitext(out_zipfile)[0],
                'format': 'zip',
                'root_dir': data_repo.local_path,
                'base_dir': data_dirname})


@task
@might_call(['fetch', 'check_repo'])
@cmdopts([
    ('force-dev', '', 'Whether to allow development versions of repos to be built'),
    ('python=', '', 'The python interpreter to use'),
], share_with=['check_repo'])
def build_bin(options):
    """
    Build frozen binaries of InVEST.
    """

    pyi_repo = REPOS_DICT['pyinstaller']
    call_task('check_repo', options={
        'force_dev': options.build_bin.force_dev,
        'repo': pyi_repo.local_path,
        'fetch': True,
    })

    # if pyinstaller repo is at version 2.1, remove six.py because it conflicts
    # with the version that matplotlib requires.  Pyinstaller provides
    # six==1.0.0, matplotlib requires six>=1.3.0.
    print 'Checking and removing deprecated six.py in pyinstaller if needed'
    if pyi_repo.current_rev() == pyi_repo.format_rev('v2.1'):
        six_glob = os.path.join(pyi_repo.local_path, 'PyInstaller', 'lib', 'six.*')
        for six_file in glob.glob(six_glob):
            dry('rm %s' % six_file, os.remove, six_file)

    # if the InVEST built binary directory exists, it should always
    # be deleted.  This is because we've had some weird issues with builds
    # not working properly when we don't just do a clean rebuild.
    invest_dist_dir = os.path.join('pyinstaller', 'dist', 'invest_dist')
    if os.path.exists(invest_dist_dir):
        dry('rm -r %s' % invest_dist_dir,
            shutil.rmtree, invest_dist_dir)

    pyinstaller_file = os.path.join('..', 'src', 'pyinstaller', 'pyinstaller.py')

    python_exe = os.path.abspath(options.build_bin.python)

    # For some reason, pyinstaller doesn't locate the natcap.versioner package
    # when it's installed and available on the system.  Placing
    # natcap.versioner's .egg in the pyinstaller eggs/ directory allows
    # natcap.versioner to be located.  Hacky but it works.
    # Assume we're working within the built virtualenv.
    sitepkgs = sh('{python} -c "import distutils.sysconfig; '
                  'print distutils.sysconfig.get_python_lib()"'.format(
                      python=python_exe), capture=True).rstrip()
    pathsep = ';' if platform.system() == 'Windows' else ':'

    # env_site_pkgs should be relative to the repo root
    env_site_pkgs = os.path.abspath(
        os.path.normpath(os.path.join(options.env.envname, 'lib')))
    if platform.system() != 'Windows':
        env_site_pkgs = os.path.join(env_site_pkgs, 'python2.7')
    env_site_pkgs = os.path.join(env_site_pkgs, 'site-packages')
    try:
        print "PYTHONPATH: %s" % os.environ['PYTHONPATH']
    except KeyError:
        print "Nothing in 'PYTHONPATH'"
    sh('%(python)s %(pyinstaller)s --clean --noconfirm --paths=%(paths)s invest.spec' % {
        'python': python_exe,
        'pyinstaller': pyinstaller_file,
        'paths': env_site_pkgs,
    }, cwd='exe')

    bindir = os.path.join('exe', 'dist', 'invest_dist')

    # Write the package versions to a text file for the record.
    # Assume we're in a virtualenv
    pip_bin = os.path.join(os.path.dirname(python_exe), 'pip')
    sh('{pip} freeze > package_versions.txt'.format(pip=pip_bin), cwd=bindir)

    # Record the hg path, branch, sha1 of this repo to a text file. This will help us down
    # the road to differentiate between built binaries from different forks.
    with open(os.path.join(bindir, 'buildinfo.txt'), 'w') as buildinfo_textfile:
        hg_path = sh('hg paths', capture=True)
        buildinfo_textfile.write(hg_path)

        branchname = sh('hg branch', capture=True)
        buildinfo_textfile.write('branch = %s' % branchname)

        commit_sha1 = sh('hg log -r . --template="{node}\n"', capture=True)
        buildinfo_textfile.write(commit_sha1)

    # If we're on windows, set the CLI to have slightly different default
    # behavior when the binary is clicked.  In this case, the CLI should prompt
    # for the user to define which model they would like to run.
    if platform.system() == 'Windows':
        iui_dir = os.path.join(bindir, 'natcap', 'invest', 'iui')
        with open(os.path.join(iui_dir, 'cli_config.json'), 'w') as json_file:
            json.dump({'prompt_on_empty_input': True}, json_file)

    if not os.path.exists('dist'):
        dry('mkdir dist',
            os.makedirs, 'dist')

    invest_dist = os.path.join('dist', 'invest_dist')
    if os.path.exists(invest_dist):
        dry('rm -r %s' % invest_dist,
            shutil.rmtree, invest_dist)

    dry('cp -r %s %s' % (bindir, invest_dist),
        shutil.copytree, bindir, invest_dist)

    # Mac builds seem to need an egg placed in just the right place.
    if platform.system() in ['Darwin', 'Linux']:
        sitepkgs_egg_glob = os.path.join(sitepkgs, 'natcap.versioner-*.egg')
        try:
            # If natcap.versioner was installed as an egg, just take that and
            # put it into the eggs/ dir.
            latest_egg = sorted(glob.glob(sitepkgs_egg_glob), reverse=True)[0]
            egg_dir = os.path.join(invest_dist, 'eggs')
            if not os.path.exists(egg_dir):
                dry('mkdir %s' % egg_dir, os.makedirs, egg_dir)

            dest_egg = os.path.join(invest_dist, 'eggs', os.path.basename(latest_egg))
            dry('cp {src_egg} {dest_egg}'.format(
                src_egg=latest_egg, dest_egg=dest_egg), shutil.copyfile,
                latest_egg, dest_egg)
        except IndexError:
            # Couldn't find any eggs in the local site-packages, use pip to
            # download the source archive, then build and copy the egg from the
            # archive.

            # Get version spec from requirements.txt
            with open('requirements.txt') as requirements_file:
                for requirement in pkg_resources.parse_requirements(requirements_file.read()):
                    if requirement.project_name == 'natcap.versioner':
                        versioner_spec = str(requirement)
                        break

            # Download a valid source tarball to the dist dir.

            sh('{pip_ep} install --no-deps --no-use-wheel --download {distdir} \'{versioner}\''.format(
                pip_ep=os.path.join(os.path.dirname(python_exe), 'pip'),
                distdir='dist',
                versioner=versioner_spec
            ))

            cwd = os.getcwd()
            # Unzip the tar.gz and run bdist_egg on it.
            versioner_tgz = os.path.abspath(
                glob.glob('dist/natcap.versioner-*.tar.gz')[0])
            os.chdir('dist')
            dry('unzip %s' % versioner_tgz,
                lambda tgz: tarfile.open(tgz, 'r:gz').extractall('.'),
                versioner_tgz)
            os.chdir(cwd)

            versioner_dir = versioner_tgz.replace('.tar.gz', '')
            sh('python setup.py bdist_egg', cwd=versioner_dir)

            # Copy the new egg to the built distribution with the eggs in it.
            # Both these folders should already be absolute paths.
            versioner_egg = glob.glob(os.path.join(versioner_dir, 'dist',
                                                   'natcap.versioner-*'))[0]
            egg_dirname = os.path.join(invest_dist, 'eggs')
            versioner_egg_dest = os.path.join(egg_dirname,
                                              os.path.basename(versioner_egg))
            if not os.path.exists(egg_dirname):
                os.makedirs(egg_dirname)
            dry('cp %s %s' % (versioner_egg, versioner_egg_dest),
                shutil.copyfile, versioner_egg, versioner_egg_dest)

    if platform.system() == 'Windows':
        binary = os.path.join(invest_dist, 'invest.exe')
        _write_console_files(binary, 'bat')
    else:
        binary = os.path.join(invest_dist, 'invest')
        _write_console_files(binary, 'sh')


@task
@might_call('build_bin')
@might_call('fetch')
@cmdopts([
    ('bindir=', 'b', ('Folder of binaries to include in the installer. '
                      'Defaults to dist/invest-bin')),
    ('insttype=', 'i', ('The type of installer to build. '
                        'Defaults depend on the current system: '
                        'Windows=nsis, Mac=dmg, Linux=deb')),
    ('arch=', 'a', 'The architecture of the binaries'),
    ('force-dev', '', 'Allow a build when a repo version differs from tracked versions'),
], share_with=['check_repo'])
def build_installer(options):
    """
    Build an installer for the target OS/platform.
    """

    if not os.path.exists(options.build_installer.bindir):
        raise BuildFailure(('WARNING: Binary dir %s not found.'
                           'Run `paver build_bin`' % options.bindir))

    # version comes from the installed version of natcap.invest
    invest_bin = os.path.join(options.build_installer.bindir, 'invest')
    version_string = sh('{invest_bin} --version'.format(invest_bin=invest_bin), capture=True)
    for possible_version in version_string.split('\n'):
        if possible_version != '':
            version = possible_version

    command = options.insttype.lower()
    if command == 'nsis':
        call_task('check_repo', options={
            'force_dev': options.build_installer.force_dev,
            'repo': REPOS_DICT['invest-2'].local_path,
            'fetch': True,
        })
        _build_nsis(version, options.build_installer.bindir, 'x86')
    elif command == 'dmg':
        _build_dmg(version, options.build_installer.bindir)
    elif command == 'deb':
        _build_fpm(version, options.build_installer.bindir, 'deb')
    elif command == 'rpm':
        _build_fpm(version, options.build_installer.bindir, 'rpm')
    else:
        raise BuildFailure('ERROR: build type not recognized: %s' % command)


def _build_fpm(version, bindir, pkg_type):
    print "WARNING:  Building linux packages is not yet fully supported"
    print "WARNING:  The package will build but won't yet install properly"
    print

    # debian packages dont like it when versions don't start with digits
    if version.startswith('null'):
        version = version.replace('null', '0.0.0')

    # copy the bindir into a properly named folder here.
    new_bindir = 'invest-bin'
    if os.path.exists(new_bindir):
        sh('rm -r %s' % new_bindir)
    sh('cp -r %s %s' % (bindir, new_bindir))

    options = {
        'pkg_type': pkg_type,
        'version': version,
        'bindir': new_bindir,
    }

    fpm_command = (
        'fpm -s dir -t %(pkg_type)s'
        ' -n invest'    # deb packages don't do well with uppercase
        ' -v %(version)s'
        ' -p dist/'
        ' --prefix /usr/lib/natcap/invest'  # assume that other tools will go in natcap as well
        ' -m "James Douglass <jdouglass@stanford.edu>"'
        ' --url http://naturalcapitalproject.org'
        ' --vendor "Natural Capital Project"'
        ' --license "Modified BSD"'
        ' --provides "invest"'
        ' --description "InVEST family of ecosystem service analysis tools'
        '\n\n'
        'InVEST (Integrated Valuation of Ecosystem Services '
        'and Tradeoffs) is a family of tools for quantifying the values '
        'of natural capital in clear, credible, and practical ways. In '
        'promising a return (of societal benefits) on investments in '
        'nature, the scientific community needs to deliver knowledge and '
        'tools to quantify and forecast this return. InVEST enables '
        'decision-makers to quantify the importance of natural capital, '
        'to assess the tradeoffs associated with alternative choices, and '
        'to integrate conservation and human development.'
        '\n\n'
        'The Natural Capital Project is a collaboration between Stanford '
        'University Woods Institute for the Environment, the World Wildlife'
        ' Fund, The Nature Conservancy and the University of Minnesota '
        'Institute on the Environment."'
        ' --after-install ./installer/linux/postinstall.sh'
        ' --after-remove ./installer/linux/postremove.sh'
        ' %(bindir)s') % options
    sh(fpm_command)


def _build_nsis(version, bindir, arch):
    """
    Build an NSIS installer.

    The InVEST NSIS script *requires* the following conditions are met:
        * The User's guide has been built (paver build_docs)
        * The invest-2 repo has been cloned to src (paver fetch src/invest-natcap.default)

    If these two conditions have not been met, the installer will fail.
    """
    # determine makensis path
    possible_paths = [
        'C:\\Program Files\\NSIS\\makensis.exe',
        'C:\\Program Files (x86)\\NSIS\\makensis.exe',
    ]
    makensis = None
    for makensis_path in possible_paths:
        if os.path.exists(makensis_path):
            makensis = '"%s"' % makensis_path

    if makensis is None:
        raise BuildFailure("Can't find nsis in %s" % possible_paths)

    if platform.system() != 'Windows':
        makensis = 'wine "%s"' % makensis

    # copying the dist dir into the cwd, since that's where NSIS expects it
    # also, NSIS (and our shortcuts) care very much about the dirname.
    nsis_bindir = 'invest-3-x86'
    if os.path.exists(nsis_bindir):
        raise BuildFailure("ERROR: %s exists in CWD.  Remove it and re-run")
    dry('cp %s %s' % (bindir, nsis_bindir),
        shutil.copytree, bindir, nsis_bindir)

    # copy the InVEST icon from the installer dir into the bindir.
    invest_icon_src = os.path.join('installer', 'windows', 'InVEST-2.ico')
    invest_icon_dst = os.path.join(nsis_bindir, 'InVEST-2.ico')
    dry('cp %s %s' % (invest_icon_src, invest_icon_dst),
        shutil.copyfile, invest_icon_src, invest_icon_dst)

    nsis_bindir = nsis_bindir.replace('/', r'\\')

    if 'post' in version:
        short_version = 'develop'
    else:
        short_version = version

    hg_path = sh('hg paths', capture=True).rstrip()
    forkuser, forkreponame = hg_path.split('/')[-2:]
    if forkuser == 'natcap':
        data_location = 'invest-data'
        forkname = ''
    else:
        data_location = 'nightly-build/invest-forks/%s/data' % forkuser
        forkname = forkuser

    nsis_params = [
        '/DVERSION=%s' % version,
        '/DVERSION_DISK=%s' % version,
        '/DINVEST_3_FOLDER=%s' % nsis_bindir,
        '/DSHORT_VERSION=%s' % short_version,
        '/DARCHITECTURE=%s' % arch,
        '/DFORKNAME=%s' % forkname,
        '/DDATA_LOCATION=%s' % data_location,
        'invest_installer.nsi'
    ]
    makensis += ' ' + ' '.join(nsis_params)
    sh(makensis, cwd=os.path.join('installer', 'windows'))

    # copy the completd NSIS installer file into dist/
    for exe_file in glob.glob('installer/windows/*.exe'):
        dest_file = os.path.join('dist', os.path.basename(exe_file))
        dry('cp installer/windows/*.exe dist',
            shutil.copyfile, exe_file, dest_file)

    # clean up the bindir we copied into cwd.
    dry('rm -r %s' % nsis_bindir,
        shutil.rmtree, nsis_bindir)


def _build_dmg(version, bindir):
    bindir = os.path.abspath(bindir)
    sh('./build_dmg.sh %s %s' % (version, bindir), cwd='installer/darwin')
    sh('cp installer/darwin/InVEST*.dmg dist')


def _get_local_version():
    # determine the version string.
    # If this is an archive, build the version string from info in
    # .hg_archival.
    if os.path.exists('.hg_archival.txt'):
        repo_data = yaml.load_safe(open('.hg_archival.txt'))
    elif os.path.exists('.hg'):
        # we're in an hg repo, so we can just get the information.
        repo = HgRepository('.', '')
        latesttagdistance = repo._format_log('{latesttagdistance}')
        if latesttagdistance is None:
            # When there's never been a tag.
            latesttagdistance = repo._format_log('{rev}')
        repo_data = {
            'latesttag': repo._format_log('{latesttag}'),
            'latesttagdistance': latesttagdistance,
            'branch': repo._format_log('{branch}'),
            'short_node': repo._format_log('{shortest(node, 6)}'),
        }
    else:
        print 'ERROR: Not an hg repo, not an hg archive, cannot determine version.'
        return

    # null from loading tag from hg, None from yaml
    if repo_data['latesttag'] in ['null', None]:
        repo_data['latesttag'] = '0.0'

    if repo_data['latesttagdistance'] == 0:
        version = repo_data['latesttag']
    else:
        version = "%(latesttag)s.dev%(latesttagdistance)s-%(short_node)s" % repo_data
    return version


def _write_console_files(binary, mode):
    """
    Write simple console files, one for each model presented by IUI.

    Parameters:
        binary (string): The path to the invest binary.
        mode (string): one of ["bat", "sh"]

    Returns:
        Nothing.
        Writes console files in the same directory as the binary.  Consoles
        are named according to "invest_<modelname>.<extension>"
    """

    windows_template = """
.\{binary} {modelname}
"""
    posix_template = """
./{binary} {modelname}
"""

    templates = {
        'bat': windows_template,
        'sh': posix_template,
    }
    filename_template = "{prefix}{modelname}.{extension}"

    exclude_prefix = set([
        'delineateit',
        'routedem',
    ])

    bindir = os.path.dirname(binary)
    for line in sh('{bin} --list'.format(bin=binary), capture=True).split('\n'):
        if line.startswith('    '):
            model_name = line.replace('UNSTABLE', '').lstrip().rstrip()

            if model_name not in exclude_prefix:
                prefix = 'invest_'
            else:
                prefix = ''

            console_filename = os.path.join(bindir, filename_template).format(
                modelname=model_name, extension=mode, prefix=prefix)
            print 'Writing console %s' % console_filename

            with open(console_filename, 'w') as console_file:
                formatted_template = templates[mode].format(
                    binary=os.path.basename(binary),
                    modelname=model_name)
                console_file.write(formatted_template)

            # Add executable bit if we're on linux or mac.
            if mode == 'sh':
                os.chmod(console_filename, 0744)


@task
def selftest():
    """
    Do a dry-run on all tasks found in this pavement file.
    """
    module = imp.load_source('pavement', __file__)

    def istask(reference):
        return isinstance(reference, paver.tasks.Task)

    for taskname, _ in inspect.getmembers(module, istask):
        if taskname != 'selftest':
            subprocess.call(['paver', '--dry-run', taskname])


@task
@cmdopts([
    ('force-dev', '', 'Allow development versions of repositories to be used.'),
    ('skip-data', '', "Don't build the data zipfiles"),
    ('skip-installer', '', "Don't build the installer"),
    ('skip-python', '', "Don't build python binaries"),
    ('skip-bin', '', "Don't build the binaries"),
    ('envname=', 'e', ('The name of the environment to use')),
    ('python=', '', "The python interpreter to use.  If not provided, an env will be built for you."),
], share_with=['build_docs', 'build_installer', 'build_bin', 'collect_release_files', 'check_repo'])
@might_call('check')
@might_call('env')
@might_call('build_data')
@might_call('build_docs')
@might_call('build_installer')
@might_call('build_bin')
@might_call('collect_release_files')
def build(options):
    """
    Build the installer, start-to-finish.  Includes binaries, docs, data, installer.

    If no extra options are specified, docs, data and binaries will all be generated.
    Any missing and needed repositories will be cloned.
    """

    # Allowing errors will still print them, just not fail the build.  It's
    # possible that the user might not want to build all available components.
    call_task('check', options={
        'fix_namespace': False,
        'allow_errors': True
    })

    # Check repositories up front so we can fail early if needed.
    # Here, we're only checking that if a repo exists, not cloning it.
    # The appropriate tasks will clone the repos they need.
    for repo, taskname, skip_condition in [
            (REPOS_DICT['users-guide'], 'build_docs', 'skip_guide'),
            (REPOS_DICT['invest-data'], 'build', 'skip_data'),
            (REPOS_DICT['invest-2'], 'build', 'skip_installer'),
            (REPOS_DICT['pyinstaller'], 'build', 'skip_bin')]:
        tracked_rev = repo.tracked_version()

        # Options are shared between several tasks, so we need to be sure that
        # the setting is bing fetched from the correct set of options.
        task_options = getattr(options, taskname)
        if not getattr(task_options, skip_condition):
            call_task('check_repo', options={
                'force_dev': options.build.force_dev,
                'repo': repo.local_path,
                'fetch': False,
            })
        print 'Repo %s is expected to be at rev %s' % (repo.local_path,
                                                       tracked_rev)

    call_task('clean', options=options)

    # build the env with our custom args for this context, but only if the user
    # has not already specified a python interpreter to use.
    if options.build.python == _PYTHON:
        call_task('env', options={
            'system_site_packages': True,
            'clear': True,
            'envname': options.build.envname,
            'with_invest': True,
            'with_pygeoprocessing': True,
            'requirements': '',
        })

    def _python():
        """
        Return the path to the environment's python exe.
        """
        if platform.system() == 'Windows':
            return os.path.join(options.build.envname, 'Scripts', 'python.exe')
        return os.path.join(options.build.envname, 'bin', 'python')

    if not options.build.skip_bin:
        call_task('build_bin', options={
            'python': _python(),
            'force_dev': options.build.force_dev,
        })
    else:
        print 'Skipping binaries per user request'

    if not options.build.skip_data:
        call_task('build_data', options=options.build_data)
    else:
        print 'Skipping data per user request'

    if (not options.build_docs.skip_api or
            not options.build_docs.skip_guide):
        call_task('build_docs', options={
            'skip_api': options.build_docs.skip_api,
            'skip_guide': options.build_docs.skip_guide,
            'python': _python(),
        })
    else:
        print 'Skipping documentation per user request'

    if not options.build.skip_python:
        # Wheel has an issue with namespace packages on windows.
        # See https://bitbucket.org/pypa/wheel/issues/91
        # I've implemented cgohlke's fix and pushed it to my fork of wheel.
        # To install a working wheel package, do this on your windows install:
        #   pip install hg+https://bitbucket.org/jdouglass/wheel@default
        #
        # This requires that you have command-line hg installed.
        if platform.system() == 'Windows':
            py_bin = 'bdist_wininst bdist_wheel'
        else:
            py_bin = 'bdist_wheel'

        # We always want to zip the sdist as a gztar because we said so.
        sh('{envpython} setup.py sdist --formats=gztar {py_bin}'.format(
            envpython=_python(), py_bin=py_bin))
    else:
        print 'Skipping python binaries per user request'

    if not options.build.skip_installer:
        call_task('build_installer', options=options.build_installer)
    else:
        print 'Skipping installer per user request'

    call_task('collect_release_files', options={
        'python': _python(),
    })


@task
@might_call('zip')
@cmdopts([
    ('python=', '', 'The python interpreter to use'),
])
def collect_release_files(options):
    """
    Collect release-specific files into a single distributable folder.
    """
    # make a distribution folder for this build version.
    # rstrip to take off the newline
    invest_version = _invest_version(options.collect_release_files.python)
    dist_dir = os.path.join('dist', 'release_%s' % invest_version)
    if not os.path.exists(dist_dir):
        dry('mkdir %s' % dist_dir, os.makedirs, dist_dir)

    # put the data zipfiles into a new folder.
    data_dir = os.path.join(dist_dir, 'data')
    if not os.path.exists(data_dir):
        dry('mkdir %s' % data_dir, os.makedirs, data_dir)

    for data_zip in glob.glob(os.path.join('dist', '*.zip')):
        if os.path.basename(data_zip).startswith('invest'):
            # Skip the api and userguide zipfiles
            continue

        out_filename = os.path.join(data_dir, os.path.basename(data_zip))
        dry('cp %s %s' % (data_zip, out_filename),
            shutil.copyfile, data_zip, out_filename)
        dry('rm %s' % out_filename,
            os.remove, data_zip)

    # copy the installer(s) into the new folder
    installer_files = []
    for pattern in ['*.exe', '*.dmg', '*.deb', '*.rpm', '*.zip', '*.whl',
                    '*.tar.gz']:
        glob_pattern = os.path.join('dist', pattern)
        installer_files += glob.glob(glob_pattern)

    for installer in installer_files:
        new_file = os.path.join(dist_dir, os.path.basename(installer))
        dry('cp %s %s' % (installer, new_file),
            shutil.copyfile, installer, new_file)
        dry('rm %s' % installer,
            os.remove, installer)

    # copy HTML documentation into the new folder.
    html_docs = os.path.join('doc', 'users-guide', 'build', 'html')
    out_dir = os.path.join(dist_dir, 'documentation')
    if os.path.exists(html_docs):
        if os.path.exists(out_dir):
            dry('rm -r %s' % out_dir,
                shutil.rmtree, out_dir)
        dry('cp -r %s %s' % (html_docs, out_dir),
            shutil.copytree, html_docs, out_dir)

    else:
        print "Skipping docs, since html docs were not built"

    # Copy PDF docs into the new folder
    try:
        pdf = glob.glob(os.path.join('doc', 'users-guide', 'build',
                                     'latex', '*.pdf'))[0]
    except IndexError:
        print "Skipping pdf, since pdf was not built."
    else:
        out_pdf = os.path.join(dist_dir, os.path.basename(pdf))
        out_pdf = out_pdf.replace('+VERSION+', invest_version)
        dry('cp %s %s' % (pdf, out_pdf),
            shutil.copyfile, pdf, out_pdf)

    # Archive the binaries dir.
    invest_dist = os.path.join('dist', 'invest_dist')
    if os.path.exists(invest_dist):
        os_name = platform.system().lower()
        architecture = 'x%s' % platform.architecture()[0][:2]
        zipfile_name = 'invest-{ver}-{plat}-{arch}'.format(
            ver=invest_version,
            plat=os_name,
            arch=architecture
        )
        call_task('zip', args=[
            os.path.join(dist_dir, zipfile_name),
            invest_dist
        ])


@task
@might_call('clean')
@might_call('build')
@might_call('jenkins_push_artifacts')
@cmdopts([
    ('nodata=', '', "Don't build the data zipfiles"),
    ('nobin=', '', "Don't build the binaries"),
    ('nodocs=', '', "Don't build the documentation"),
    ('noinstaller=', '', "Don't build the installer"),
    ('nopython=', '', "Don't build the various python installers"),
    ('nopush=', '', "Don't Push the build artifacts to dataportal"),
], share_with=['clean', 'build', 'jenkins_push_artifacts'])
def jenkins_installer(options):
    """
    Run a jenkins build via paver.

    Allows for the user to build only the pieces needed.  Especially handy for
    dev builds on a fork.

    All parameters passed in must be strings, either 'true' or 'false'.
    Empty values, '', "", 0, and various capitalizations of false will evaluate to False.
    Only 1 and various capitalizations of true will evaluate to True.
    An exception will be raised if any other value is provided.
    """

    # Process build options up front so that we can fail earlier.
    # Assume we're in a virtualenv.
    build_options = {}
    if platform.system() == 'Windows':
        # force building with msvc on jenkins on Windows
        build_options['compiler'] = 'msvc'

    for opt_name, build_opts, needed_repo in [
            ('nodata', ['skip_data'], 'data/invest-data'),
            ('nodocs', ['skip_guide', 'skip_api'], 'doc/users-guide'),
            ('noinstaller', ['skip_installer'], 'src/invest-natcap.default'),
            ('nopython', ['skip_python'], None),
            ('nobin', ['skip_bin'], 'src/pyinstaller')]:
        # set these options based on whether they were provided.
        try:
            user_option = getattr(options.jenkins_installer, opt_name)
            if user_option.lower() in ['true', '1']:
                user_option = True
            elif user_option.lower() in ['', "''", '""', 'false', '0']:
                # Skip this option entirely.  build() expects this option to be
                # absent from the build_options dict if we want to not provide
                # the build option.
                if needed_repo is not None:
                    call_task('check_repo', options={
                        'repo': needed_repo,
                        'fetch': True,
                    })

                raise AttributeError
            else:
                raise Exception('Invalid option: %s' % user_option)
            for build_opt in build_opts:
                build_options[build_opt] = user_option
        except AttributeError:
            print 'Skipping option %s' % opt_name
            pass

    call_task('clean', options=options)
    call_task('build', options=build_options)

    try:
        nopush_str = getattr(options.jenkins_installer, 'nopush')
        if nopush_str in ['false', 'False', '0', '', '""']:
            push = True
        else:
            push = False
    except AttributeError:
        push = True

    if push:
        python = os.path.join(
            options.env.envname,
            'Scripts' if platform.system() == 'Windows' else 'bin',
            'python')

        call_task('jenkins_push_artifacts', options={
            'python': python,
            'username': 'dataportal',
            'host': 'data.naturalcapitalproject.org',
            'dataportal': 'public_html',
            # Only push data zipfiles if we're on Windows.
            # Have to pick one, as we're having issues if all slaves are trying
            # to push the same large files.
            'include_data': platform.system() == 'Windows',
        })


@task
@consume_args
def zip(args):
    """
    Zip a folder and save it to an output zip file.

    Usage: paver zip archivename dirname

    Arguments:
        archivename - the filename of the output archive
        dirname - the name of the folder to archive.
        prefix - (optional) the directory to store files in.
    """

    if len(args) > 3:
        raise BuildFailure('zip takes <=3 arguments.')

    archive_name = args[0]
    source_dir = os.path.abspath(args[1])

    try:
        prefix = args[2]
        dest_dir = os.path.join(os.path.dirname(source_dir), prefix)
        if os.path.exists(dest_dir):
            dry('rm -r %s' % dest_dir,
                shutil.rmtree, dest_dir)
        dry('cp -r %s %s' % (source_dir, prefix),
            shutil.copytree, source_dir, dest_dir)
    except IndexError:
        prefix = os.path.basename(source_dir)

    dry('zip -r %s %s.zip' % (source_dir, archive_name),
        shutil.make_archive, **{
            'base_name': archive_name,
            'format': 'zip',
            'root_dir': os.path.dirname(source_dir),
            'base_dir': prefix})


@task
@cmdopts([
    ('attr-file=', 'u', 'Save path attributes to a file'),
])
def forked_by(options):
    """
    Print the name of the user who forked this repo.
    """

    hg_path = sh('hg paths', capture=True).rstrip()

    username, reponame = hg_path.split('/')[-2:]
    print 'username=%s' % username

    try:
        with open(options.uname_file, 'w') as username_file:
            username_file.write(username)
    except AttributeError:
        pass

@task
@consume_args
def compress_raster(args):
    """
    Compress a raster.

    Call `paver compress_raster --help` for full details.
    """
    parser = argparse.ArgumentParser(description=(
        'Compress a GDAL-compatible raster.'))
    parser.add_argument('-x', '--blockxsize', default=0, type=int, help=(
        'The block size along the X axis.  Default=inraster block'))
    parser.add_argument('-y', '--blockysize', default=0, type=int, help=(
        'The block size along the Y axis.  Default=inraster block'))
    parser.add_argument('-c', '--compression', default='LZW', type=str, help=(
        'Compress the raster.  Valid options: NONE, LZW, DEFLATE, PACKBITS. '
        'Default: LZW'))
    parser.add_argument('inraster', type=str, help=(
        'The raster to compress'))
    parser.add_argument('outraster', type=str, help=(
        'The path to the output raster'))

    parsed_args = parser.parse_args(args)

    # import GDAL here because I don't want it to be a requirement to be able
    # to run all paver functions.
    from osgeo import gdal
    in_raster = gdal.Open(parsed_args.inraster)
    in_band = in_raster.GetRasterBand(1)
    block_x, block_y = in_band.GetBlockSize()
    if parsed_args.blockxsize == 0:
        parsed_args.blockxsize = block_x
    block_x_opt = '-co "BLOCKXSIZE=%s"' % parsed_args.blockxsize

    if parsed_args.blockysize == 0:
        parsed_args.blockysize = block_y
    block_y_opt = '-co "BLOCKYSIZE=%s"' % parsed_args.blockysize

    if parsed_args.blockysize % 2 == 0 and parsed_args.blockysize % 2 == 0:
        tile_cmd = '-co "TILED=YES"'
    else:
        tile_cmd = ''

    sh(('gdal_translate '
        '-of "GTiff" '
        '{tile} '
        '{block_x} '
        '{block_y} '
        '-co "COMPRESS=LZW" '
        '{in_raster} {out_raster}').format(
            tile=tile_cmd,
            block_x=block_x_opt,
            block_y=block_y_opt,
            in_raster=os.path.abspath(parsed_args.inraster),
            out_raster=os.path.abspath(parsed_args.outraster),
        ))


@task
@consume_args
def test(args):
    """Run the suite of InVEST tests within a virtualenv.

    When run, paver will determine whether InVEST needs to be installed into
    the active virtualenv, creating the env if needed.  InVEST will be
    installed into the virtualenv if:

        * InVEST is not already installed into the virtualenv.
        * The version of InVEST installed into the virtualenv is older than
          what is currently available to be installed from the source tree.
        * There are uncommitted changes in the source tree.

    Default behavior is to run all tests contained in tests/*.py and
    src/natcap/invest/tests/*.py.

    If --jenkins is provided, xunit reports and extra logging will be produced.
    If --with-data is provided, test data repos will be cloned.

    --jenkins implies --with-data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--jenkins', default=False, action='store_true',
            help='Use options that are useful for Jenkins reports')
    parser.add_argument('--with-data', default=False, action='store_true',
            help='Clone/update the data repo if needed')
    parser.add_argument('nose_args', nargs='*',
                        help=('Nosetests-compatible strings indicating '
                              'filename[:classname[.testname]]'),
                        metavar='TEST')
    parsed_args = parser.parse_args(args)
    print 'parsed args: ', parsed_args

    if parsed_args.with_data:
        call_task('fetch', args=[REPOS_DICT['test-data'].local_path])
        call_task('fetch', args=[REPOS_DICT['invest-data'].local_path])

    @paver.virtual.virtualenv(paver.easy.options.dev_env.envname)
    def _run_tests():
        """
        Run tests within a virtualenv.  If we're running with the --jenkins
        flag, add a couple more options suitable for that environment.
        """
        if parsed_args.jenkins:
            call_task('check_repo', options={
                'repo': REPOS_DICT['test-data'].local_path,
                'fetch': True,
            })
            call_task('check_repo', options={
                'repo': REPOS_DICT['invest-data'].local_path,
                'fetch': True,
            })
            jenkins_flags = (
                '--with-xunit '
                '--with-coverage '
                '--cover-xml '
                '--cover-tests '
                '--logging-filter=None '
                '--nologcapture '
            )
        else:
            jenkins_flags = ''

        if len(parsed_args.nose_args) == 0:
            # Specifying all tests by hand here because Windows doesn't like the *
            # wildcard operator like Linux/Mac does.
            regression_tests = glob.glob(os.path.join('tests', '*.py'))
            _unit_glob = glob.glob(os.path.join('src', 'natcap', 'invest',
                                                'tests', '*.py'))
            unit_tests = [t for t in _unit_glob
                        if os.path.basename(t) != '__init__.py']
            tests = regression_tests + unit_tests
        else:
            # If the user gave us some test names to run, run those instead!
            tests = parsed_args.nose_args

        sh(('nosetests -vs {jenkins_opts} {tests}').format(
                jenkins_opts=jenkins_flags,
                tests=' '.join(tests)
            ))

    @paver.virtual.virtualenv(paver.easy.options.dev_env.envname)
    def _update_invest():
        """
        Determine if InVEST needs to be updated based on known version strings.
        If so, remove the existing installation of InVEST and reinstall.
        Runs within the virtualenv.
        """
        # If there are uncommitted changes, or the installed version of InVEST
        # differs from the local version, reinstall InVEST into the virtualenv.
        changed_files = sh((
            'hg status -a -m -r -d '
            'src/natcap/invest/ pavement.py setup.py setup.cfg MANIFEST.in'),
            capture=True)
        print 'Changed files: ' + changed_files
        changes_uncommitted = changed_files.strip() != ''
        if not changes_uncommitted:
            # If no uncommitted changes, check that the versions match.
            # If versions don't match, reinstall.
            try:
                installed_version = sh(('python -c "import natcap.invest;'
                                        'print natcap.invest.__version__"'),
                                        capture=True)
                local_version = sh('python setup.py --version', capture=True)
            except BuildFailure:
                # When natcap.invest is not installed, so force reinstall
                installed_version = False
                local_version = True
        else:
            # If changes are uncommitted, force reinstall.
            installed_version = True
            local_version = True

        if changes_uncommitted or (installed_version != local_version):
            try:
                sh('pip uninstall -y natcap.invest')
            except BuildFailure:
                pass
            sh('python setup.py install')

    # Build an env if needed.
    if not os.path.exists(paver.easy.options.dev_env.envname):
        call_task('dev_env')
    else:
        _update_invest()

    # run the tests within the virtualenv.
    _run_tests()

@task
@might_call('push')
@cmdopts([
    ('python=', '', 'Python exe'),
    ('username=', '', 'Remote username'),
    ('host=', '', 'URL of the remote server'),
    ('dataportal=', '', 'Path to the dataportal'),
    ('upstream=', '', 'The URL to the upstream REPO.  Use this when this repo is moved'),
    ('password', '', 'Prompt for a password'),
    ('private-key=', '', 'Use this private key to push'),
    ('include-data', '', 'Include data zipfiles in the push'),
])
def jenkins_push_artifacts(options):
    """
    Push artifacts to a remote server.
    """

    # get fork name
    try:
        hg_path = getattr(options.jenkins_push_artifacts, 'upstream')
    except AttributeError:
        hg_path = sh('hg paths', capture=True).rstrip()

    username, reponame = hg_path.split('/')[-2:]

    version_string = _invest_version(getattr(options.jenkins_push_artifacts, 'python', sys.executable))

    def _get_release_files():
        release_files = []
        for filename in glob.glob('dist/release_*/*'):
            if not os.path.isdir(filename):
                release_files.append(filename)
        return release_files

    release_files = _get_release_files()
    data_files = glob.glob('dist/release_*/data/*')
    if username == 'natcap' and reponame == 'invest':
        # We're not on a fork!  Binaries are pushed to invest-releases
        # dirnames are relative to the dataportal root
        if 'post' in version_string:
            data_dirname = 'develop'
        else:
            data_dirname = version_string
        data_dir = os.path.join('invest-data', data_dirname)
        release_dir = os.path.join('invest-releases', version_string)
    else:
        # We're on a fork!
        # Push the binaries, documentation to nightly-build
        release_dir = os.path.join('nightly-build', 'invest-forks', username)
        data_dir = os.path.join(release_dir, 'data')

    pkey = None
    if getattr(options.jenkins_push_artifacts, 'private_key', False):
        pkey = options.jenkins_push_artifacts.private_key
    elif platform.system() in ['Windows', 'Darwin', 'Linux']:
        # Assume a default private key location for jenkins builds
        # On Windows, this assumes that the key is in .ssh (might be the cygwin
        # home directory).
        pkey = os.path.join(os.path.expanduser('~'),
                            '.ssh', 'dataportal-id_rsa')
    else:
        print ('No private key provided, and not on a known system, so not '
               'assuming a default private key file')

    push_args = {
        'user': getattr(options.jenkins_push_artifacts, 'username'),
        'host': getattr(options.jenkins_push_artifacts, 'host'),
    }

    def _push(target_dir):
        """
        Format the push configuration string based on the given target_dir.
        """
        push_args['dir'] = os.path.join(
            getattr(options.jenkins_push_artifacts, 'dataportal'),
            target_dir)

        push_config = []
        if getattr(options.jenkins_push_artifacts, 'password', False):
            push_config.append('--password')

        push_config.append('--private-key=%s' % pkey)
        push_config.append('--makedirs')

        push_config.append('{user}@{host}:{dir}'.format(**push_args))
        return push_config

    if len(release_files) > 0:
        call_task('push', args=_push(release_dir) + release_files)

    try:
        include_data = options.jenkins_push_artifacts.include_data
    except AttributeError:
        include_data = False
    finally:
        if len(data_files) == 0:
            print 'No data files to push.'
        elif not include_data:
            print 'Excluding data files from push per user preference'
        else:
            call_task('push', args=_push(data_dir) + data_files)

    def _archive_present(substring):
        """
        Is there a file in release_files that ends in `substring`?
        Returns a boolean.
        """
        archive_present = reduce(
            lambda x, y: x or y,
            [x.endswith(substring) for x in release_files])
        return archive_present

    zips_to_unzip = []
    if not _archive_present('apidocs.zip'):
        print 'API documentation was not built.'
    else:
        zips_to_unzip.append('*apidocs.zip')

    if not _archive_present('userguide.zip'):
        print 'User guide was not built'
    else:
        zips_to_unzip.append('*userguide.zip')

    if len(zips_to_unzip) == 0:
        print 'Nothing to unzip on the remote.  Skipping.'
        return

    # unzip the API docs and HTML documentation.  This will overwrite anything
    # else in the release dir.
    import paramiko
    from paramiko import SSHClient
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if pkey is not None:
        pkey = paramiko.RSAKey.from_private_key_file(pkey)

    print 'Connecting to host'
    ssh.connect(push_args['host'], 22, username=push_args['user'], password=None, pkey=pkey)

    # correct the filepath from Windows to Linux
    if platform.system() == 'Windows':
        release_dir = release_dir.replace(os.sep, '/')

    if release_dir.startswith('public_html/'):
        release_dir = release_dir.replace('public_html/', '')

    for filename in zips_to_unzip:
        print 'Unzipping %s on remote' % filename
        _, stdout, stderr = ssh.exec_command(
            'cd public_html/{releasedir}; unzip -o `ls -tr {zipfile} | tail -n 1`'.format(
                releasedir=release_dir,
                zipfile=filename
            )
        )

        print "STDOUT:"
        for line in stdout:
            print line

        print "STDERR:"
        for line in stderr:
            print line

    ssh.close()
