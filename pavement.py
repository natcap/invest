import os
import logging
import sys
import json
import platform
import collections
import getpass
import shutil
import warnings
import zipfile
import glob
import textwrap
import imp
import subprocess
import inspect

import paver.svn
import paver.path
from paver.easy import *
import virtualenv
import yaml

LOGGER = logging.getLogger('invest-bin')
_SDTOUT_HANDLER = logging.StreamHandler(sys.stdout)
_SDTOUT_HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(_SDTOUT_HANDLER)


class Repository(object):
    tip = ''
    statedir = ''
    cmd = ''

    def __init__(self, local_path, remote_url):
        self.local_path = local_path
        self.remote_url = remote_url

    def ischeckedout(self):
        return os.path.exists(os.path.join(self.local_path, self.statedir))

    def clone(self):
        raise Exception

    def pull(self):
        raise Exception

    def update(self):
        raise Exception

    def tracked_version(self):
        return json.load(open('versions.json'))[self.local_path]

    def at_known_rev(self):
        tracked_version = self.format_rev(self.tracked_version())
        return self.current_rev() == tracked_version

    def format_rev(self, rev):
        raise Exception

    def current_rev(self):
        raise Exception


class HgRepository(Repository):
    tip = 'tip'
    statedir = '.hg'
    cmd = 'hg'

    def clone(self, rev=None):
        if rev is None:
            rev = self.tracked_version()
        sh('hg clone %(url)s %(dest)s -u %(rev)s' % {'url': self.remote_url,
                                                     'dest': self.local_path,
                                                     'rev': rev})

    def pull(self):
        sh('hg pull -R %(dest)s' % {'dest': self.local_path})

    def update(self, rev):
        sh('hg update -R %(dest)s -r %(rev)s' % {'dest': self.local_path,
                                                 'rev': rev})

    def _format_log(self, template='', rev='.'):
        return sh('hg log -R %(dest)s -r %(rev)s --template="%(template)s"' % {
            'dest': self.local_path, 'rev': rev, 'template': template},
            capture=True).rstrip()

    def format_rev(self, rev):
        return self._format_log('{node}', rev=rev)

    def current_rev(self):
        return self._format_log('{node}')

    def tracked_version(self):
        json_version = Repository.tracked_version(self)
        return self._format_log(template='{node}', rev=json_version)


class SVNRepository(Repository):
    tip = 'HEAD'
    statedir = '.svn'
    cmd = 'svn'

    def clone(self, rev=None):
        if rev is None:
            rev = self.tracked_version()
        paver.svn.checkout(self.remote_url, self.local_path, revision=rev)

    def pull(self):
        # svn is centralized, so there's no concept of pull without a checkout.
        return

    def update(self, rev):
        paver.svn.update(self.local_path, rev)

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

class GitRepository(Repository):
    tip = 'master'
    statedir = '.git'
    cmd = 'git'

    def clone(self, rev=None):
        sh('git clone %(url)s %(dest)s' % {'url': self.remote_url,
                                              'dest': self.local_path})
        if rev is None:
            rev = self.tracked_version()
            self.update(rev)

    def pull(self):
        sh('git fetch', cwd=self.local_path)

    def update(self, rev):
        sh('git checkout %(rev)s' % {'rev': rev}, cwd=self.local_path)

    def current_rev(self):
        return sh('git rev-parse --verify HEAD', cwd=self.local_path,

    def format_rev(self, rev):
        return sh('git log --format=format:%H -1 %(rev)s' % {'rev': rev},
                  capture=True, cwd=self.local_path)

REPOS_DICT = {
    'users-guide': HgRepository('doc/users-guide', 'https://bitbucket.org/natcap/invest.users-guide'),
    'pygeoprocessing': HgRepository('src/pygeoprocessing', 'https://bitbucket.org/richpsharp/pygeoprocessing'),
    'invest-data': SVNRepository('data/invest-data', 'svn://scm.naturalcapitalproject.org/svn/invest-sample-data'),
    'invest-2': HgRepository('src/invest-natcap.default', 'http://bitbucket.org/natcap/invest.arcgis'),
    'pyinstaller': GitRepository('src/pyinstaller', 'https://github.com/pyinstaller/pyinstaller.git'),
}
REPOS = REPOS_DICT.values()


def _invest_version():
    """
    Load the InVEST version string and return it.

    Fetches the string from natcap.invest if the package is installed and
    is able to be imported.  Otherwise, fetches the version string from
    the natcap.invest source.

    Returns:
        The version string.
    """
    try:
        import natcap.invest as invest
    except ImportError:
        invest = imp.load_source('_invest', 'src/natcap/invest/__init__.py')
    return invest.__version__

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

    if not repo.at_known_rev() and options.force_dev is False:
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
    try:
        if options.json and options.save:
            print "ERROR: --json and --save are mutually exclusive"
            return
    except AttributeError:
        pass

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
                if at_known_rev is False:
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

# options are accessed by virtualenv bootstrap command somehow.
options(
    virtualenv=Bunch(
        dest_dir='test_env',
        script_name="bootstrap.py"
    )
)


@task
@cmdopts([
    ('system-site-packages', '', ('Give the virtual environment access '
                                  'to the global site-packages')),
    ('clear', '', 'Clear out the non-root install and start from scratch.'),
    ('envname=', 'e', ('The name of the environment to use')),
    ('with-invest', '', 'Install the current version of InVEST into the env'),
    ('requirements=', 'r', 'Install requirements from a file'),
])
def env(options):
    """
    Set up a virtualenv for the project.
    """

    # Detect whether the user called `paver env` with the system-site-packages
    # flag.  If so, modify the paver options object so that bootstrapping will
    # use the virtualenv WITH the system-site-packages linked in.
    try:
        use_site_pkgs = options.env.system_site_packages
    except AttributeError:
        use_site_pkgs = False
    options.virtualenv.system_site_packages = use_site_pkgs

    # check whether the user wants to use a clean environment.
    # Assume False if not provided.
    try:
        options.env.clear
    except AttributeError:
        options.env.clear = False

    try:
        options.virtualenv.dest_dir = options.envname
        print "Using user-defined env name: %s" % options.envname
    except AttributeError:
        print "Using the default envname: %s" % options.virtualenv.dest_dir

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
    else:
        bindir = 'bin'

    """

    requirements_files = ['requirements.txt']
    extra_reqs = getattr(options, 'requirements', None)
    if extra_reqs is not None:
        requirements_files.append(extra_reqs)

    pip_template = "    subprocess.call([join(home_dir, bindir, 'pip'), 'install', '%s'])\n"
    for reqs_file in requirements_files:
        for pkgname in open(reqs_file).read().rstrip().split('\n'):
            install_string += pip_template % pkgname
    try:
        if options.with_invest is True:
            install_string += "    subprocess.call([join(home_dir, bindir, 'python'), 'setup.py', 'install'])\n"
    except AttributeError:
        print "Skipping installation of natcap.invest"

    output = virtualenv.create_bootstrap_script(textwrap.dedent(install_string))
    open(options.virtualenv.script_name, 'w').write(output)

    # Built the bootstrap env via a subprocess call.
    # Calling via the shell so that virtualenv has access to environment
    # vars as needed.
    env_dirname = options.virtualenv.dest_dir
    bootstrap_cmd = "%(python)s %(bootstrap_file)s %(site-pkgs)s %(clear)s %(env_name)s"
    bootstrap_opts = {
        "python": sys.executable,
        "bootstrap_file": options.virtualenv.script_name,
        "env_name": env_dirname,
        "site-pkgs": '--system-site-packages' if use_site_pkgs else '',
        "clear": '--clear' if options.env.clear else '',
    }
    sh(bootstrap_cmd % bootstrap_opts)


    print '*** Virtual environment created successfully.'
    print '*** To activate the env, run:'
    if platform.system() == 'Windows':
        print r'    call .\%s\Scripts\activate' % env_dirname
    else:  # assume all POSIX systems behave the same way
        print '    source %s/bin/activate' % env_dirname


@task
@consume_args  # when consuuming args, it's a list of str arguments.
def fetch(args=None):
    """
    Clone repositories the correct locations.
    """

    # figure out which repos/revs we're hoping to update.
    # None is our internal, temp keyword representing the LATEST possible
    # rev.
    user_repo_revs = {}  # repo -> version
    repo_paths = map(lambda x: x.local_path, REPOS)
    if args is None:
        args = []
    args_queue = collections.deque(args[:])

    while len(args_queue) > 0:
        current_arg = args_queue.popleft()

        # If the user provides repo revisions, it MUST be a specific repo.
        if current_arg in repo_paths:
            # the user might provide a revision.
            # It's a rev if it's not a repo.
            try:
                possible_rev = args_queue.popleft()
            except IndexError:
                # When no other args after the repo
                user_repo_revs[current_arg] = None
                continue

            if possible_rev in repo_paths:
                # then it's not a revision, it's a repo.  put it back.
                # Also, assume user wants the repo we're currently working with
                # to be updated to the tip OR whatever.
                user_repo_revs[current_arg] = None
                args_queue.appendleft(possible_rev)
                continue
            elif possible_rev in ['-r', '--rev']:
                requested_rev = args_queue.popleft()
                user_repo_revs[current_arg] = requested_rev
            else:
                print "ERROR: unclear arg %s" % possible_rev
                return

    # determine which groupings the user wants to operate on.
    # example: `src` would represent all repos under src/
    # example: `data` would represent all repos under data/
    # example: `src/pygeoprocessing` would represent the pygeoprocessing repo
    repos = set([])
    for argument in args:
        if not argument.startswith('-'):
            repos.add(argument)

    def _user_requested_repo(local_repo_path):
        """
        Check if the user requested this repository.
        Does so by checking prefixes provided by the user.

        Arguments:
            local_repo_path (string): the path to the local repository
                relative to the CWD. (example: src/pygeoprocessing)

        Returns:
            Boolean: Whether the user did request this repo.
        """
        # check that the user wants to update this repo
        for user_arg_prefix in repos:
            if local_repo_path.startswith(user_arg_prefix):
                return True
        return False

    for repo in REPOS:
        LOGGER.debug('Checking %s', repo.local_path)

        # If the user did not request this repo AND the user didn't want to
        # update everything (by specifying no positional args), skip this repo.
        if not _user_requested_repo(repo.local_path) and len(repos) > 0:
            continue

        # does repo exist?  If not, clone it.
        if not repo.ischeckedout():
            repo.clone()
        else:
            LOGGER.debug('Repository %s exists', repo.local_path)

        # is repo up-to-date?  If not, update it.
        # If the user specified a target revision, use that instead.
        try:
            target_rev = user_repo_revs[repo.local_path]
            if target_rev is None:
                raise KeyError
        except KeyError:
            try:
                target_rev = repo.tracked_version()
            except KeyError:
                print 'WARNING: repo not tracked in versions.json: %s' % repo.local_path
                return 1

        repo.pull()
        repo.update(target_rev)


@task
@consume_args
def push(args):
    """Push a file or files to a remote server.

    Usage:
        paver push [--password] [user@]hostname[:target_dir] file1, file2, ...

    Uses pythonic paramiko-based SCP to copy files to the remote server.

    If --password is provided at the command line, the user will be prompted
    for a password.  This is sometimes required when the remote's private key
    requires a password to decrypt.

    If a target username is not provided ([user@]...), the current user's username
    used for the transfer.

    If a target directory is not provided (hostname[:target_dir]), the current
    directory of the target user is used.
    """
    import paramiko
    from paramiko import SSHClient
    from scp import SCPClient
    ssh = SSHClient()
    ssh.load_system_host_keys()

    # Clean out all of the user-configurable options flags.
    config_opts = []
    for argument in args:
        if argument.startswith('--'):
            config_opts.append(argument)
            args.remove(argument)

    use_password = '--password' in config_opts

    try:
        destination_config = args[0]
    except IndexError:
        print "ERROR: destination config must be provided"
        return

    files_to_push = args[1:]
    if len(files_to_push) == 0:
        print "ERROR: At least one file must be given"
        return

    # ASSUME WE'RE ONLY DOING ONE HOST PER PUSH
    # split apart the configuration string.
    # format:
    #    [user@]hostname[:directory]
    if '@' in destination_config:
        username = destination_config.split('@')[0]
        destination_config = destination_config.replace(username + '@', '')
    else:
        username = getpass.getuser()

    if ':' in destination_config:
        target_dir = destination_config.split(':')[-1]
        destination_config = destination_config.replace(':' + target_dir, '')
    else:
        # just use the SCP default
        target_dir = None

    # hostname is whatever remains of the dest config.
    hostname = destination_config

    # start up the SSH connection
    if use_password:
        password = getpass.getpass()
    else:
        password = None

    try:
        ssh.connect(hostname, username=username, password=password)
    except paramiko.BadAuthenticationType:
        print 'ERROR: incorrect password or bad SSH key.'
        return
    except paramiko.PasswordRequiredException:
        print 'ERROR: password required to decrypt private key on remote.  Use --password flag'
        return

    scp = SCPClient(ssh.get_transport())
    for transfer_file in files_to_push:
        file_basename = os.path.basename(transfer_file)
        if target_dir is not None:
            target_filename = os.path.join(target_dir, file_basename)
        else:
            target_filename = file_basename

        print 'Transferring %s -> %s:%s ' % (transfer_file, hostname, target_filename)
        scp.put(transfer_file, target_filename)


@task
def clean(options):
    """
    Remove files and folders known to be generated by build scripts.
    """

    folders_to_rm = ['build', 'dist', 'tmp', 'bin', 'test',
                     options.virtualenv.dest_dir,
                     'installer/darwin/temp',
                     'api_env',
                     'exe/dist',
                     'exe/build',
                     ]
    files_to_rm = [
        options.virtualenv.script_name,
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
@cmdopts([
    ('force-dev', '', 'Force development'),
    ('skip-api', '', 'Skip building the API docs'),
    ('skip-guide', '', "Skip building the User's Guide"),
])
def build_docs(options):
    """
    Build the sphinx user's guide for InVEST.

    Builds the sphinx user's guide in HTML, latex and PDF formats.
    Compilation of the guides uses sphinx and requires that all needed
    libraries are installed for compiling html, latex and pdf.

    Requires make and sed.
    """


    invest_version = sh('python setup.py --version', capture=True).rstrip()
    archive_template = os.path.join('dist', 'invest-%s-%s' % (invest_version, '%s'))

    # If the user has not provided the skip-guide flag, build the User's guide.
    skip_guide = getattr(options, 'skip_guide', False)
    if skip_guide is False:
        if not _repo_is_valid(REPOS_DICT['users-guide'], options):
            raise BuildFailure('User guide version is out of sync and force-dev not provided')
        guide_dir = os.path.join('doc', 'users-guide')
        latex_dir = os.path.join(guide_dir, 'build', 'latex')
        sh('make html', cwd=guide_dir)
        sh('make latex', cwd=guide_dir)
        sh('make all-pdf', cwd=latex_dir)

        archive_name = archive_template % 'userguide'
        build_dir = os.path.join(guide_dir, 'build', 'html')
        call_task('zip', args=[archive_name, build_dir])
    else:
        print "Skipping the User's Guide"

    skip_api = getattr(options, 'skip_api', False)
    api_env = os.path.join(os.getcwd(), 'api_env')
    if skip_api is False:
    #original state in case they were modified in a previous run
    for file in ['conf.py', 'index.rst', 'carbonstorage.rst',
                    'managed_timber_production_model.rst']:
        sh('./jenkins/api-docs.sh -e %s' % api_env)
        archive_name = archive_template % 'apidocs'
        sh("sed -i -e 's/+VERSION+/" + version + "/g' ./%s" % file, cwd=source_dir)

        call_task('zip', args=[archive_name, 'build/sphinx/html'])
    else:
        print "Skipping the API docs"


@task
def check():
    """
    Perform reasonable checks to verify the build environment.


    This paver task checks that the following is true:
        Executables are available: hg, git


    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    class FoundEXE(Exception):
        pass

    # verify required programs exist
    errors_found = False
    for program in ['hg', 'git', 'make']:
        # Inspired by this SO post: http://stackoverflow.com/a/855764/299084

        fpath, fname = os.path.split(program)
        if fpath:
            if not is_exe(program):
                print "ERROR: executable not found: %s" % program
                errors_found = True
        else:
            try:
                for path in os.environ["PATH"].split(os.pathsep):
                    path = path.strip('"')
                    exe_file = os.path.join(path, program)
                    if is_exe(exe_file):
                        raise FoundEXE
            except FoundEXE:
                continue
            else:
                print "ERROR: executable %s not found on the PATH" % fname
                errors_found = True

    if errors_found:
        return 1
    else:
        print "All's well."


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
    if not _repo_is_valid(data_repo, options):
        return

    dist_dir = 'dist'
    if not os.path.exists(dist_dir):
        dry('mkdir %s' % dist_dir, os.makedirs, dist_dir)

    for data_dirname in os.listdir(data_repo.local_path):
        out_zipfile = os.path.abspath(os.path.join(dist_dir, data_dirname + ".zip"))
        if not os.path.isdir(os.path.join(data_repo.local_path, data_dirname)):
            continue
        if data_dirname == data_repo.statedir:
            continue

        dry('zip -r %s %s' % (out_zipfile, data_dirname),
            shutil.make_archive, **{
                'base_name': os.path.splitext(out_zipfile)[0],
                'format': 'zip',
                'root_dir': os.path.join(data_repo.local_path, data_dirname),
                'base_dir': '.'})


@task
def build_bin():
    """
    Build frozen binaries of InVEST.
    """
    # if the InVEST built binary directory exists, it should always
    # be deleted.  This is because we've had some weird issues with builds
    # not working properly when we don't just do a clean rebuild.
    invest_dist_dir = os.path.join('pyinstaller', 'dist', 'invest_dist')
    if os.path.exists(invest_dist_dir):
        dry('rm -r %s' % invest_dist_dir,
            shutil.rmtree, invest_dist_dir)

    pyinstaller_file = os.path.join('..', 'src', 'pyinstaller', 'pyinstaller.py')
    sh('%(python)s %(pyinstaller)s --noconfirm invest.spec' % {
            'python': sys.executable,
            'pyinstaller': pyinstaller_file,
        }, cwd='exe')

    bindir = os.path.join('exe', 'dist', 'invest_dist')
    sh('pip freeze > package_versions.txt', cwd=bindir)

    if not os.path.exists('dist'):
        dry('mkdir dist',
            os.makedirs, 'dist')

    invest_dist = os.path.join('dist', 'invest_dist')
    if os.path.exists(invest_dist):
        dry('rm -r %s' % invest_dist,
            shutil.rmtree, invest_dist)

    dry('cp -r %s %s' % (bindir, invest_dist),
        shutil.copytree, bindir, invest_dist)

@task
@cmdopts([
    ('bindir=', 'b', ('Folder of binaries to include in the installer. '
                      'Defaults to dist/invest-bin')),
    ('insttype=', 'i', ('The type of installer to build. '
                        'Defaults depend on the current system: '
                        'Windows=nsis, Mac=dmg, Linux=deb')),
    ('arch=', 'a', 'The architecture of the binaries'),
    ('force-dev', '', 'Allow a build when a repo version differs from tracked versions'),
])
def build_installer(options):
    """
    Build an installer for the target OS/platform.
    """
    default_installer = {
        'Darwin': 'dmg',
        'Windows': 'nsis',
        'Linux': 'deb'
    }

    # set default options if they have not been set by the user.
    # options don't exist in the options object unless the user defines it.
    defaults = [
        ('bindir', os.path.join('dist', 'invest_dist')),
        ('insttype', default_installer[platform.system()]),
        ('arch', platform.machine())
    ]
    for option_name, default_val in defaults:
        try:
            getattr(options, option_name)
        except AttributeError:
            setattr(options, option_name, default_val)

    if not os.path.exists(options.bindir):
        print 'WARNING: Binary dir %s not found' % options.bindir
        print 'WARNING: Regenerating binaries'
        call_task('build_bin')

    # version comes from the installed version of natcap.invest
    version = _invest_version()
    command = options.insttype.lower()

    if not os.path.exists(options.bindir):
        print "ERROR: Binary directory %s not found" % options.bindir
        print "ERROR: Run `paver build_bin` to make new binaries"
        return

    if command == 'nsis':
        _build_nsis(version, options.bindir, 'x86')
    elif command == 'dmg':
        _build_dmg(version, options.bindir)
    elif command == 'deb':
        _build_fpm(version, options.bindir, 'deb')
    elif command == 'rpm':
        _build_fpm(version, options.bindir, 'rpm')
    else:
        print 'ERROR: command not recognized: %s' % command
        return 1


def _build_fpm(version, bindir, pkg_type):
    print "WARNING:  Building linux packages is not yet fully supported"
    print "WARNING:  The package will build but won't yet install properly"
    print

    options = {
        'pkg_type': pkg_type,
        'version': version,
        'bindir': bindir
    }

    fpm_command = (
        'fpm -s dir -t %(pkg_type)s'
        ' -n invest'    # deb packages don't do well with uppercase
        ' -v %(version)s'
        ' -p dist/'
        ' --prefix /usr/lib/'
        ' -m James Douglass <jdouglass@stanford.edu>'
        ' --url http://naturalcapitalproject.org'
        ' --vendor "Natural Capital Project"'
        ' --license "Modified BSD"'
        ' --provides "invest"'
        ' --description "InVEST (Integrated Valuation of Ecosystem Services '
            'and Tradeoffs) is a family of tools for quantifying the values '
            'of natural capital in clear, credible, and practical ways. In '
            'promising a return (of societal benefits) on investments in '
            'nature, the scientific community needs to deliver knowledge and '
            'tools to quantify and forecast this return. InVEST enables '
            'decision-makers to quantify the importance of natural capital, '
            'to assess the tradeoffs associated with alternative choices, and '
            'to integrate conservation and human development."'
        '--after-install ./installer/linux/postinstall.sh'
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
    invest_repo = REPOS_DICT['invest-2']
    if not os.path.exists(invest_repo.local_path):
        call_task('fetch', args=[invest_repo.local_path])

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

    nsis_bindir = nsis_bindir.replace('/', r'\\')

    nsis_params = [
        '/DVERSION=%s' % version,
        '/DVERSION_DISK=%s' % version,
        '/DINVEST_3_FOLDER=%s' % nsis_bindir,
        '/DSHORT_VERSION=%s' % version,  # some other value?
        '/DARCHITECTURE=%s' % arch,
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
@consume_args
    ('force-dev', '', 'Allow development versions of repositories to be used.'),
    ('insttype=', 'i', ('The type of installer to build.  Defaults depend on '
                        'the current system: Windows=nsis, Mac=dmg, Linux=deb. '
                        'rpm is also available.')),
    ('arch=', 'a', 'The architecture of the binaries.  Defaults to the sustem arch.'),
    ('nodata', '', "Don't build the data zipfiles"),
    ('nodocs', '', "Don't build the documentation"),
    ('nobin', '', "Don't build the binaries"),
])
def zip(args):
    """
    Zip a folder and save it to an output zip file.

    Usage: paver zip archivename dirname
    Any missing and needed repositories will be cloned.
    """

    Arguments:
        archivename - the filename of the output archive
            repo.clone()
            repo.update(repo.tracked_version())

        # if we ARE NOT allowing dev builds
        if getattr(options, 'force-dev', False) is False:
            current_rev = repo.current_rev()
            tracked_rev = repo.tracked_version()
            if not repo.at_known_rev():
                raise BuildFailure(('ERROR: %(local_path)s at rev %(cur_rev)s, '
                                    'but expected to be at rev %(exp_rev)s') % {
                                        'local_path': repo.local_path,
                                        'cur_rev': current_rev,
                                        'exp_rev': tracked_rev})
        else:
            print 'WARNING: %s revision differs, but --force-dev provided' % repo.local_path
        print 'Repo %s is at rev %s' % (repo.local_path, tracked_rev)


    # Call these tasks unless the user requested not to.
        dirname - the name of the folder to archive.
        ('nodata', False),
        ('nobin', False),
        ('nodocs', False),
    ]
    for attr, default_value in defaults:
        task_base = attr[2:]
        try:
            getattr(options, attr)
        except AttributeError:
            # when the user doesn't provide a --no(data|bin|docs) option,
            # AttributeError is raised.
            task_name = 'build_%s' % task_base
            call_task(task_name)
        else:
            print 'Skipping task %s' % task_base

    # The installer task has its own parameter defaults.  Let the
    # build_installer task handle most of them.  We can pass in some of the
    # parameters, though.
    installer_options = {
        'bindir': os.path.join('exe', 'dist', 'invest_dist'),
    }
    for arg in ['insttype', 'arch']:
        try:
            installer_options[arg] = getattr(options, arg)
        except AttributeError:
            # let the build_installer task handle this default.
            pass
    call_task('build_installer', options=installer_options)
    call_task('collect_release_files')


@task
def collect_release_files():
    """
    Collect release-specific files into a single distributable folder.
    """
    # make a distribution folder for this build version.
    # rstrip to take off the newline
    _invest = imp.load_source('versioning', 'src/natcap/invest/__init__.py')
    invest_version = _invest.__version__
    dist_dir = os.path.join('dist', 'invest_%s' % invest_version)
    if not os.path.exists(dist_dir):
        dry('mkdir %s' % dist_dir, os.makedirs, dist_dir)

    if len(args) > 2:
    data_dir = os.path.join(dist_dir, 'data')
    if not os.path.exists(data_dir):
        raise BuildFailure('zip takes 2 arguments only.')

    archive_name = args[0]
    source_dir = args[1]
        dry('cp %s %s' % (data_zip, out_filename),
            shutil.copyfile, data_zip, out_filename)
        dry('rm %s' % out_filename,
            os.remove, data_zip)

    # copy the installer(s) into the new folder
    installer_files = []
    for pattern in ['*.exe', '*.dmg', '*.deb', '*.rpm']:
        glob_pattern = os.path.join('dist', pattern)
        installer_files += glob.glob(glob_pattern)

    for installer in installer_files:
        new_file = os.path.join(dist_dir, os.path.basename(installer))
    dry('zip -r %s %s.zip' % (source_dir, archive_name),
        shutil.make_archive, **{
            'base_name': archive_name,
            os.remove, installer)

    # copy HTML documentation into the new folder.
    html_docs = os.path.join('doc', 'users-guide', 'build', 'html')
            'format': 'zip',
            'root_dir': source_dir,
    out_dir = os.path.join(dist_dir, 'documentation')
    if os.path.exists(html_docs):
        if os.path.exists(out_dir):
            'base_dir': '.'})
                shutil.rmtree, out_dir)
        dry('cp -r %s %s' % (html_docs, out_dir),
            shutil.copytree, html_docs, out_dir)

        out_pdf = os.path.join(dist_dir, os.path.basename(pdf))
        dry('cp %s %s' % (pdf, out_pdf),
            shutil.copyfile, pdf, out_pdf)
    else:
        print "Skipping docs, since html docs were not built"


@task
def jenkins_installer():
    """
    Run a jenkins build via paver.
    """

    call_task('clean')
    call_task('fetch')
    call_task('env', options={
        'system_site_packages': True,
        'clear': True,
        'with_invest': True,
        'envname': 'release_env',
    })

    # call the
    if platform.system() == 'Windows':
        sh(r'jenkins\windows_build.bat')
    else:
        sh(r'jenkins/posix_build.sh')

