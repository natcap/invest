import subprocess
import logging
import platform
import collections
import shutil
import os
import tempfile
import atexit
import yaml

HG_CALL = 'hg log -r . --config ui.report_untrusted=False'

LOGGER = logging.getLogger('natcap.invest.versioning')
LOGGER.setLevel(logging.ERROR)


def build_data():
    """Returns a dictionary of relevant build data."""
    data = {
        'release': get_latest_tag(),
        'build_id': get_build_id(),
        'py_arch': get_py_arch(),
        'version_str': version(),
        'branch': get_branch(),
        'pep440': get_pep440(branch=False),
        'pep440branch': get_pep440(branch=True),
    }
    return data

def write_build_info(source_file_uri):
    """Write the build information to the file specified as `source_file_uri`.
    """
    temp_file_uri = _temporary_filename()
    temp_file = open(temp_file_uri, 'w+')

    source_file = open(os.path.abspath(source_file_uri))
    for line in source_file:
        if line == "__version__ = 'dev'\n":
            temp_file.write("__version__ = '%s'\n" % get_pep440(branch=False))
        elif line == "build_data = None\n":
            build_information = build_data()
            temp_file.write("build_data = %s\n" % str(build_information.keys()))
            for key, value in sorted(build_information.iteritems()):
                temp_file.write("%s = '%s'\n" % (key, value))
        else:
            temp_file.write(line)
    temp_file.flush()
    temp_file.close()

    source_file.close()
    os.remove(source_file_uri)
    shutil.copyfile(temp_file_uri, source_file_uri)

def _temporary_filename():
    """Returns a temporary filename using mkstemp. The file is deleted
        on exit using the atexit register.  This function was migrated from
        the invest-3 raster_utils file, rev 11354:1029bd49a77a.

        returns a unique temporary filename"""

    file_handle, path = tempfile.mkstemp()
    os.close(file_handle)

    def remove_file(path):
        """Function to remove a file and handle exceptions to register
            in atexit"""
        try:
            os.remove(path)
        except OSError as exception:
            #This happens if the file didn't exist, which is okay because maybe
            #we deleted it in a method
            pass

    atexit.register(remove_file, path)
    return path

def get_py_arch():
    """This function gets the python architecture string.  Returns a string."""
    return platform.architecture()[0]

def get_release_version():
    """This function gets the release version.  Returns either the latest tag
    (if we're on a release tag) or None, if we're on a dev changeset."""
    if get_tag_distance() == 0:
        return get_latest_tag()
    return None

def version():
    """This function gets the module's version string.  This will be either the
    dev build ID (if we're on a dev build) or the current tag if we're on a
    known tag.  Either way, the return type is a string."""
    release_version = get_release_version()
    if release_version == None:
        return build_dev_id(get_build_id())
    return release_version

def build_dev_id(build_id=None):
    """This function builds the dev version string.  Returns a string."""
    if build_id == None:
        build_id = get_build_id()
    return 'dev%s' % (build_id)

def get_architecture_string():
    """Return a string representing the operating system and the python
    architecture on which this python installation is operating (which may be
    different than the native processor architecture.."""
    return '%s%s' % (platform.system().lower(),
        platform.architecture()[0][0:2])

def get_version_from_hg():
    """Get the version from mercurial.  If we're on a tag, return that.
    Otherwise, build the dev id and return that instead."""
    # TODO: Test that Hg exists before getting this information.
    if get_tag_distance() == 0:
        return get_latest_tag()
    else:
        return build_dev_id()

def _increment_tag(version_string):
    split_string = version_string.split('.dev')

    if len(split_string) == 1:
        # When the version string is just the tag, we return the tag
        return version_string
    else:
        # increment the minor version number and not the update num.
        tag = split_string[0].split('.')

        # If there's never been a tag, assume 0.0.0 was the tag at rev 0.
        if len(tag) == 1 and tag[0] in ['null', 'None']:
            tag = ['0', '0', '0']

        tag[-2] = str(int(tag[-2]) + 1)
        tag[-1] = '0'
        return '.'.join(tag) + '.dev' + split_string[1]

def get_pep440(branch=True):
    """
    Build a PEP440-compliant version.  Returns a string.
    """
    template_string = "%(latesttag)s.dev%(tagdist)s+n%(node)s"
    if branch is True:
        template_string += "-%(branch)s"

    if is_archive():
        data = {
            'tagdist': get_archive_attr('latesttagdistance'),
            'latesttag': get_archive_attr('latesttag'),
            'node': get_archive_attr('node')[:8],
            'branch': get_archive_attr('branch'),
        }
        return _increment_tag(template_string % data)
    else:
        data = {
            'tagdist': '{latesttagdistance}',
            'latesttag': '{latesttag}',
            'node': '{node|short}',
            'branch': '{branch}',
        }
        cmd = HG_CALL + ' --template "%s"' % template_string % data
        return _increment_tag(run_command(cmd))

def get_build_id():
    """Call mercurial with a template argument to get the build ID.  Returns a
    python bytestring."""
    if is_archive():
        tagdist = get_archive_attr('latesttagdistance')
        latesttag = get_archive_attr('latesttag')
        node = get_archive_attr('node')[:8]
        return "%s:%s [%s]" % (tagdist, latesttag, node)
    cmd = HG_CALL + ' --template "{latesttagdistance}:{latesttag} [{node|short}]"'
    return run_command(cmd)

def get_tag_distance():
    """Call mercurial with a template argument to get the distance to the latest
    tag.  Returns an int."""
    if is_archive():
        return get_archive_attr('latesttagdistance')
    cmd = HG_CALL + ' --template "{latesttagdistance}"'
    return int(run_command(cmd))

def get_latest_tag():
    """Call mercurial with a template argument to get the latest tag.  Returns a
    python bytestring."""
    if is_archive():
        return get_archive_attr('latesttag')
    cmd = HG_CALL + ' --template "{latesttag}"'
    return run_command(cmd)

def get_branch():
    """
    Get the branch from hg.
    """
    if is_archive():
        return get_archive_attr('branch')
    cmd = HG_CALL + ' --template "{branch}"'
    return run_command(cmd)

def run_command(cmd):
    """Run a subprocess.Popen command.  This function is intended for internal
    use only and ensures a certain degree of uniformity across the various
    subprocess calls made in this module.

    cmd - a python string to be executed in the shell.

    Returns a python bytestring of the output of the input command."""
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.read()

def is_archive():
    if os.path.exists('.hg_archival.txt'):
        return True
    return False

def get_archive_attr(attr):
    """
    If we're in an hg archive, there will be a file '.hg_archival.txt' in the
    repo root.  If this is the case, we can fetch relevant build information
    from this file that we might normally be able to get directly from hg.

    Parameters:
        attr (string): The archive attr to fetch.  One of
        "repo"|"node"|"branch"|"latesttag"|"latesttagdistance"|"changessincelatesttag"

    Returns:
        The value of the attribute in the .hg_archival file.

    Raises:
        IOError when the .hg_archival.txt file cannot be found.
        KeyError when `attr` is not in .hg_archival.txt
    """
    return yaml.safe_load(open('.hg_archival.txt'))[attr]


