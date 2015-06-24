import subprocess
import imp
import os
import logging
import traceback
import platform
import codecs

HG_CALL = 'hg log -r . --config ui.report_untrusted=False'

LOGGER = logging.getLogger('natcap.invest.build_utils')
LOGGER.setLevel(logging.ERROR)

def invest_version(uri=None, force_new=False, attribute='version_str',
        exec_dir=None):
    """Get the version of InVEST by importing natcap.invest.invest_version and
    using the appropriate version string from that module.  If
    natcap.invest.invest_version cannot be found, it is programmatically
    generated and then reimported.

    NOTE: natcap.invest.invest_version should be generated and distributed with
    the natcap.invest package, or else we run the risk of causing natcap.invest
    programs to crash if the do not have CLI mercurial installed.

    attribute - an attribute to fetch from the invest_version file.
    exec_dir - a string URI to a folder on disk where the execution should take
        place.  It need not be inside of a python directory, but it must be
        inside a mercurial installation.

    Returns a python bytestring with the version identifier, as appropriate for
    the development version or the release version."""

    def get_file_name(uri):
        """This function gets the file's basename without the extension."""
        return os.path.splitext(os.path.basename(uri))[0]

    # if the user provided an execution director, switch to that folder before
    # executing all this.  This is mostly a hack so that we can access the
    # natcap.invest version information from other projects like RIOS.
    current_dir = os.getcwd()
    if exec_dir == None:
        directory = os.path.dirname(__file__)
    else:
        directory = exec_dir
    os.chdir(directory)

    if uri == None:
        # get the location of the site-packages folder by looking at where the
        # os module is located.  Would use distutils.sysconfig,  but it was
        # causing a really nasty importError I couldn't fix when building the
        # windows exe's.
        new_uri = os.path.join(os.path.abspath(os.path.dirname(os.__file__)),
            'site-packages', 'natcap.invest', 'invest_version.pyc')
        if not os.path.exists(new_uri):
            LOGGER.debug('URI %s does not exist.  Defaulting to local paths',
                new_uri)
            new_uri = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                'invest_version.pyc')
    else:
        new_uri = uri

    LOGGER.debug('Getting the InVEST version for URI=%s' % new_uri)
    try:
        name = get_file_name(new_uri)
        new_uri_extension = os.path.splitext(new_uri)[1]
        if new_uri_extension == '.pyc':
            version_info = imp.load_compiled(name, new_uri)
        elif new_uri_extension == '.py':
            version_info = imp.load_source(name, new_uri)
        elif new_uri_extension == '.pyd':
            version_info = imp.load_dynamic(name, new_uri)
        else:
            os.chdir(current_dir)
            raise IOError('Module %s must be importable by python' % new_uri)
#        print 'imported version'
        LOGGER.debug('Successfully imported version file')
        found_file = True
    except (ImportError, IOError, AttributeError, TypeError) as error:
        # ImportError thrown when we can't import the target source
        # IOError thrown if the target source file does not exist on disk
        # AttributeError thrown in some odd cases
        # TypeError thrown when uri == None.
        # In any of these cases, try creating the version file and import
        # once again.
        LOGGER.debug('Problem importing: %s', str(error))
        LOGGER.debug('Unable to import version.  Creating a new file at %s',
            new_uri)
#        print "can't import version from %s" % new_uri
        found_file = False

    return_value = None
    # We only want to write a new version file if the user wants to force the
    # file's creation, OR if the user specified a URI, but we can't find it.
    LOGGER.debug('Force_new=%s found_file=%s', force_new, found_file)
    if force_new or (not found_file and uri != None):
        try:
            write_version_file(new_uri)
            name = get_file_name(new_uri)
            version_info = imp.load_source(name, new_uri)
            LOGGER.debug('Wrote a new version file to %s', new_uri)
            print 'Wrote a new version file'
        except ValueError as e:
            # Thrown when Mercurial is not found to be installed in the local
            # directory.  This is a band-aid fix for when we import InVEST from
            # within a distributed version of RIOS.
            # When this happens, return the exception as a string.
            LOGGER.debug('ValueError encountered: %s', str(e))
            return_value = str(e)
    elif not found_file and uri == None and attribute == 'version_str':
        # If we have not found the version file and no URI is provided, we need
        # to get the version info from HG.
#        print 'getting version from hg'
        LOGGER.debug('Getting the version number from HG')
        try:
            return_value = get_version_from_hg()
        except ValueError:
            # When mercurial is not installed
            return_value = 'dev'

    try:
        LOGGER.debug('Returning attribute %s', attribute)
        return_value = getattr(version_info, attribute)
    except UnboundLocalError:
        LOGGER.debug('Unable to import version information, skipping.')

    os.chdir(current_dir)
    return return_value

def write_version_file(filepath):
    """Write the version number to the file designated by filepath.  Returns
    nothing."""
    comments = [
        'The version noted below is used throughout InVEST as a static version',
        'that differs only from build to build.  Its value is determined by ',
        'setup.py and is based off of the time and date of the last revision.',
        '',
        'This file is programmatically generated when natcap.invest is built. ',
    ]

    # Open the version file for writing
    fp = codecs.open(filepath, 'w', encoding='utf-8')

    # Write the comments as comments to the file and write the version to the
    # file as well.
    for comment in comments:
        fp.write('# %s\n' % comment)

    # Determine how to record the release version in the invest_version file.
    if get_tag_distance() == 0:
        release_version = get_latest_tag()
    else:
        release_version = None
    fp.write('release = \'%s\'\n' % release_version)

    # Even though we're also saving the release version, we also want to save
    # the build_id, as it can be very informative.
    build_id = get_build_id()
    fp.write('build_id = \'%s\'\n' % build_id)

    # We also care about the python architecture on which this copy of InVEST is
    # built, so record that here
    architecture = platform.architecture()[0]
    fp.write('py_arch = \'%s\'\n' % architecture)

    # Compose a full version string and save it to the file.
    if release_version == None:
        full_version_string = build_dev_id(build_id)
    else:
        full_version_string = release_version
    fp.write("version_str = '%s'\n" % (full_version_string))

    # Close the file.
    fp.close()

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

def get_build_id():
    """Call mercurial with a template argument to get the build ID.  Returns a
    python bytestring."""
    cmd = HG_CALL + ' --template "{latesttagdistance}:{latesttag} [{node|short}]"'
    return run_command(cmd)

def get_tag_distance():
    """Call mercurial with a template argument to get the distance to the latest
    tag.  Returns an int."""
    cmd = HG_CALL + ' --template "{latesttagdistance}"'
    return int(run_command(cmd))

def get_latest_tag():
    """Call mercurial with a template argument to get the latest tag.  Returns a
    python bytestring."""
    cmd = HG_CALL + ' --template "{latesttag}"'
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

