import os
import json
import functools
import subprocess
from unittest import SkipTest


def skip_if_data_missing(repo_path):
    """
    Decorator for unittest.TestCase test functions.  Raises SkipTest if the
    pygeoprocessing data repository has not been cloned.

    Arguments:
        repo_path (string): The path to an SVN data repo on disk.

    Returns:
        A reference to a decorated unittest.TestCase test method that will
        raise SkipTest when executed.
    """
    message = 'Data repo %s is not cloned' % os.path.basename(repo_path)

    def data_repo_aware_skipper(item):

        @functools.wraps(item)
        def skip_if_data_not_cloned(self, *args, **kwargs):
            if not os.path.exists(repo_path):
                raise SkipTest(message)
            item(self, *args, **kwargs)
        return skip_if_data_not_cloned
    return data_repo_aware_skipper


def checkout_svn(local_path, remote_path, rev=None):
    """
    Check out (or update) an SVN repository to the target revision.

    Arguments:
        local_path (string): The path to an SVN repository on disk.
        remote_path (string): The path to the SVN repository to check out.
        rev=None (string or None): The revision to check out.  If None, the
            latest revision will be checked out.

    """
    if rev is None:
        rev = 'HEAD'
    else:
        rev = str(rev)

    if os.path.exists(local_path):
        cur_dir = os.getcwd()
        os.chdir(local_path)
        subprocess.call(['svn', 'update', '-r', rev])
        os.chdir(cur_dir)
    else:
        subprocess.call(['svn', 'checkout', remote_path, local_path, '-r',
                         rev])


def load_config(config_file):
    """
    Load the SCM configuration from a JSON configuration file.

    Expects that the config file has three values:
    {
        "local": <local path to repo>,
        "remote": <path to remote repo>,
        "rev": <target revision>
    }
    """

    data = json.load(open(config_file))
    if not os.path.isabs(data['local']):
        # assume that the data path is relative to the configuration file.
        config_file_dir = os.path.dirname(config_file)
        data['local'] = os.path.join(config_file_dir, data['local'])

    return data
