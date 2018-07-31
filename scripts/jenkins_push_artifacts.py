# encoding=UTF-8
"""jenkins_push_artifacts.py"""
import os
import platform
import time
import subprocess
import logging
import glob

import paramiko
from paramiko import SSHClient
import setuptools_scm


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
if platform.system() == 'Windows':
    _HOME_DIR = os.path.join('C:\\', 'cygwin', 'home', 'SYSTEM')
else:
    _HOME_DIR = os.path.expanduser('~')
JENKINS_PRIVATE_KEY_PATH = os.path.join(_HOME_DIR, '.ssh', 'dataportal-id_rsa')
DATAPORTAL_USER = 'dataportal'
DATAPORTAL_HOST = 'data.naturalcapitalproject.org'
DIST_DIR = 'dist'


def _fix_path(path):
    """Fix up a windows path to work on linux.

    Converts any windows-specific path separators to linux.

    Parameters:
        path (string): The path to modify.

    Returns:
        The modified path."""
    # destination OS is linux, so adjust windows filepaths to match
    if platform.system() == 'Windows':
        return path.replace(os.sep, '/')
    return path


def _sftp_callback(bytes_transferred, total_bytes):
    """Callback to display progress when uploading files via SFTP."""
    try:
        current_time = time.time()
        if current_time - _sftp_callback.last_time > 2:
            tx_ratio = bytes_transferred / float(total_bytes)
            tx_ratio = round(tx_ratio*100, 2)

            print 'SFTP copied {transf} out of {total} ({ratio} %)'.format(
                transf=bytes_transferred,
                total=total_bytes,
                ratio=tx_ratio)
            _sftp_callback.last_time = current_time
    except AttributeError:
        _sftp_callback.last_time = time.time()


def push(target_dir, files_to_push, files_to_unzip=None):
    """Push (and optionally unzip) files on dataportal.

    Parameters:
        target_dir (string): The directory on dataportal where files should be
            uploaded.
        files_to_push (list of strings): A list of string paths to files that
            should all be uploaded to ``target_dir``.
        files_to_unzip (list of strings): A list of string paths that are in
            ``files_to_push`` that should be unzipped after upload.

    Returns:
        None."""

    if files_to_unzip is None:
        files_to_unzip = []

    LOGGER.info('Writing paramiko logging to paramiko-log.txt')
    paramiko.util.log_to_file('paramiko-log.txt')

    ssh = SSHClient()
    ssh.load_system_host_keys()

    # Automatically add host key if needed
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key = paramiko.RSAKey.from_private_key_file(
        JENKINS_PRIVATE_KEY_PATH)

    ssh.connect(DATAPORTAL_HOST, 22, username=DATAPORTAL_USER, password=None,
                pkey=private_key)

    # Make folders on remote if needed.
    ssh.exec_command(
        'if [ ! -d "{dir}" ]; then mkdir -p -v "{dir}"; fi'.format(
            dir=_fix_path(target_dir)))

    print 'Opening SCP connection'
    sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())

    for transfer_file in files_to_push:
        target_filename = os.path.join(target_dir,
                                       os.path.basename(transfer_file))

        # Convert windows to linux paths
        target_filename = _fix_path(target_filename)
        print 'Transferring %s -> %s ' % (transfer_file,
                                          target_filename)
        for repeat in [True, True, False]:
            try:
                sftp.put(transfer_file, target_filename,
                         callback=_sftp_callback)
            except IOError as filesize_inconsistency:
                # IOError raised when the file on the other end reports a
                # different filesize than what we sent.
                if not repeat:
                    raise filesize_inconsistency

    for filename in files_to_unzip:
        remote_zipfile_path = _fix_path(os.path.join(
            target_dir, os.path.basename(filename)))
        print 'Unzipping %s on remote' % remote_zipfile_path
        _, stdout, stderr = ssh.exec_command(
            ('cd {releasedir}; '
             'unzip -o `ls -tr {zipfile} | tail -n 1`').format(
                 releasedir=_fix_path(target_dir),
                 zipfile=os.path.basename(remote_zipfile_path)))
        print "STDOUT:"
        for line in stdout:
            print line

        print "STDERR:"
        for line in stderr:
            print line

    print 'Closing down SCP'
    sftp.close()

    print 'Closing down SSH'
    ssh.close()


def main():
    """Determine which and where to upload files."""
    hg_path = subprocess.check_output('hg showconfig paths.default',
                                      shell=True).rstrip()

    username, reponame = hg_path.split('/')[-2:]
    version_string = setuptools_scm.get_version(
        root=os.path.join(os.path.dirname(__file__), '..'),
        version_scheme='post-release',
        local_scheme='node-and-date')
    print 'Using version string %s' % version_string

    if username == 'natcap' and reponame == 'invest':
        if 'post' in version_string:
            data_dirname = 'develop'
        else:
            data_dirname = version_string

        data_dir = os.path.join('public_html', 'invest-data', data_dirname)
        release_dir = os.path.join('public_html', 'invest-releases',
                                   version_string)

    else:
        release_dir = os.path.join('public_html', 'nightly-build',
                                   'invest-forks', username)
        data_dir = os.path.join(release_dir, 'data')

    LOGGER.debug('Release dir: %s', release_dir)
    LOGGER.debug('Data dir: %s', data_dir)

    LOGGER.info('Uploading data files')
    push(data_dir, glob.glob(os.path.join(DIST_DIR, 'data', '*.zip')))

    files_to_upload = filter(lambda x: version_string in x,
                             glob.glob(os.path.join(DIST_DIR, '*.*')))
    files_to_unzip = filter(lambda x: 'userguide' in x or 'apidocs' in x,
                            files_to_upload)
    LOGGER.debug('Files to upload: %s', files_to_upload)
    LOGGER.debug('Files to unzip on remote: %s', files_to_unzip)
    LOGGER.info('Uploading non-data files, unzipping documentation')
    push(release_dir, files_to_upload, files_to_unzip)


if __name__ == '__main__':
    main()
