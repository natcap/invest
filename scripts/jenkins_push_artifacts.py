# encoding=UTF-8
"""jenkins_push_artifacts.py"""
import os
import platform
import getpass
import socket
import time
import subprocess
import logging

import paramiko
from paramiko import SSHClient
import setuptools_scm


LOGGER = logging.getLogger(__name__)
logging.basicConfig()
JENKINS_PRIVATE_KEY_PATH = os.path.expanduser(
    os.path.join('~', '.ssh', 'dataportal-id_rsa'))
DATAPORTAL_USER = 'dataportal'
DATAPORTAL_HOST = 'data.naturalcapitalproject.org'


def _fix_path(path):
    """Fix up a windows path to work on linux"""
    # destination OS is linux, so adjust windows filepaths to match
    if platform.system() == 'Windows':
        return path.replace(os.sep, '/')
    return path


def _sftp_callback(bytes_transferred, total_bytes):
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



def push(private_key_path, target_dir, files_to_push, files_to_unzip):
    """Push (and optionally unzip) files on dataportal.

    Parameters:
        private_key_path (string): The path to a private key on disk that
            should be used in addition to any system keys.
        target_dir (string): The directory on dataportal where files should be
            uploaded.
        files_to_push (list of strings): A list of string paths to files that
            should all be uploaded to ``target_dir``.
        files_to_unzip (list of string): A list of string paths that are in
            ``files_to_push`` that should be unzipped after upload.

    Returns:
        None."""
    hg_path = subprocess.check_output('hg showconfig paths.default',
                                      shell=True).rstrip()

    username, reponame = hg_path.split('/')[-2:]
    version_string = setuptools_scm.get_version(
        root=os.path.join(os.path.dirname(__file__), '..'),
        version_scheme='post-release',
        local_scheme='node-and-date')

    if username == 'natcap' and reponame == 'invest':
        if 'post' in version_string:
            data_dirname = 'develop'
        else:
            data_dirname = version_string

        data_dir = os.path.join('invest-data', data_dirname)
        release_dir = os.path.join('invest-releases', version_string)

    else:
        release_dir = os.path.join('nightly-build', 'invest-forks', username)
        data_dir = os.path.join(release_dir, 'data')

    LOGGER.debug('Release dir: %s', release_dir)
    LOGGER.debug('Data dir: %s', data_dir)

    LOGGER.info('Writing paramiko logging to paramiko-log.txt')
    paramiko.util.log_to_file('paramiko-log.txt')

    ssh = SSHClient()
    ssh.load_system_host_keys()

    # Automatically add host key if needed
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

    target_dir = 'public_html'

    ssh.connect(DATAPORTAL_HOST, 22, username=DATAPORTAL_USER, password=None,
                pkey=private_key)

    # Make folders on remote if needed.
    if target_dir is not None:
        ssh.exec_command(
            'if [ ! -d "{dir}" ]\nthen\nmkdir -p -v {dir}\nfi'.format(
                dir=target_dir))

    print 'Opening SCP connection'
    sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
    for transfer_file in files_to_push:
        file_basename = os.path.basename(transfer_file)
        if target_dir is not None:
            target_filename = os.path.join(target_dir, file_basename)
        else:
            target_filename = file_basename

        # Convert windows to linux paths
        target_filename = _fix_path(target_filename)
        print 'Transferring %s -> %s ' % (os.path.basename(transfer_file),
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

    # correct the filepath from Windows to Linux
    if platform.system() == 'Windows':
        release_dir = release_dir.replace(os.sep, '/')

    if release_dir.startswith('public_html/'):
        release_dir = release_dir.replace('public_html/', '')

    for filename in files_to_unzip:

        print 'Unzipping %s on remote' % filename
        ssh.exec_command(
            ('cd public_html/{releasedir}; '
             'unzip -o `ls -tr {zipfile} | tail -n 1`').format(
                 releasedir=release_dir,
                 zipfile=filename
             )
        )

    print 'Closing down SCP'
    sftp.close()

    print 'Closing down SSH'
    ssh.close()


if __name__ == '__main__':
    push()



