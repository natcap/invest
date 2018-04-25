# encoding=UTF-8
"""jenkins_push_artifacts.py"""
import os
import platform
import getpass
import socket
import time
import subprocess

import paramiko
import setuptools_scm


JENKINS_PRIVATE_KEY_PATH = os.expanduser(
    os.path.join('~', '.ssh', 'dataportal-id_rsa'))
DATAPORTAL_USER = 'dataportal'
DATAPORTAL_HOST = 'data.naturalcapitalproject.org'


def push(args):
    """Push a file or files to a remote server.

    Usage:
        paver push [--private-key=KEYFILE] [--password] [--makedirs] [user@]hostname[:target_dir] file1, file2, ...

    Uses pythonic paramiko-based SCP to copy files to the remote server.

    if --private-key=KEYFILE is provided, KEYFILE must be the path to the private
    key file to use.  If this file cannot be found, ValueError will be raised.

    If --password is provided at the command line, the user will be prompted
    for a password.  This is sometimes required when the remote's private key
    requires a password to decrypt.

    If --makedirs is provided, intermediate directories will be created as needed.

    If a target username is not provided ([user@]...), the current user's username
    used for the transfer.

    If a target directory is not provided (hostname[:target_dir]), the current
    directory of the target user is used.
    """
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
                raise ValueError(
                    'Cannot fild private key file %s' % private_key_file)
            print 'Using private key %s' % private_key_file
            private_key = paramiko.RSAKey.from_private_key_file(private_key_file)
            break
    try:
        destination_config = args[0]
    except IndexError:
        raise ValueError("ERROR: destination config must be provided")

    files_to_push = args[1:]
    if len(files_to_push) == 0:
        raise ValueError("ERROR: At least one file must be given")

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
        raise ValueError('ERROR: incorrect password or bad SSH key')
    except paramiko.PasswordRequiredException:
        raise ValueError('ERROR: password required to decrypt private key on remote.  Use --password flag')
    except socket.error as other_error:
        raise ValueError(other_error)

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


def unzip_on_dataportal(zipfile, release_dir):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.RSAKey.from_private_key_file(JENKINS_PRIVATE_KEY_PATH)

    print 'Connecting to host'
    ssh.connect(DATAPORTAL_HOST, 22, username=DATAPORTAL_USER, password=None,
                pkey=pkey)

    # correct the filepath from Windows to Linux
    if platform.system() == 'Windows':
        release_dir = release_dir.replace(os.sep, '/')

    if release_dir.startswith('public_html/'):
        release_dir = release_dir.replace('public_html/', '')

    print 'Unzipping %s on remote' % filename
    _, stdout, stderr = ssh.exec_command(
        'cd public_html/{releasedir}; unzip -o `ls -tr {zipfile} | tail -n 1`'.format(
            releasedir=release_dir,
            zipfile=zipfile
        )
    )

    print "STDOUT:"
    for line in stdout:
        print line

    print "STDERR:"
    for line in stderr:
        print line

    ssh.close()



def main():
    hg_path = subprocess.check_output('hg showconfig paths.default',
                                      shell=True).rstrip()

    username, reponame = hg_path.split('/')[-2:]
    version_string = setuptools_scm.get_version(
        version_scheme='post_release',
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


    




if __name__ == '__main__':
    # 'username': 'dataportal',
    # 'host': 'data.naturalcapitalproject.org',
    # 'dataportal': 'public_html',
    # # Only push data zipfiles if we're on Windows.
    # # Have to pick one, as we're having issues if all slaves are trying
    # # to push the same large files.
    # 'include_data': platform.system() == 'Windows',



