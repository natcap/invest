#!/usr/bin/env python3
"""Service script to sign InVEST exe and dmg installers."""

import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback

import pexpect  # apt install python3-pexpect
import requests  # apt install python3-requests
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

RELEASES_BUCKET = 'releases.naturalcapitalproject.org'
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
CERTIFICATE = sys.argv[1]

# I had to install rust to a custom location in order to get the latest
# version, so it's easier to just refer to the specific rcodesign binary we
# want instead of trying to assume that whatever is on the PATH is the correct
# version.
RCODESIGN = '/opt/rust/bin/rcodesign'

FILE_DIR = os.path.dirname(__file__)
QUEUE_TOKEN_FILE = os.path.join(FILE_DIR, "access_token.txt")
with open(QUEUE_TOKEN_FILE) as token_file:
    ACCESS_TOKEN = token_file.read().strip()

SLACK_TOKEN_FILE = os.path.join(FILE_DIR, "slack_token.txt")
with open(SLACK_TOKEN_FILE) as token_file:
    SLACK_ACCESS_TOKEN = token_file.read().strip()

OS_EMOJI = {
    '.dmg': ':mac:',
    '.exe': ':windowsxp:',
}

SLACK_NOTIFICATION_SUCCESS = (
    ":lower_left_fountain_pen: {os_emoji} Successfully signed and uploaded "
    "{username}'s build `{filename}` to <{url}|google cloud>"
)

SLACK_NOTIFICATION_ALREADY_SIGNED = textwrap.dedent(
    """\
    :lower_left_fountain_pen: `{filename}` is already signed! <{url}|google cloud>
    """)

SLACK_NOTIFICATION_FAILURE = textwrap.dedent(
    """\
    :red-flag: Something went wrong while signing {filename}:
    ```
    {traceback}
    ```
    Please investigate on ncp-inkwell using:
    ```
    sudo journalctl -u natcap-codesign.service
    ```

    When ready to try re-signing, requeue the file on a workstation with:
    ```
    python codesigning/enqueue-binary.py {filename}
    ```
    """)


def post_to_slack(message):
    """Post a message to the slack channel.

    Args:
        message (str): The message to post.

    Returns:
        ``None``
    """
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {SLACK_ACCESS_TOKEN}",
            "Content-Type": "application/json; charset=utf-8"
        },
        json={
            "channel": "CESG428BH",  # sw-invest
            "text": message
        })
    resp.raise_for_status()


def get_from_queue():
    """Get an item to sign from the queue.

    Returns:
        ``None`` if there are no items in the queue, the JSON response dict
        otherwise.
    """
    response = requests.post(
        "https://us-west1-natcap-servers.cloudfunctions.net/codesigning-queue",
        headers={"Content-Type": "application/json"},
        json={
            "token": ACCESS_TOKEN,
            "action": "dequeue"
        })
    if response.status_code == 204:
        return None
    else:
        return response.json()


def download_file(url):
    """Download an arbitrarily large file.

    Adapted from https://stackoverflow.com/a/16696317

    Args:
        url (str): The URL to download.

    Returns:
        ``None``
    """
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def upload_to_bucket(filename, path_on_bucket):
    """Upload a file to a GCS bucket.

    Args:
        filename (str): The local file to upload.
        path_on_bucket (str): The path to the file on the GCS bucket, including
            the ``gs://`` prefix.

    Returns:
        ``None``
    """
    subprocess.run(['gsutil', 'cp', filename, path_on_bucket], check=True)


def sign_exe_file(file_to_sign):
    """Sign a local .exe file.

    Uses ``osslsigncode`` to sign the file using the private key stored on a
    Yubikey, and the corresponding certificate that has been exported from the
    PIV slot 9c.

    Args:
        file_to_sign (str): The local filepath to the file to sign.

    Returns:
        ``None``
    """
    signed_file = f"{file_to_sign}.signed"
    pass_file = os.path.join(FILE_DIR, 'pass.txt')

    signcode_command = textwrap.dedent(f"""\
        osslsigncode sign \
            -pkcs11engine /usr/lib/x86_64-linux-gnu/engines-3/pkcs11.so \
            -pkcs11module /usr/lib/x86_64-linux-gnu/libykcs11.so \
            -key "pkcs11:id=%02;type=private" \
            -certs {CERTIFICATE} \
            -h sha256 \
            -ts http://timestamp.sectigo.com \
            -readpass {pass_file} \
            -verbose \
            -in {file_to_sign} \
            -out {signed_file}""")

    process = pexpect.spawnu(signcode_command)
    process.expect('Enter PKCS#11 key PIN for Private key for Digital Signature:')
    with open(pass_file) as passfile:
        process.sendline(passfile.read().strip())

    # print remainder of program output for our logging.
    print(process.read())

    shutil.move(signed_file, file_to_sign)


def sign_dmg_file(file_to_sign):
    """Sign a local .dmg file.

    Args:
        file_to_sign (str): The local filepath to the DMG to sign.

    Returns:
        ``None``
    """

    p12_file = os.path.join(FILE_DIR, 'mac-certificate.p12')
    p12_pass_file = os.path.join(FILE_DIR, 'mac-certificate-pass.txt')

    subprocess.run(
        [RCODESIGN, 'sign', '--p12-file', p12_file, '--p12-password-file',
         p12_pass_file, file_to_sign], check=True, capture_output=False)


def note_signature_complete(local_filepath, target_gs_uri):
    """Create a small file next to the signed file to indicate signature.

    Args:
        gs_uri (str): The GCS URI of the signed file.
    """
    if local_filepath.endswith('.exe'):
        # Using osslsigncode to verify the output always fails for me, even though
        # the signature is clearly valid when checked on Windows.
        process = subprocess.run(
            ['osslsigncode', 'verify', '-in', local_filepath], check=False,
            capture_output=True)
    elif local_filepath.endswith('.dmg'):
        process = subprocess.run(
            [RCODESIGN, 'print-signature-info',
             local_filepath], check=True, capture_output=True)
    else:
        raise ValueError(f'Unknown filetype for {local_filepath}')

    temp_filepath = f'/tmp/{os.path.basename(local_filepath)}.signed'
    with open(temp_filepath, 'w') as signature_file:
        signature_file.write(process.stdout.decode())

    try:
        # Upload alongside the original file
        subprocess.run(
            ['gsutil', 'cp', temp_filepath, f'{target_gs_uri}.signature'],
            check=True)
    finally:
        os.remove(temp_filepath)


def has_signature(filename):
    """Check to see if the file has already been signed.

    Any signature found, if any, is not verified as a part of this function, we
    are only checking for the presence of a signature.

    Args:
        filename (str): The path to a local exe or dmg file.

    Returns:
        ``True`` if the file has a signature, ``False`` otherwise.
    """
    if filename.endswith('.exe'):
        return exe_has_signature(filename)
    elif filename.endswith('.dmg'):
        return dmg_has_signature(filename)
    else:
        raise ValueError(
            'Cannot verify signature; Unknown file extension '
            f'{os.path.splitext(filename)[1]}')


def exe_has_signature(filename):
    """Check if a file is already signed.

    Args:
        filename (str): The local filepath to the file to check.

    Returns:
        ``True`` if the file is signed, ``False`` otherwise.
    """
    process = subprocess.run(
        ['osslsigncode', 'verify', '-in', filename], capture_output=True,
        check=False)

    # Handle the case where it's possible there might not be any stdout or
    # stderr to decode.
    process_output = ""
    for output in (process.stdout, process.stderr):
        if output is not None:
            process_output += output.decode()

    if 'No signature found' in process_output:
        return False
    return True


def dmg_has_signature(filename):
    """Check if a DMG is already signed.

    Args:
        filename (str): The local filepath to the file to check.

    Returns:
        ``True`` if the file is signed, ``False`` otherwise.
    """
    process = subprocess.run(
        [RCODESIGN, 'print-signature-info', filename], capture_output=True,
        check=True)
    info = yaml.load(process.stdout.decode('utf-8'), Loader=Loader)
    if not info[0]['entity']['dmg']['signature']:
        return False
    return True


def main():
    while True:
        try:
            file_to_sign = get_from_queue()
            if file_to_sign is None:
                LOGGER.info('No items in the queue')
            else:
                LOGGER.info(f"Dequeued and downloading {file_to_sign['https-url']}")
                filename = download_file(file_to_sign['https-url'])

                LOGGER.info(f"Checking if {filename} is already signed")
                if has_signature(filename):
                    LOGGER.info(f"{filename} is already signed, skipping")
                    post_to_slack(
                        SLACK_NOTIFICATION_ALREADY_SIGNED.format(
                            filename=filename,
                            url=file_to_sign['https-url']))
                    note_signature_complete(filename, file_to_sign['gs-uri'])
                else:
                    if filename.endswith('.exe'):
                        LOGGER.info(f"Signing {filename} with osslsigncode")
                        sign_exe_file(filename)
                    elif filename.endswith('.dmg'):
                        LOGGER.info(f'Signing {filename} with rcodesign')
                        sign_dmg_file(filename)
                    LOGGER.info(f"Uploading signed file to {file_to_sign['gs-uri']}")
                    upload_to_bucket(filename, file_to_sign['gs-uri'])
                    LOGGER.info(
                        f"Adding {file_to_sign['https-url']} to signed files list")
                    note_signature_complete(filename, file_to_sign['gs-uri'])

                    LOGGER.info('Posting alert to slack')
                    if RELEASES_BUCKET in file_to_sign['https-url']:
                        username = 'natcap'
                    else:
                        username = file_to_sign['https-url'].split('/')[5]
                    post_to_slack(
                        SLACK_NOTIFICATION_SUCCESS.format(
                            os_emoji=OS_EMOJI[os.path.splitext(filename)[1]],
                            username=username,
                            filename=filename,
                            url=file_to_sign['https-url']))
                    LOGGER.info("Signing complete.")

                LOGGER.info(f"Removing {filename}")
                os.remove(filename)
        except Exception as e:
            LOGGER.exception(f"Unexpected error signing file: {e}")
            post_to_slack(
                SLACK_NOTIFICATION_FAILURE.format(
                    filename=file_to_sign['https-url'],
                    traceback=traceback.format_exc()))
        time.sleep(60)


if __name__ == '__main__':
    main()
