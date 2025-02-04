#!/usr/bin/env python3
"""Service script to sign InVEST windows binaries."""

import json
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

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
CERTIFICATE = sys.argv[1]

FILE_DIR = os.path.dirname(__file__)
QUEUE_TOKEN_FILE = os.path.join(FILE_DIR, "access_token.txt")
with open(QUEUE_TOKEN_FILE) as token_file:
    ACCESS_TOKEN = token_file.read().strip()

SLACK_TOKEN_FILE = os.path.join(FILE_DIR, "slack_token.txt")
with open(SLACK_TOKEN_FILE) as token_file:
    SLACK_ACCESS_TOKEN = token_file.read().strip()


SLACK_NOTIFICATION_SUCCESS = textwrap.dedent(
    """\
    :lower_left_fountain_pen: Successfully signed and uploaded `{filename}` to
    <{url}|google cloud>
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


def sign_file(file_to_sign):
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


def add_file_to_signed_list(url):
    """Add a file to the list of signed files on GCS.

    Args:
        url (str): The public HTTPS URL of the file to add to the list.

    Returns:
        ``None``
    """
    # Since this process is the only one that should be writing to this file, we
    # don't need to worry about race conditions.
    remote_signed_files_path = 'gs://natcap-codesigning/signed_files.json'
    local_signed_files_path = os.path.join(FILE_DIR, 'signed_files.json')

    # Test to see if the signed files json file exists in the bucket; create it
    # if not.
    exists_proc = subprocess.run(
        ['gsutil', '-q', 'stat', remote_signed_files_path], check=False)
    if exists_proc.returncode != 0:
        signed_files_dict = {'signed_files': []}
    else:
        subprocess.run(
            ['gsutil', 'cp', remote_signed_files_path,
             local_signed_files_path], check=True)
        with open(local_signed_files_path, 'r') as signed_files:
            signed_files_dict = json.load(signed_files)

    with open(local_signed_files_path, 'w') as signed_files:
        signed_files_dict['signed_files'].append(url)
        json.dump(signed_files_dict, signed_files)

    subprocess.run(
        ['gsutil', 'cp', local_signed_files_path,
         remote_signed_files_path], check=True)
    LOGGER.info(f"Added {url} to {remote_signed_files_path}")


def note_signature_complete(local_filepath, gs_uri):
    """Create a small file next to the signed file to indicate signature.

    Args:
        gs_uri (str): The GCS URI of the signed file.
    """
    # Using osslsigncode to verify the output always fails for me, even though
    # the signature is clearly valid when checked on Windows.
    process = subprocess.run(
        ['osslsigncode', 'verify', '-in', local_filepath], check=False,
        capture_output=True)

    temp_filepath = f'/tmp/{os.path.basename(local_filepath)}.signed'
    with open(temp_filepath, 'w') as signature_file:
        signature_file.write(process.stdout.decode())

    try:
        subprocess.run(
            ['gsutil', 'cp', temp_filepath, f'{gs_uri}.signature'], check=True)
    finally:
        os.remove(temp_filepath)


def main():
    while True:
        try:
            file_to_sign = get_from_queue()
            if file_to_sign is None:
                LOGGER.info('No items in the queue')
            else:
                LOGGER.info(f"Dequeued and downloading {file_to_sign['https-url']}")
                filename = download_file(file_to_sign['https-url'])
                LOGGER.info(f"Signing {filename}")
                sign_file(filename)
                LOGGER.info(f"Uploading signed file to {file_to_sign['gs-uri']}")
                upload_to_bucket(filename, file_to_sign['gs-uri'])
                LOGGER.info(
                    f"Adding {file_to_sign['https-url']} to signed files list")
                add_file_to_signed_list(file_to_sign['https-url'])
                note_signature_complete(filename, file_to_sign['gs-uri'])
                LOGGER.info(f"Removing {filename}")
                post_to_slack(
                    SLACK_NOTIFICATION_SUCCESS.format(
                        filename=filename,
                        url=file_to_sign['https-url']))
                os.remove(filename)
                LOGGER.info("Signing complete.")
        except Exception as e:
            LOGGER.exception(f"Unexpected error signing file: {e}")
            post_to_slack(
                SLACK_NOTIFICATION_FAILURE.format(
                    filename=file_to_sign['https-url'],
                    traceback=traceback.format_exc()))
        time.sleep(60)


if __name__ == '__main__':
    main()
