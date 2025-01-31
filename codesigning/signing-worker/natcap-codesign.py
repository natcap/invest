#!/usr/bin/env python3

import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pexpect  # apt install python3-pexpect
import requests  # apt install python3-requests

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
CERTIFICATE = sys.argv[1]

FILE_DIR = os.path.dirname(__file__)
TOKEN_FILE = os.path.join(FILE_DIR, "access_token.txt")
with open(TOKEN_FILE) as token_file:
    ACCESS_TOKEN = token_file.read().strip()


def get_from_queue():
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


# See https://stackoverflow.com/a/16696317
def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def upload_to_bucket(filename, path_on_bucket):
    subprocess.run(['gsutil', 'cp', filename, path_on_bucket], check=True)


def sign_file(file_to_sign):
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
                LOGGER.info(f"Removing {filename}")
                os.remove(filename)
                LOGGER.info("Signing complete.")
        except Exception as e:
            LOGGER.exception("Unexpected error signing file")
            raise e
        time.sleep(60)


if __name__ == '__main__':
    main()
