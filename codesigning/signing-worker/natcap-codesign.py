#!/usr/bin/env python3

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

TOKEN_FILE = os.path.join(os.path.dirname(__file__), "access_token.txt")
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

    signcode_command = textwrap.dedent(f"""\
        osslsigncode sign \
            -pkcs11engine /usr/lib/aarch64-linux-gnu/engines-3/pkcs11.so \
            -pkcs11module /usr/lib/aarch64-linux-gnu/libykcs11.so.2 \
            -key "pkcs11:id=%02;type=private" \
            -certs {CERTIFICATE} \
            -h sha256 \
            -ts http://timestamp.sectigo.com \
            -readpass pass.txt \
            -verbose \
            -in {file_to_sign} \
            -out {signed_file}""")

    process = pexpect.spawnu(signcode_command)
    process.expect('Enter PKCS#11 key PIN for Private key for Digital Signature:')
    with open('pass.txt') as passfile:
        process.sendline(passfile.read().strip())

    # print remainder of program output for our logging.
    print(process.read())

    shutil.move(signed_file, file_to_sign)


def main():
    while True:
        try:
            file_to_sign = get_from_queue()
            if file_to_sign is None:
                LOGGER.info('No items in the queue')
            else:
                filename = download_file(file_to_sign['https-url'])
                sign_file(filename)
                upload_to_bucket(filename, file_to_sign['gs-uri'])
                os.remove(filename)
        except Exception as e:
            LOGGER.exception("Unexpected error signing file")
            raise e
        time.sleep(60)


if __name__ == '__main__':
    main()
