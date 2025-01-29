#!/usr/bin/env python3

import sys
import textwrap

import pexpect  # apt install python3-pexpect

CERTIFICATE = sys.argv[1]
FILETOSIGN = sys.argv[2]
SIGNED = sys.argv[3]

SIGNCODE_COMMAND = textwrap.dedent(f"""\
    osslsigncode sign \
        -pkcs11engine /usr/lib/aarch64-linux-gnu/engines-3/pkcs11.so \
        -pkcs11module /usr/lib/aarch64-linux-gnu/libykcs11.so.2 \
        -key "pkcs11:id=%02;type=private" \
        -certs {CERTIFICATE} \
        -h sha256 \
        -ts http://timestamp.sectigo.com \
        -readpass pass.txt \
        -verbose \
        -in {FILETOSIGN} \
        -out {SIGNED}""")


process = pexpect.spawnu(SIGNCODE_COMMAND)
process.expect('Enter PKCS#11 key PIN for Private key for Digital Signature:')
with open('pass.txt') as passfile:
    process.sendline(passfile.read().strip())

# print remainder of program output for our logging.
print(process.read())
