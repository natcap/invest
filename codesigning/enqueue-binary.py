"""Enqueue a windows binary for signing.

To call this script, you need to set the ACCESS_TOKEN environment variable from
the software team secrets store.

Example invocation:

    $ ACCESS_TOKEN=abcs1234 python3 enqueue-binary.py <gs:// uri to binary on gcs>
"""

import os
import sys

import requests

DATA = {
    'token': os.environ['ACCESS_TOKEN'],
    'action': 'enqueue',
    'url': sys.argv[1].replace(
        'gs://', 'https://storage.googleapis.com/'),
}
response = requests.post(
    'https://us-west1-natcap-servers.cloudfunctions.net/codesigning-queue',
    json=DATA
)
if response.status_code >= 400:
    print(response.text)
    sys.exit(1)
