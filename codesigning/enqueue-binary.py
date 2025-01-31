"""Enqueue a windows binary for signing.

To call this script, you need to set the ACCESS_TOKEN environment variable from
the software team secrets store.

Example invocation:

    $ ACCESS_TOKEN=abcs1234 python3 enqueue-binary.py <public https url to binary on gcs>
"""

import os
import sys
from urllib import parse
from urllib import request

DATA = parse.urlencode({
    'token': os.environ['ACCESS_TOKEN'],
    "url": sys.argv[1],
    "action": "enqueue",
}).encode()

req = request.Request(
    'https://us-west1-natcap-servers.cloudfunctions.net/codesigning-queue',
    data=DATA)
response = request.urlopen(req)
