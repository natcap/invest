import contextlib
import datetime
import json
import logging
import os
import time
from urllib.parse import unquote

import functions_framework
import google.cloud.logging  # pip install google-cloud-logging
import requests
from flask import jsonify
from google.cloud import storage  # pip install google-cloud-storage

GOOGLE_PREFIX = 'https://storage.googleapis.com'
CODESIGN_DATA_BUCKET = 'natcap-codesigning'
LOG_CLIENT = google.cloud.logging.Client()
LOG_CLIENT.setup_logging()


@contextlib.contextmanager
def get_lock():
    """Acquire a GCS-based mutex.

    This requires that the bucket we are using has versioning.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(CODESIGN_DATA_BUCKET)

    lock_obtained = False
    n_tries = 100
    for i in range(n_tries):
        lockfile = bucket.blob('mutex.lock')
        if not lockfile.generation:
            lockfile.upload_from_string(
                f"Lock acquired {datetime.datetime.now().isoformat()}")
            lock_obtained = True
            break
        else:
            time.sleep(0.1)

    if not lock_obtained:
        raise RuntimeError(f'Could not obtain lock after {n_tries} tries')

    try:
        yield
    finally:
        lockfile.delete()


@functions_framework.http
def main(request):
    """Handle requests to this GCP Cloud Function.

    All requests must be POST requests and have a JSON body with the following
    attributes:

        * token: a secret token that matches the ACCESS_TOKEN environment
            variable that is defined in the cloud function configuration.
        * action: either 'enqueue' or 'dequeue'

    If the action is 'enqueue', the request must also have a 'url' attribute.
    The 'url' attribute, when provided, must be a URL to a file that meets
    these requirements:
        * The URL must be a publicly accessible URL
        * The URL must be a file that ends in '.exe' or '.dmg'
        * The URL must be located in either the releases bucket, or else
            in the dev builds bucket.  It doesn't necessarily have to be an
            InVEST binary.
        * The URL must be a file that is not older than June 1, 2024
        * The URL must be a file that is not already in the queue
        * The URL should be a file that is not already signed (if the file has
            already been signed, its signature will be overwritten)
    """
    data = request.get_json()
    if data['token'] != os.environ['ACCESS_TOKEN']:
        logging.info('Rejecting request due to invalid token')
        return jsonify('Invalid token'), 403

    if request.method != 'POST':
        logging.info('Rejecting request due to invalid HTTP method')
        return jsonify('Invalid request method'), 405

    storage_client = storage.Client()
    bucket = storage_client.bucket(CODESIGN_DATA_BUCKET)

    logging.debug(f'Data POSTed: {data}')

    if data['action'] == 'dequeue':
        with get_lock():
            queuefile = bucket.blob('queue.json')
            queue_dict = json.loads(queuefile.download_as_string())
            try:
                next_file_url = queue_dict['queue'].pop(0)
            except IndexError:
                # No items in the queue!
                logging.info('No binaries are currently queued for signing')
                return jsonify('No items in the queue'), 204

            queuefile.upload_from_string(json.dumps(queue_dict))

        data = {
            'https-url': next_file_url,
            'basename': os.path.basename(next_file_url),
            'gs-uri': unquote(next_file_url.replace(
                f'{GOOGLE_PREFIX}/', 'gs://')),
        }
        logging.info(f'Dequeued {next_file_url}')
        return jsonify(data)

    elif data['action'] == 'enqueue':
        url = data['url']
        logging.info(f'Attempting to enqueue url {url}')

        if not url.endswith(('.exe', '.dmg')):
            logging.info("Rejecting URL because it doesn't end in .exe or .dmg")
            return jsonify('Invalid URL to sign'), 400

        if not url.startswith(GOOGLE_PREFIX):
            logging.info(f'Rejecting URL because it does not start with {GOOGLE_PREFIX}')
            return jsonify('Invalid host'), 400

        if not url.startswith((
                f'{GOOGLE_PREFIX}/releases.naturalcapitalproject.org/',
                f'{GOOGLE_PREFIX}/natcap-dev-build-artifacts/')):
            logging.info('Rejecting URL because the bucket is incorrect')
            return jsonify("Invalid target bucket"), 400

        # Remove http character quoting
        url = unquote(url)

        binary_bucket_name, *binary_obj_paths = url.replace(
            GOOGLE_PREFIX + '/', '').split('/')
        codesign_bucket = storage_client.bucket(CODESIGN_DATA_BUCKET)

        # If the file does not exist at this URL, reject it.
        response = requests.head(url)
        if response.status_code >= 400:
            logging.info('Rejecting URL because it does not exist')
            return jsonify('Requested file does not exist'), 403

        # If the file is too old, reject it.  Trying to avoid a
        # denial-of-service by invoking the service with very old files.
        # I just pulled June 1 out of thin air as a date that is a little while
        # ago, but not so long ago that we could suddenly have many files
        # enqueued.
        mday, mmonth, myear = response.headers['Last-Modified'].split(' ')[1:4]
        modified_time = datetime.datetime.strptime(
            ' '.join((mday, mmonth, myear)), '%d %b %Y')
        if modified_time < datetime.datetime(year=2024, month=6, day=1):
            logging.info('Rejecting URL because it is too old')
            return jsonify('File is too old'), 400

        response = requests.head(f'{url}.signature')
        if response.status_code != 404:
            logging.info('Rejecting URL because it has already been signed.')
            return jsonify('File has already been signed'), 204

        with get_lock():
            # Since the file has not already been signed, add the file to the
            # queue
            queuefile = codesign_bucket.blob('queue.json')
            if not queuefile.exists():
                queue_dict = {'queue': []}
            else:
                queue_dict = json.loads(queuefile.download_as_string())

            if url not in queue_dict['queue']:
                queue_dict['queue'].append(url)
            else:
                return jsonify(
                    'File is already in the queue', 200, 'application/json')

            queuefile.upload_from_string(json.dumps(queue_dict))

        logging.info(f'Enqueued {url}')
        return jsonify("OK"), 200

    else:
        return jsonify('Invalid action request'), 405
