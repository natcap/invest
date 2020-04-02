# encoding=UTF-8
import argparse
import mimetypes
import os
import logging
logging.basicConfig(level=logging.DEBUG)

import github3
import retrying

# The full list of standard media types can be found at
# https://www.iana.org/assignments/media-types/media-types.xhtml
MIME_TYPE_BY_FILE_EXT = {
    'zip': 'application/zip',
    'exe': 'application/vnd.microsoft.portable-executable',
    'gz': 'application/gzip',

    # No standard for Wheels yet.  See this stackoverflow post for
    # the suggestion used here: # https://stackoverflow.com/a/58543864/299084
    'whl': 'application/x-wheel+zip'
}

LOGGER = logging.getLogger(__name__)


def upload_file(repo, tagname, token, filepaths):
    session = github3.GitHub(token=token)
    repository = session.repository(*repo.split('/'))
    release = repository.release_from_tag(tagname)

    files_with_unknown_filetypes = []
    for filepath in filepaths:
        # If we don't know the filetype of the file, guess via python
        # mimetypes library.  If it's still not known, raise an error later.
        # GitHub requires that the MIME type be set by the user.
        try:
            # Get the extension without the leading "."
            extension = os.path.splitext(filepath)[1].lower()[1:]
            content_type = MIME_TYPE_BY_FILE_EXT[extension]
        except KeyError:
            content_type, content_encoding = mimetypes.guess_type(filepath)
            if not content_type:
                files_with_unknown_filetypes.append(filepath)
                continue

        with retrying.Retrying(stop_max_attempt_number=5,
                               wait_exponential_multiplier=10000,
                               wait_exponential_max=10000):
            release.upload_asset(
                content_type=content_type,
                name=os.path.basename(filepath),
                asset=open(filepath, 'rb'),
            )

    if files_with_unknown_filetypes:
        raise ValueError(
            "Some file(s) being uploaded have unknown filetypes: %s" %
            files_with_unknown_filetypes)


def main(args=None):
    parser = argparse.ArgumentParser(description=(
        "Upload the target files to an existing github release. "
        "This script requires github3.py (pip install github3.py) to work."
    ))

    parser.add_argument('repo', help=(
        "The GitHub user/repo string that the target release is attached to. "
        "Example: 'natcap/invest'"))
    parser.add_argument('tagname', help="The tag of the target release.")
    parser.add_argument('token', help="The GitHub auth token to use.")
    parser.add_argument('filepaths', nargs='+', help=(
        "The files to upload to the target release.  There must be at least "
        "one file provided."))

    return parser.parse_args(args)


if __name__ == '__main__':
    upload_file(**main().__dict__)
