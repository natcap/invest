# encoding=UTF-8
"""Increment the User's Guide revision in Makefile.

This script fetches the latest user's guide revision on the master branch from
the github repository and updates it in the Makefile.

To invoke:
    $ pip install requests
    $ python ci/release/increment-userguide-revision.py
"""

import shutil
import os
import requests

API_TARGET = (
    'https://api.github.com/repos/natcap/invest.users-guide/commits/master')


def update_userguide_rev_in_makefile():
    """Update the userguide revision in the Makefile.

    Fetches the latest user's guide revision from the master branch and updates
    it in the Makefile.

    Returns:
        ``None``

    """
    req = requests.get(API_TARGET)
    new_rev = req.json()['sha']

    if not os.path.exists('build'):
        os.makedirs('build')
    new_makefile_path = os.path.join('build', 'Makefile')
    makefile_path = 'Makefile'

    # The Makefile uses \n newlines, so enforce that newline character in the
    # new Makefile.
    with open(new_makefile_path, 'w', newline='\n') as new_makefile:
        with open(makefile_path) as makefile:
            for line in makefile:
                if line.startswith('GIT_UG_REPO_REV'):
                    line_prefix, old_sha = line.split(':=')
                    print('Replacing Makefile SHA %s with %s' % (
                        old_sha.rstrip(), new_rev))
                    new_makefile.write('%s:= %s\n' % (
                        line_prefix, new_rev))
                else:
                    new_makefile.write(line)

    shutil.copyfile(new_makefile_path, makefile_path)


if __name__ == '__main__':
    update_userguide_rev_in_makefile()
