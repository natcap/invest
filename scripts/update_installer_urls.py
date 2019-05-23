# open the json copied from bucket
# parse the installer string, grep for post to see if it's a dev build
# if dev: compare to version string already in the json file's dev key
# if release: compare to version string already in json file's windows key
# if new version > existing version, write new string to json

# do this for....the windows installer filename from make? any .exe or .zip in the dist dir?


import argparse
import json
import re
import os
from pkg_resources import parse_version

from pprint import pprint

def main(json_file, outgoing_filenames, dist_url_base):
    """Key a download location to a static url fragment identifier."""

    version_re = {
        'windows': 'InVEST_(.+?)_x86_Setup.exe$',
        'mac': 'InVEST-(.+?)-mac.zip$'}
    public_url_base = dist_url_base.replace('gs://', 'http://')

    with open(json_file) as file:
        lookup = json.load(file)
        pprint(lookup)
        print '\n'

    for outgoing_filename in outgoing_filenames:
        # import pdb; pdb.set_trace()
        if re.search('post', outgoing_filename):
            branch = 'dev'
        else:
            branch = ''

        if re.search('.*exe$', outgoing_filename):
            opsys = 'windows'
            match = re.search(version_re[opsys], outgoing_filename)
            if match:
                outgoing_version = match.group(1)

        elif re.search('.*zip$', outgoing_filename):
            opsys = 'mac'
            match = re.search(version_re[opsys], outgoing_filename)
            if match:
                outgoing_version = match.group(1)

        frag_id = '-'.join(['#latest', 'invest', opsys])
        if branch:
            frag_id = '-'.join([frag_id, branch])
        if frag_id in lookup:
            existing_filename = os.path.basename(lookup[frag_id])
            match = re.search(version_re[opsys], existing_filename)
            if match:
                existing_version = match.group(1)

            if parse_version(outgoing_version) > parse_version(existing_version):
                lookup[frag_id] = '/'.join((public_url_base, outgoing_filename))
        else:
            print('fragment identifier %s is not in the lookup and will not be created' % frag_id)
            continue

    with open(json_file, 'wb') as file:
        json.dump(lookup, file)
    pprint(lookup)



def test_main():
    import tempfile
    lookup = {
      "#latest-invest-windows": "http://releases.naturalcapitalproject.org/invest/3.7.0/InVEST_3.7.0_x86_Setup.exe",
      "#latest-invest-windows-dev": "http://releases.naturalcapitalproject.org/invest/3.7.0/InVEST_3.7.0_x86_Setup.exe",
      "#latest-invest-mac": "http://releases.naturalcapitalproject.org/invest/3.7.0/InVEST-3.7.0-mac.zip",
      "#latest-invest-userguide": "http://releases.naturalcapitalproject.org/invest-userguide/latest/",
    }
    dist_url_base = 'gs://releases.naturalcapitalproject.org/invest/9.9.9'  # this number doesn't matter
    public_url_base = dist_url_base.replace('gs://', 'http://')

    def get_updated_lookup(outgoing_filename):
        f, fname = tempfile.mkstemp()
        os.close(f)
        with open(fname, 'wb') as file:
            json.dump(lookup, file)

        main(fname, [outgoing_filename], dist_url_base)
        with open(fname, 'r') as file:
            updated_lookup = json.load(file)
        return updated_lookup


    # when version # is greater, assert lookup value == outgoing filename
    outgoing_filename = "InVEST_3.8.0_x86_Setup.exe"
    updated_lookup = get_updated_lookup(outgoing_filename)
    assert(updated_lookup["#latest-invest-windows"] == '/'.join((public_url_base, outgoing_filename)))

    # when version # is less than, assert lookup value == same as original
    updated_lookup = get_updated_lookup("InVEST_3.6.0_x86_Setup.exe")
    assert(updated_lookup["#latest-invest-windows"] == lookup["#latest-invest-windows"])

    # when version # is ==, assert lookup value == same as original
    updated_lookup = get_updated_lookup("InVEST_3.7.0_x86_Setup.exe")
    assert(updated_lookup["#latest-invest-windows"] == lookup["#latest-invest-windows"])

    # when outgoing is a release and more recent than latest dev, compare and update the dev too.
    outgoing_filename = "InVEST_3.7.0_x86_Setup.exe"
    updated_lookup = get_updated_lookup(outgoing_filename)
    assert(updated_lookup["#latest-invest-windows-dev"] == '/'.join((public_url_base, outgoing_filename)))

    # when outgoing name includes 'post', assert chosen lookup key includes -dev
    # when outgoing name excludes 'post', assert chosen lookup key excludes -dev

    # when outgoing name includes exe, assert chosen lookup key includes windows
    # when outgoing name includes zip, assert chosen lookup key includes mac

    # what if outgoing filename does not match any of the regex?
    # what are all the outcomes of parse_version? what if fails to parse?

    # test with single outgoing file and multiple
    # main(fname, (outgoing_filename), dist_url_base)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('download_url', help='')
    # args = parser.parse_args()

    # $(WINDOWS_ISNTALLER_FILE)
    # outgoing_filename = "InVEST_3.7.0.post13+h97f55543230e.d20190521_x86_Setup.exe"
    # # $(DIST_URL_BASE)
    # dist_url_base = 'gs://releases.naturalcapitalproject.org/invest/3.7.0.post13+h97f55543230e.d20190522'
    # json_filename = "../latest.json"
    # main(json_filename, (outgoing_filename), dist_url_base)
    test_main()