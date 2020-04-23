import argparse
import json
import re
import os
from pkg_resources import parse_version
import tempfile
import unittest
from pprint import pprint


class InstallerURLTests(unittest.TestCase):
    """Tests for updating latest installer URLs."""

    def setUp(self):
        """Overriding setUp function to create temp file."""
        # this lets us delete the file after its done no matter the
        # the rest result
        self.lookup = {
            "#latest-invest-windows": "http://releases.naturalcapitalproject.org/invest/3.7.0/InVEST_3.7.0_x86_Setup.exe",
            "#latest-invest-windows-dev": "http://releases.naturalcapitalproject.org/invest/3.7.0.post525+h31b10cfee0d4/InVEST_3.7.0.post525+h31b10cfee0d4_x86_Setup.exe",
            "#latest-invest-mac": "http://releases.naturalcapitalproject.org/invest/3.7.0/InVEST-3.7.0-mac.zip",
        }
        self.dist_url_base = 'gs://releases.naturalcapitalproject.org/invest/9.9.9'  # this number doesn't matter
        self.public_url_base = self.dist_url_base.replace('gs://', 'http://')
        f, self.json_file = tempfile.mkstemp()
        os.close(f)

    def tearDown(self):
        """Overriding tearDown function to remove temporary file."""
        os.remove(self.json_file)

    def do_update(self, outgoing_filenames):
        """Helper for the tests that creates the initial lookup file
        and opens the udpated file for easy comparison back to original.

        Parameters:
            outgoing_filenames (list): A list of strings passed to main as
                the list of outgoing artifact filenames.

        Return: dict.

        """
        with open(self.json_file, 'wb') as file:
            json.dump(self.lookup, file)

        main(self.json_file, self.dist_url_base, outgoing_filenames)

        with open(self.json_file, 'r') as file:
            updated_lookup = json.load(file)
        return updated_lookup

    def test_newer_release(self):
        """Test outgoing release > than existing dev and release."""
        outgoing_filename = "InVEST_3.8.0_x86_Setup.exe"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup["#latest-invest-windows"], '/'.join((self.public_url_base, outgoing_filename)))
        self.assertEqual(updated_lookup["#latest-invest-windows-dev"], '/'.join((self.public_url_base, outgoing_filename)))
        self.assertEqual(updated_lookup["#latest-invest-mac"], self.lookup["#latest-invest-mac"])

    def test_older_release(self):
        """Test outgoing release < existing release and dev."""
        updated_lookup = InstallerURLTests.do_update(self, ["InVEST_3.6.0_x86_Setup.exe"])
        self.assertEqual(updated_lookup, self.lookup)

    def test_equal_versions(self):
        """Test outgoing release, existing release."""
        updated_lookup = InstallerURLTests.do_update(self, ["InVEST_3.7.0_x86_Setup.exe"])
        self.assertEqual(updated_lookup, self.lookup)

    def test_newer_dev(self):
        """Test outgoing dev > existing dev and release."""
        outgoing_filename = "InVEST_3.7.0.post9999+h31b10cfee0d4_x86_Setup.exe"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup["#latest-invest-windows"], self.lookup["#latest-invest-windows"])
        self.assertEqual(updated_lookup["#latest-invest-windows-dev"], '/'.join((self.public_url_base, outgoing_filename)))

    def test_older_dev(self):
        """Test outgoing dev < existing dev and release."""
        outgoing_filename = "InVEST_3.6.0.post9999+h31b10cfee0d4_x86_Setup.exe"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup, self.lookup)

    def test_mac(self):
        """Test outgoing mac release > existing release."""
        outgoing_filename = "InVEST-3.8.0-mac.zip"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup["#latest-invest-mac"], '/'.join((self.public_url_base, outgoing_filename)))
        self.assertEqual(updated_lookup["#latest-invest-windows-dev"], self.lookup["#latest-invest-windows-dev"])
        self.assertEqual(updated_lookup["#latest-invest-windows"], self.lookup["#latest-invest-windows"])

    def test_bogus_filename(self):
        """Test outgoing filename is totally bogus."""
        outgoing_filename = "adfadfa.zip"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup, self.lookup)

    def test_bogus_version(self):
        """Test outgoing filename looks valid but version string is bogus.

        There's actually no such thing as a bogus version in pkg_resources.parse_version.
        this example parses to < parse_version('0.0'), which is desireable.
        """
        outgoing_filename = "InVEST_???_x86_Setup.exe"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup, self.lookup)

    def test_multiple_artifacts(self):
        """Test with multiple outgoing files."""
        outgoing_mac = "InVEST-3.8.0-mac.zip"
        outgoing_windows = "InVEST_3.8.0_x86_Setup.exe"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_mac, outgoing_windows])
        self.assertEqual(updated_lookup["#latest-invest-mac"], '/'.join((self.public_url_base, outgoing_mac)))
        self.assertEqual(updated_lookup["#latest-invest-windows-dev"], '/'.join((self.public_url_base, outgoing_windows)))
        self.assertEqual(updated_lookup["#latest-invest-windows"], '/'.join((self.public_url_base, outgoing_windows)))

    def test_arch(self):
        """Test exe with an architecture different than existing exe."""
        outgoing_filename = "InVEST_3.8.0_x64_Setup.exe"
        updated_lookup = InstallerURLTests.do_update(self, [outgoing_filename])
        self.assertEqual(updated_lookup["#latest-invest-windows"], '/'.join((self.public_url_base, outgoing_filename)))
        self.assertEqual(updated_lookup["#latest-invest-windows-dev"], '/'.join((self.public_url_base, outgoing_filename)))
        self.assertEqual(updated_lookup["#latest-invest-mac"], self.lookup["#latest-invest-mac"])

    def test_string_instead_of_list(self):
        """Test passing a string instead of a list of outgoing filenames.

        Because of the use of argparse, this shouldn't be possible when
        running with command-line args, but it's an easy mistake otherwise..
        """
        updated_lookup = InstallerURLTests.do_update(self, "InVEST_3.6.0_x86_Setup.exe")
        self.assertEqual(updated_lookup, self.lookup)


def _is_dev_build(filename):
    x = False
    if re.search('post', filename):
        x = True
    return x


def main(json_file, dist_url_base, outgoing_filenames):
    """Index a download location for an S3 object to a static url fragment id.

    This function checks the existing lookup table of fragment ids and
    object names and compares version strings in existing object names
    with those in `outgoing_filenames`. If outgoing versions are more recent,
    the lookup table is updated with the more recent filenames.

    Parameters:
        json_file (string): path to json file holding the existing lookup
            of fragment id keys and download URL values.
        dist_url_base (string): public domain name of bucket where artifacts
            are deployed.
        outgoing_filenames (list): A list of strings passed to main as
            the list of outgoing artifact filenames.

    Side effects:
        If updates to the json_file are needed, json_file is overwritten
        with the updated lookup table.

    Returns:
        None

    """

    # To extract version strings from filenames and
    # validate filenames are installers.
    version_re = {
        'windows': 'InVEST_(.+?)_x\d\d_Setup.exe$',  # accepts x86 or x64
        'mac': 'InVEST-(.+?)-mac.zip$',
        }

    public_url_base = dist_url_base.replace('gs://', 'http://')

    with open(json_file) as file:
        lookup = json.load(file)

    if isinstance(outgoing_filenames, str):
        outgoing_filenames = [outgoing_filenames]

    for outgoing_filename in outgoing_filenames:
        # Which OS is the outgoing artifact, and is it a valid installer?
        outgoing_version = None
        for opsys in version_re:
            match = re.search(version_re[opsys], outgoing_filename)
            if match:
                outgoing_version = match.group(1)
                break

        if not outgoing_version:
            print('outgoing artifact %s does not look like an invest installer'
                  % outgoing_filename)
            continue

        for frag_id in lookup:
            # only compare versions for artifacts of same OS
            if opsys in frag_id:
                existing_filename = os.path.basename(lookup[frag_id])
                match = re.search(version_re[opsys], existing_filename)
                existing_version = match.group(1)

                if parse_version(outgoing_version) > parse_version(existing_version):

                    if not _is_dev_build(outgoing_filename):
                        # Official release, update both release and dev URLs
                        lookup[frag_id] = '/'.join(
                            (public_url_base, outgoing_filename))
                    else:
                        # Dev build, only update the dev URL
                        if _is_dev_build(existing_filename):
                            lookup[frag_id] = '/'.join(
                                (public_url_base, outgoing_filename))

    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(lookup, file)
    print('deployed artifacts with static redirect identifiers will include:')
    pprint(lookup)
    print('usage: http://releases.naturalcapitalproject.org/#latest-invest-windows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('json_file', type=str)
    parser.add_argument('bucket_url', type=str)
    parser.add_argument('artifacts', type=str, nargs='+')
    args = parser.parse_args()

    main(args.json_file, args.bucket_url, args.artifacts)
