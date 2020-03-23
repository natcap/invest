import os
import subprocess
import unittest


def fetch_version_from_setup():
    """Fetch the next version string from setup.py.

    Returns:
        The UTF-8 encoded string returned by calling ``python setup.py
        --version``.

    """
    setup_py_location = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'setup.py'))
    process_data = subprocess.run(
        ['python', setup_py_location, '--version'], capture_output=True)
    return process_data.stdout.rstrip().decode('UTF-8')


def increment_bugfix_version(post_version_string):
    """Increment the bugfix version number.

    This function takes the user-defined version string and returns the next
    bugfix version release version string.

    Note:
        This approach requires that a post-versioning scheme is used.  If
        ``post`` is not present within the provided version string, an error
        will be raised.

    Parameters:
        post_version_string (string): The post-versioned string to parse and
            increment.

    Returns:
        The version string of the next bugfix release.

    """
    major, minor, bugfix, remainder = post_version_string.split(
        '.', maxsplit=4)
    return f'{major}.{minor}.{int(bugfix)+1}'


class IncrementerTests(unittest.TestCase):
    def test_increment(self):
        """Test the incrementing function works as expected."""
        self.assertEqual(
            increment_bugfix_version('1.2.3.post123+gabcdef1234'),
            '1.2.4')

    def test_error_on_no_post(self):
        """Test that there's an error when not given a post-version."""
        with self.assertRaises(Exception):
            increment_bugfix_version('1.2.3')

    def test_setup_version(self):
        """Test we can get the version from setup.py."""
        version_string = fetch_version_from_setup()
        self.assertTrue(isinstance(version_string, str))

    def test_setup_version_chdir(self):
        """Test that we can get version from setup.py from elsewhere."""
        try:
            current_dir = os.getcwd()
            os.chdir('tests')
            version_string = fetch_version_from_setup()
            self.assertTrue(isinstace(version_string), str)
        except Exception:
            os.chdir(current_dir)


if __name__ == '__main__':
    print(increment_bugfix_version(fetch_version_from_setup()))
