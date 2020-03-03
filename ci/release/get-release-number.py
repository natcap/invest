import re
import subprocess


def incrememt_bugfix_version():
    process_data = subprocess.run(
        ['python', 'setup.py', '--version'], capture_output=True)
    version_string = process_data.stdout.rstrip().decode('UTF-8')

    major, minor, bugfix, remainder = version_string.split('.', maxsplit=4)

    return f'{major}.{minor}.{int(bugfix)+1}'


if __name__ == '__main__':
    print(incrememt_bugfix_version())
