# This script determines the filename of the binary that is built on the
# current operating system and enqueues it for code signing.

import os
import platform
import subprocess
import sys

# setuptools_scm is required, but not directly imported here


def main():
    # Calling this from the CLI makes sure that our pyproject.toml config is
    # parsed, which isn't the case by default when setuptools_scm.version() is
    # called.
    version = subprocess.run(
        [sys.executable, '-m', 'setuptools_scm'],
        capture_output=True, check=True).stdout.decode('utf-8').strip()

    this_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(this_dir)
    dist_url_proc = subprocess.run(
        ['make', '-C', repo_root, '--no-print-directory',
         'print-DIST_URL_BASE'],
        capture_output=True, check=True)
    dist_url = dist_url_proc.stdout.decode('utf-8').strip().split(' ')[2]

    os_alias = platform.system().lower()
    if os_alias == 'windows':
        os_alias = 'win32'
        arch = 'x64'
        ext = 'exe'

    elif os_alias == 'darwin':
        if platform.machine() == 'arm64':
            arch = 'arm64'
        else:
            arch = 'x64'
        ext = 'dmg'

    else:
        raise RuntimeError(f'Unsupported platform: {os_alias}')

    url = (
        f"{dist_url}/workbench/"
        f"invest_{version}_workbench_{os_alias}_{arch}.{ext}"
    )
    print(f"Enqueueing URL for signing: {url}")
    subprocess.run(
        ['python', os.path.join(this_dir, 'enqueue-binary.py'), url],
        check=True)


if __name__ == '__main__':
    main()
