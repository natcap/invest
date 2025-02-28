#!/usr/bin/env sh
#
# This script determines the filename of the binary that is built on the
# current operating system and enqueues it for code signing.
#
# NOTE: this script must be run from the directory containing this script.

version=$(python -m setuptools_scm)
url_base=$(make -C .. --no-print-directory print-DIST_URL_BASE | awk ' { print $3 } ')
platform=$(python -c "import platform;p=platform.system().lower();print(p if p != 'windows' else 'win32')")

if [ "$platform" = "win32" ]; then
    url="${url_base}/workbench/invest_${version}_workbench_${platform}_x64.exe"
elif [ "$platform" = "darwin" ]; then
    architecture=$(python -c "import platform;print(platform.machine())")
    if [ "$architecture" = "arm64" ]; then
        url="${url_base}/workbench/invest_${version}_workbench_${platform}_arm64.dmg"
    else
        url="${url_base}/workbench/invest_${version}_workbench_${platform}_x64.dmg"
    fi
else
    echo "Unsupported platform: ${platform}"
    exit 1
fi


echo "Enqueuing URL ${url}"
python enqueue-binary.py "${url}"
