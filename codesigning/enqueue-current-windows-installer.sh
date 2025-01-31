#!/usr/bin/env sh
#
# Run this script to enqueue the windows binary for this current version of the
# InVEST windows workbench installer for code signing.
#
# NOTE: this script must be run from the directory containing this script.

version=$(python -m setuptools_scm)
url_base=$(make -C .. --no-print-directory print-DIST_URL_BASE | awk ' { print $3 } ')
url="${url_base}/workbench/invest_${version}_workbench_win32_x64.exe"

echo "Enqueuing URL ${url}"
python enqueue-binary.py "${url}"
