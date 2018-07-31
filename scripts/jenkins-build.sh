#!/usr/bin/env bash -e

# Execute the jenkins build on a mac.
#
# This script takes no parameters
#
# This script assumes that it will be called from the repo root through the
# ``make jenkins`` target on a mac.

make env
env/bin/python -m pip install --upgrade .
env/bin/python -m pip install -r requirements-gui.txt
make PYTHON=env/bin/python mac_zipfile userguide 
env/bin/python setup.py bdist_wheel
env/bin/python scripts/jenkins_push_artifacts.py
