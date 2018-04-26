#!/usr/bin/env bash

# Execute the jenkins build on a mac.
#
# This script takes no parameters
#
# This script assumes that it will be called from the repo root through the
# ``make jenkins`` target on a mac.

make env
env/bin/python.exe -m pip install --upgrade .
make PYTHON=env/bin/python mac_installer userguide 
env/bin/python setup.py bdist_wheel
env/bin/python scripts/jenkins_push_artifacts.py


