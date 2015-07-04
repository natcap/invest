#!/bin/bash
#
# Script to build the api documentation.
#
# Execute this from the repository root:
#     ./jenkins/api-docs.sh


ENV=doc_env
paver env --system-site-packages --with-invest --envname=$ENV
source $ENV/bin/activate
pip install -r requirements-docs.txt --force-reinstall --upgrade
python setup.py build_sphinx

