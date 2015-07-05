#!/bin/bash
#
# Script to build the api documentation.
#
# Execute this from the repository root:
#     ./jenkins/api-docs.sh 
#
# Arguments:
#   -e envname  : The name of the environment to use.


while getopts ":e:" opt 
do
    case $opt in 
        e) ENV="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG"
        ;;
    esac
done

# If -e was not provided, assume default environment name.
if [ "$ENV" = "" ]
then
    ENV=docenv
fi

paver env --system-site-packages --with-invest --envname=$ENV
source $ENV/bin/activate
pip install -r requirements-docs.txt --force-reinstall --upgrade
python setup.py build_sphinx

