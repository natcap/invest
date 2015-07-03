#!/bin/bash
#
# Jenkins build script for posix systems
# Necessary because controlling a virtual environment doesn't work
# so well within a python session

ENV=release_env
source $ENV/bin/activate
#paver build

