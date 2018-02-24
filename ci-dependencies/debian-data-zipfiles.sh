#!/usr/bin/env sh

######################################################################
# @file        : build_install_zipfiles
# @description : Install apt packages to be able to build sampledata zipfiles. 
#                Assumes debian:stretch.
######################################################################

apt-get update && apt-get install -y \
    subversion=1.9.5-1+deb9u1 \
    zip=3.0-11+b1 \
    make=4.1-9.1 \
    mercurial=4.0-1+deb9u1
