#!/usr/bin/env sh

######################################################################
# @description : Install apt packages to build HTML, PDF userguides.
#                Assumes we're running on debian:stretch.
######################################################################

apt-get update && apt-get install -y \
    make=4.1-9.1 \
    mercurial=4.0-1+deb9u1 \
    pandoc=1.17.2~dfsg-3 \
    python-setuptools=33.1.1-1 \
    cython=0.25.2-1 \
    python-numpy=1:1.12.1-3 \
    python-sphinx=1.4.9-2 \
    texlive-latex-base=2016.20170123-5 \
    texlive-generic-extra=2016.20170123-5 \
    texlive-latex-recommended=2016.20170123-5 \
    texlive-fonts-recommended=2016.20170123-5 \
    texlive-latex-extra=2016.20170123-5 \
    python-pip=9.0.1-2 \
    dvipng=1.14-2+b3

pip install setuptools_scm==1.15.7
    

