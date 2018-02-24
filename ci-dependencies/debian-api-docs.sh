#!/usr/bin/env sh

######################################################################
# @description : Install apt packages to build API documentation.
#                Assumes we're running on debian:stretch.
######################################################################

apt-get update && apt-get install -y \
    zip=3.0-11+b1 \
    make=4.1-9.1 \
    mercurial=4.0-1+deb9u1 \
    pandoc=1.17.2~dfsg-3 \
    build-essential=12.3 \
    python-sphinx=1.4.9-2 \
    python-setuptools=33.1.1-1 \
    cython=0.25.2-1 \
    python-numpy=1:1.12.1-3 \
    python-scipy=0.18.1-2 \
    python-matplotlib=2.0.0+dfsg1-2 \
    python-shapely=1.5.17-1 \
    python-rtree=0.8.3+ds-1 \
    python-gdal=2.1.2+dfsg-5 \
    python-sphinx-rtd-theme=0.1.9-1 \
    python-mock=2.0.0-3
