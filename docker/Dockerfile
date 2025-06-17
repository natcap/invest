# syntax=docker/dockerfile:1

# Build the InVEST wheel in a separate container stage
FROM debian:12.11 AS build
RUN apt update && apt install -y python3 python3-dev python3-pip python3-build build-essential git python3.11-venv libgdal-dev

# only copy in the files needed for building the wheel
ADD src /invest/src
ADD pyproject.toml /invest/pyproject.toml
ADD setup.py /invest/setup.py
ADD requirements.txt /invest/requirements.txt
ADD .git /invest/.git
RUN cd /invest && python3 -m build

# Create the container for distribution that has runtime dependencies.
FROM mambaorg/micromamba:2.1.1-debian12-slim
# Python version should match the version used in stage 1.
# If we update the stage 1 debian version, also update this python version
ARG PYTHON_VERSION="3.11"
COPY --from=build /invest/dist/*.whl /tmp/


# Define this to be the GDAL version used to build the InVEST wheel in the debian step.
# Check the debian package repo for the debian release above to confirm the version.
# Debian's GDAL version will undoubtedly be earlier than what's available in conda-forge.
ARG DEBIAN_GDAL_VERSION="3.6"

# The environment.yml file will be built during github actions.
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base --override-channels -c conda-forge python==${PYTHON_VERSION} gdal=${DEBIAN_GDAL_VERSION} && \
        micromamba install -y -n base --override-channels -c conda-forge -f /tmp/environment.yml && \
        micromamba clean --all --yes && \
        /opt/conda/bin/python -m pip install /tmp/*.whl && \
        /opt/conda/bin/python -m pip cache purge && \
        micromamba remove -y -n base cxx-compiler git

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
