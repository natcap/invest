# syntax=docker/dockerfile:1

# Create the container for distribution that has runtime dependencies.
FROM mambaorg/micromamba:2.5-debian13-slim

ARG PYTHON_VERSION="3.13"

ARG GDAL_VERSION="3.10"

# Activate the base environment in subsequent commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# only copy in the files needed for building the wheel
COPY --chown=$MAMBA_USER:$MAMBA_USER src /invest/src
ADD pyproject.toml /invest/pyproject.toml
ADD setup.py /invest/setup.py
ADD requirements.txt /invest/requirements.txt
ADD LICENSE.txt /invest/LICENSE.txt
ADD NOTICE.txt /invest/NOTICE.txt
COPY --chown=$MAMBA_USER:$MAMBA_USER .git /invest/.git

# The environment.yml file will be built during github actions.
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml

WORKDIR /invest

RUN micromamba install -y -n base --override-channels -c conda-forge -f /tmp/environment.yml \
        python==${PYTHON_VERSION} gdal=${DEBIAN_GDAL_VERSION} cxx-compiler git && \
    micromamba list && \
    echo $CONDA_PREFIX && \
    which python $$ \
    pip install uv && \
    CONDA_PREFIX=/opt/conda uv pip install . --python $CONDA_PREFIX/bin/python && \
    uv cache clean && \
    micromamba clean --all --yes && \
    micromamba remove -y -n base cxx-compiler git

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
