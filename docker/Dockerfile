# syntax=docker/dockerfile:1

# Build the InVEST wheel in a separate container stage
FROM debian:12.2 as build
RUN apt update && apt install python3 python3-dev python3-pip python3-build build-essential
RUN python3 -m build && pwd

# Create the container for distribution that has runtime dependencies.
FROM mambaorg/micromamba:1.5.0-bookworm-slim
COPY --from=build /dist/*.whl /tmp/

# The environment.yml file will be built during github actions.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -c conda-forge -f /tmp/environment.yml && \
        micromamba clean --all --yes && \
        /opt/conda/bin/python -m pip install /tmp/*.whl

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

