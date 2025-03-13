FROM quay.io/pypa/manylinux_2_34_x86_64

RUN useradd runner
RUN usermod -aG wheel runner
RUN yum install -y sudo jq proj
WORKDIR /home/runner/invest

COPY ../../../.. .
RUN chown -R runner /home/runner

# Install the latest release of git lfs
ARG GIT_LFS_VERSION=v3.6.1
RUN curl -LO https://github.com/git-lfs/git-lfs/releases/latest/download/git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN tar -xvzf git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN rm git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN cd git-lfs-* && ./install.sh

# ARG GDAL_VERSION=3.10.2
# RUN curl -LO https://github.com/OSGeo/gdal/releases/download/v$GDAL_VERSION/gdal-$GDAL_VERSION.tar.gz
# RUN tar -xvzf gdal-$GDAL_VERSION.tar.gz
# RUN rm gdal-$GDAL_VERSION.tar.gz
# RUN cd gdal-$GDAL_VERSION && \
#     mkdir build && cd build && \
#     cmake .. && \
#     cmake --build . && \
#     cmake --build . --target install

USER runner
