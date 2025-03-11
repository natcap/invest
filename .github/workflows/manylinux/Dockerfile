FROM quay.io/pypa/manylinux_2_34_x86_64

RUN useradd runner
RUN usermod -aG wheel runner
RUN yum install -y sudo jq
WORKDIR /home/runner/invest

COPY ../../../.. .
RUN chown -R runner /home/runner

# Install the latest release of git lfs
ARG GIT_LFS_VERSION=v3.6.1
RUN curl -LO https://github.com/git-lfs/git-lfs/releases/latest/download/git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN tar -xvzf git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN rm git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN cd git-lfs-* && ./install.sh

USER runner