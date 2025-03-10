FROM quay.io/pypa/manylinux_2_34_x86_64

RUN useradd runner
RUN usermod -aG wheel runner
RUN yum install -y sudo jq
WORKDIR /home/runner/invest
RUN chown -R runner /home/runner

COPY ../../../.. .
RUN echo $(ls)

# RUN USERNAME=runner
# RUN USER_UID=1000
# RUN USER_GID=$USER_UID
# RUN groupadd --gid $USER_GID $USERNAME
# RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
# RUN chmod 0440 /etc/sudoers.d/$USERNAME

# actions/checkout won't check out to a directory outside of /__w/invest,
# so we have to copy it afterwards


# Install the latest release of git lfs
ARG GIT_LFS_VERSION=v3.6.1
RUN curl -LO https://github.com/git-lfs/git-lfs/releases/latest/download/git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN tar -xvzf git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN rm git-lfs-linux-amd64-$GIT_LFS_VERSION.tar.gz
RUN cd git-lfs-* && ./install.sh

USER runner