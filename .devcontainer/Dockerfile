FROM docker.io/ros:jazzy

ARG ROS_DISTRO
ENV ROS_DISTRO=$ROS_DISTRO
ARG USERNAME=okmr
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install all system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    cmake \
    git \
    libpcl-dev \
    libglm-dev \
    libsdl2-dev \
    python3-scipy \
    python3-transitions \
    python3-rosdep \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN sudo rosdep init || true
RUN sudo rosdep fix-permissions

USER $USERNAME
WORKDIR /home/$USERNAME
ENV SHELL=/bin/bash

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc
