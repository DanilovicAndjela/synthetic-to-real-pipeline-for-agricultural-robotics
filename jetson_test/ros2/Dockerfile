FROM ros:humble-ros-base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade Python packaging tools.
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==11.3.0 \
    opencv-python-headless==4.11.0.86

# TensorRT Python API.
RUN python3 -m pip install --no-cache-dir \
    tensorrt-cu13-bindings==10.16.1.11 && \
    python3 -m pip install --no-cache-dir --no-deps \
    tensorrt-cu13==10.16.1.11

WORKDIR /ros2_ws

COPY src /ros2_ws/src

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

CMD ["bash"]
