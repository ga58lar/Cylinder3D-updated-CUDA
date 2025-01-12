FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    unzip \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libboost-all-dev \
    cmake \
    ninja-build \
    vim

RUN python3.8 -m pip install --upgrade pip setuptools wheel

# Install newer cmake
RUN wget https://apt.kitware.com/kitware-archive.sh && \
    chmod +x kitware-archive.sh && \
    ./kitware-archive.sh

# Install PyTorch with CUDA support first
RUN python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir 

# Set CUDA home and paths
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# Install basic dependencies
RUN python3.8 -m pip install \
    pyyaml==6.0 \
    strictyaml==1.6.1 \
    Cython==0.29.30 \
    tqdm==4.64.0

# Install spconv with CUDA support
RUN python3.8 -m pip install cumm-cu113==0.3.7 spconv-cu113==2.2.6

# Install torch-scatter with CUDA support
RUN FORCE_CUDA=1 python3.8 -m pip install --no-cache-dir torch-scatter \
    -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# Install remaining packages
RUN python3.8 -m pip install \
    nuscenes-devkit \
    numba==0.55.2

# Clear cache to reduce image size
RUN rm -rf /root/.cache/pip

WORKDIR /dev_ws