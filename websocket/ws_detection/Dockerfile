# Based on https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.0-cuda11.3-ubuntu20.04/Dockerfile

# Use nvidia/cuda image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV APP_ROOT /src
ENV CONFIG_ROOT /config
# Set non interactive for avoiding any question
ENV TZ=Europe/Paris \
    DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED definitely

RUN mkdir ${CONFIG_ROOT}
RUN mkdir ${APP_ROOT}

# Copy the current directory contents into the container at /src
ADD . ${APP_ROOT}

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    rm -rf /var/lib/apt/lists/* && \
    # Install gcc and other build essentials for compiling dependencies
    apt-get update && apt-get install -y \
    build-essential \
    python3.9-dev \
    # Additional dependencies for pycocotools and OpenCV
    libpython3.9-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    # Install OpenGL libraries
    libgl1-mesa-glx

# Copy requirements
COPY requirements.txt ${CONFIG_ROOT}/requirements.txt

# Install all packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install uvicorn[standard]
RUN pip install -r ${CONFIG_ROOT}/requirements.txt


# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

WORKDIR ${APP_ROOT}