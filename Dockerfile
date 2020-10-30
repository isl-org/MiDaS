# enables cuda support in docker
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# install python 3.6, pip and requirements for opencv-python 
# (see https://github.com/NVIDIA/nvidia-docker/issues/864)
RUN apt-get update && apt-get -y install \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip3 install torch~=1.2 torchvision opencv-python~=3.4

# copy inference code
WORKDIR /opt/MiDaS
COPY ./midas ./midas
COPY ./*.py ./

# download model weights so the docker image can be used offline
RUN curl -OL https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt
RUN python3 run.py; exit 0

# entrypoint (dont forget to mount input and output directories)
CMD python3 run.py
