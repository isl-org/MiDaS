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
RUN pip3 install torch~=1.8 torchvision opencv-python~=3.4 timm

# copy inference code
WORKDIR /opt/MiDaS
COPY ./midas ./midas
COPY ./*.py ./

# download model weights so the docker image can be used offline
RUN cd weights && {curl -OL https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/dpt_hybrid-midas-501f0c75.pt; cd -; }
RUN python3 run.py --model_type dpt_hybrid; exit 0

# entrypoint (dont forget to mount input and output directories)
CMD python3 run.py --model_type dpt_hybrid
