FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
WORKDIR /root
RUN pip install opencv-python