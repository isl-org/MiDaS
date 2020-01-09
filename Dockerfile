FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
WORKDIR /root
RUN apt-get update
RUN apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev -y
RUN pip install opencv-python==3.4.3.18
CMD ["python", "/root/run.py"]
