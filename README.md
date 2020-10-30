## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

The pre-trained model corresponds to `MIX 5` with multi-objective optimization enabled.

### Changelog 
* [Jul 2020] Added TensorFlow and ONNX code. Added [online demo](http://35.202.76.57/).
* [Dec 2019] Released new version of MiDaS - the new model is significantly more accurate and robust
* [Jul 2019] Initial release of MiDaS ([Link](https://github.com/intel-isl/MiDaS/releases/tag/v1))

### Online demo

An online demo of the model is available: http://35.202.76.57/

Please be patient. Inference might take up to 30 seconds due to hardware restrictions.

### Setup 

1) Download the model weights [model-f45da743.pt](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt) and place the
file in the root folder.

2) Set up dependencies: 

    ```shell
    conda install pytorch torchvision opencv
    ```

   The code was tested with Python 3.7, PyTorch 1.2.0, and OpenCV 3.4.2.

    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the model:

    ```shell
    python run.py
    ```

3) The resulting inverse depth maps are written to the `output` folder.


#### via Docker

1) Make sure you have installed Docker and the
   [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-\(Native-GPU-Support\)).

2) Build the Docker image:

    ```shell
    docker build -t midas .
    ```

3) Run inference:

    ```shell
    docker run --rm --gpus all -v $PWD/input:/opt/MiDaS/input -v $PWD/output:/opt/MiDaS/output midas
    ```

   This command passes through all of your NVIDIA GPUs to the container, mounts the
   `input` and `output` directories and then runs the inference.

#### via PyTorch Hub

The pretrained model is also available on [PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)

#### via TensorFlow or ONNX

See [README](https://github.com/intel-isl/MiDaS/tree/master/tf) in the `tf` subdirectory.

### Citation

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```


### License 

MIT License 
