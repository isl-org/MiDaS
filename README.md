## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

MiDaS v2.1 was trained on 10 datasets (ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir, HRWSI, ApolloScape, BlendedMVS, IRS) with
multi-objective optimization. 
The original model that was trained on 5 datasets  (`MIX 5` in the paper) can be found [here](https://github.com/intel-isl/MiDaS/releases/tag/v2).


### Changelog 
* [Nov 2020] Released MiDaS v2.1:
	- New model that was trained on 10 datasets and is on average about [10% more accurate](#Accuracy) than [MiDaS v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2)
	- New light-weight model that achieves [real-time performance](https://github.com/intel-isl/MiDaS/tree/master/mobile) on mobile platforms.
	- Sample applications for [iOS](https://github.com/intel-isl/MiDaS/tree/master/mobile/ios) and [Android](https://github.com/intel-isl/MiDaS/tree/master/mobile/android)
	- [ROS package](https://github.com/intel-isl/MiDaS/tree/master/ros) for easy deployment on robots
* [Jul 2020] Added TensorFlow and ONNX code. Added [online demo](http://35.202.76.57/).
* [Dec 2019] Released new version of MiDaS - the new model is significantly more accurate and robust
* [Jul 2019] Initial release of MiDaS ([Link](https://github.com/intel-isl/MiDaS/releases/tag/v1))

### Online demo

An online demo of the model is available: http://35.202.76.57/

Please be patient. Inference might take up to 30 seconds due to hardware restrictions.

### Setup 

1) Download the model weights [model-f6b98070.pt](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt) 
and [model-small-70d6b9c8.pt](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt) and place the
file in the root folder.

2) Set up dependencies: 

    ```shell
    conda install pytorch torchvision opencv
    ```

   The code was tested with Python 3.7, PyTorch 1.7.0, and OpenCV 4.4.0.

    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the model:

    ```shell
    python run.py
    ```

    Or run the small model:

    ```shell
    python run.py --model_weights model-small-70d6b9c8.pt --model_type small
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

#### via Mobile (iOS / Android)

See [README](https://github.com/intel-isl/MiDaS/tree/master/mobile) in the `mobile` subdirectory.

#### via ROS1 (Robot Operating System)

See [README](https://github.com/intel-isl/MiDaS/tree/master/ros) in the `ros` subdirectory.


### Accuracy

Zero-shot error (the lower - the better) and speed (FPS):

| Model |  DIW, WHDR | Eth3d, AbsRel | Sintel, AbsRel | Kitti, δ>1.25 | NyuDepthV2, δ>1.25 | TUM, δ>1.25 | Speed, FPS |
|---|---|---|---|---|---|---|---|
| **Small models:** | | | | | | | iPhone 11 |
| MiDaS v2 small | **0.1248** | 0.1550 | **0.3300** | **21.81** | 15.73 | 17.00 | 0.6 |
| MiDaS v2.1 small [URL](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt) | 0.1344 | **0.1344** | 0.3370 | 29.27 | **13.43** | **14.53** | 30 |
| Relative improvement | -7.7% | **+13.3%** | -2.1% | -34.2% | **+14.6%** | **+14.5%** | **50x** |
| | | | | | | |
| **Big models:** | | | | | | | GPU RTX 2080Ti |
| MiDaS v2 large [URL](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt) | **0.1246** | 0.1290 | **0.3270** | 23.90 | 9.55 | 14.29 | 59 |
| MiDaS v2.1 large [URL](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt) | 0.1295 | **0.1155** | 0.3285 | **16.08** | **8.71** | **12.51** | 59 |
| Relative improvement | -3.9% | **+10.5%** | -0.52% | **+32.7%** | **+8.8%** | **+12.5%** | 1x |


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
