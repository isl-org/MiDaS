## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun


and our [preprint](https://arxiv.org/abs/2103.13413):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


MiDaS was trained on 10 datasets (ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir, HRWSI, ApolloScape, BlendedMVS, IRS) with
multi-objective optimization. 
The original model that was trained on 5 datasets  (`MIX 5` in the paper) can be found [here](https://github.com/intel-isl/MiDaS/releases/tag/v2).


### Changelog
* [Sep 2021] Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/DPT-Large).
* [Apr 2021] Released MiDaS v3.0:
    - New models based on [Dense Prediction Transformers](https://arxiv.org/abs/2103.13413) are on average [21% more accurate](#Accuracy) than MiDaS v2.1
    - Additional models can be found [here](https://github.com/intel-isl/DPT)
* [Nov 2020] Released MiDaS v2.1:
	- New model that was trained on 10 datasets and is on average about [10% more accurate](#Accuracy) than [MiDaS v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2)
	- New light-weight model that achieves [real-time performance](https://github.com/intel-isl/MiDaS/tree/master/mobile) on mobile platforms.
	- Sample applications for [iOS](https://github.com/intel-isl/MiDaS/tree/master/mobile/ios) and [Android](https://github.com/intel-isl/MiDaS/tree/master/mobile/android)
	- [ROS package](https://github.com/intel-isl/MiDaS/tree/master/ros) for easy deployment on robots
* [Jul 2020] Added TensorFlow and ONNX code. Added [online demo](http://35.202.76.57/).
* [Dec 2019] Released new version of MiDaS - the new model is significantly more accurate and robust
* [Jul 2019] Initial release of MiDaS ([Link](https://github.com/intel-isl/MiDaS/releases/tag/v1))

### Setup 

1) Pick one or more models and download corresponding weights to the `weights` folder:

- For highest quality: [dpt_large](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt)
- For moderately less quality, but better speed on CPU and slower GPUs: [dpt_hybrid](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt)
- For real-time applications on resource-constrained devices: [midas_v21_small](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt)
- Legacy convolutional model: [midas_v21](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt)

2) Set up dependencies: 

    ```shell
    conda install pytorch torchvision opencv
    pip install timm
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5.

    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the model:

    ```shell
    python run.py --model_type dpt_large
    python run.py --model_type dpt_hybrid 
    python run.py --model_type midas_v21_small
    python run.py --model_type midas_v21
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

Currently only supports MiDaS v2.1. DPT-based models to be added. 


#### via Mobile (iOS / Android)

See [README](https://github.com/intel-isl/MiDaS/tree/master/mobile) in the `mobile` subdirectory.

#### via ROS1 (Robot Operating System)

See [README](https://github.com/intel-isl/MiDaS/tree/master/ros) in the `ros` subdirectory.

Currently only supports MiDaS v2.1. DPT-based models to be added. 


### Accuracy

Zero-shot error (the lower - the better) and speed (FPS):

| Model |  DIW, WHDR | Eth3d, AbsRel | Sintel, AbsRel | Kitti, δ>1.25 | NyuDepthV2, δ>1.25 | TUM, δ>1.25 | Speed, FPS |
|---|---|---|---|---|---|---|---|
| **Small models:** | | | | | | | iPhone 11 |
| MiDaS v2 small | **0.1248** | 0.1550 | **0.3300** | **21.81** | 15.73 | 17.00 | 0.6 |
| MiDaS v2.1 small [URL]() | 0.1344 | **0.1344** | 0.3370 | 29.27 | **13.43** | **14.53** | 30 |
| | | | | | | |
| **Big models:** | | | | | | | GPU RTX 3090 |
| MiDaS v2 large [URL](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt) | 0.1246 | 0.1290 | 0.3270 | 23.90 | 9.55 | 14.29 | 51 |
| MiDaS v2.1 large [URL](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt) | 0.1295 | 0.1155 | 0.3285 | 16.08 | 8.71 | 12.51 | 51 |
| MiDaS v3.0 DPT-Hybrid [URL](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt) | 0.1106 | 0.0934 | 0.2741 | 11.56 | 8.69 | 10.89 | 46 |
| MiDaS v3.0 DPT-Large [URL](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) | **0.1082** | **0.0888** | **0.2697** | **8.46** | **8.32** | **9.97** | 47 |



### Citation

Please cite our paper if you use this code or any of the models:
```
@ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}
```

If you use a DPT-based model, please also cite:

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ICCV},
	year      = {2021},
}
```

### Acknowledgements

Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models). 
We'd like to thank the author for making these libraries available.

### License 

MIT License 
