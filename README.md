## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer  
Katrin Lasinger, Rene Ranftl,  Konrad Schindler, Vladlen Koltun

The pre-trained model corresponds to `RW+MD+MV` with `MGDA` enabled and movies sampled at 4 frames per second.

### Setup 

1) Download the model weights [model.pt](https://drive.google.com/open?id=1Q9q7dVFhXiNOS1djOlaUUmnJlKMenEoU) and place the
file in the root folder.

2) Setup dependencies: 

    ```shell
    conda install pytorch torchvision opencv
    ```

   The code was tested with Python 3.7, PyTorch 1.0.1, and OpenCV 3.4.2.

    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the model:

    ```shell
    python run.py
    ```

3) The resulting depth maps are written to the `output` folder.


### Citation

Please cite our paper if you use this code in your research:
```
@article{Lasinger2019,
	author    = {Katrin Lasinger and Ren\'{e} Ranftl and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for 
        Zero-Shot Cross-Dataset Transfer},
	journal   = {arXiv:1907.01341},
	year      = {2019},
}
```


### License 

MIT License 
