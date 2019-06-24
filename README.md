# Mixing Datasets For MonoDepth

This code allows to compute a depth map based on a single input image. It runs a neural network that was trained by mixing several datasets as described in

>Mixing Datasets for Single-Image Depth Estimation in Diverse Environments.  
Rene Ranftl, Katrin Lasinger, Vladlen Koltun

## Setup

1) Download the model weights [model.pt](https://drive.google.com/open?id=1Q9q7dVFhXiNOS1djOlaUUmnJlKMenEoU) and put the file in the same folder as this README.

2) Create and activate conda environment:

    ```shell
    conda env create -f environment.yml
    conda activate mixingDatasetsForMonoDepth
    ```

## Usage

1) Put one or more input images for monocular depth estimation in the folder `input`.

2) Produce depth maps for the images in the `input` folder as follows:

    ```shell
    python run.py
    ```

3) The resulting depth maps are written to the `output` folder.
