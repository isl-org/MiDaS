## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

### TensorFlow inference using `.pb` and `.onnx` models

1. [Run inference on TensorFlow-model by using TensorFlow](#run-inference-on-tensorflow-model-by-using-tensorFlow)

2. [Run inference on ONNX-model by using TensorFlow](#run-inference-on-onnx-model-by-using-tensorflow)

3. [Make ONNX model from downloaded Pytorch model file](#make-onnx-model-from-downloaded-pytorch-model-file)


### Run inference on TensorFlow-model by using TensorFlow

1) Download the model weights [model-f45da743.pb](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pb) and place the
file in the `/tf/` folder.

2) Set up dependencies: 

```shell
# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install TensorFlow
pip install grpcio tensorflow tensorflow-addons
```

#### Usage

1) Place one or more input images in the folder `tf/input`.

2) Run the model:

    ```shell
    python tf/run_pb.py
    ```

3) The resulting inverse depth maps are written to the `tf/output` folder.


### Run inference on ONNX-model by using TensorFlow

1) Download the model weights [model-f45da743.onnx](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.onnx) and place the
file in the `/tf/` folder.

2) Set up dependencies: 

```shell
# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install TensorFlow
pip install grpcio tensorflow tensorflow-addons

# install ONNX and ONNX_TensorFlow
pip install onnx

git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e . && cd ..
dir
```

#### Usage

1) Place one or more input images in the folder `tf/input`.

2) Run the model:

    ```shell
    python tf/run_onnx.py
    ```

3) The resulting inverse depth maps are written to the `tf/output` folder.



### Make ONNX model from downloaded Pytorch model file

1) Download the model weights [model-f45da743.pt](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt) and place the
file in the root folder.

2) Set up dependencies: 

```shell
# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install ONNX
pip install onnx

# install PyTorch TorchVision
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Usage

1) Run the converter:

    ```shell
    python tf/make_onnx_model.py
    ```

2) The resulting `model-f46da743.onnx` file is written to the `/tf/` folder.


### Requirements

   The code was tested with Python 3.6.9, PyTorch 1.5.1, TensorFlow 2.2.0, TensorFlow-addons 0.8.3, ONNX 1.7.0, ONNX-TensorFlow (GitHub-master-17.07.2020) and OpenCV 4.3.0.
 
### Citation

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2019,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

### License 

MIT License 

   
