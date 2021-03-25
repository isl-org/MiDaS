import cv2
import torch
import gradio as gr
import numpy as np
from PIL import Image

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

use_large_model = True

if use_large_model:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
else:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

device = "cpu"
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
    transform = midas_transforms.default_transform
else:
    transform = midas_transforms.small_transform


def depth(img):
  cv_image = np.array(img) 
  img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

  input_batch = transform(img).to(device)
  with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
  output = prediction.cpu().numpy()
  formatted = (output * 255 / np.max(output)).astype('uint8')
  img = Image.fromarray(formatted)
  return img
    

inputs =  gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil",label="Output Image")

title = "MiDaS"
description = "demo for MiDaS v2.1 which takes in a single image for computing relative depth. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1907.01341v3'>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer</a> | <a href='https://github.com/intel-isl/MiDaS'>Github Repo</a></p>"

examples = [
    ["turtle.jpg"],
    ["lions.jpg"]
]

gr.Interface(depth, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True)