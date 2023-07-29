import torch

midasl = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midasS = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transforml = midas_transforms.default_transform
transformS = midas_transforms.small_transform
