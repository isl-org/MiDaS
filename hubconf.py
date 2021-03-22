dependencies = ["torch"]

import torch

from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small


def MiDaS(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = MidasNet()

    if pretrained:
        checkpoint = (
            "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def MiDaS_small(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = MidasNet_small(None, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})

    if pretrained:
        checkpoint = (
            "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def transforms():
    import cv2
    from torchvision.transforms import Compose
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    from midas import transforms

    transforms.default_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.small_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    return transforms
