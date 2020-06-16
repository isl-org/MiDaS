dependencies = ["torch"]

import torch

from models.midas_net import MidasNet


def MiDaS(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = MidasNet()

    if pretrained:
        checkpoint = (
            "https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model
