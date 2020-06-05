dependencies = ['torch']

from models.midas_est import MidasNet

def MiDaS(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = MidasNet()

    if pretrained:
        checkpoint = "https://drive.google.com/file/d/1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        model.load_state_dict(state_dict)

    return model
