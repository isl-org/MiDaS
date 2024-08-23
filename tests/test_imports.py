import pytest


def test_imports():
    """"Test that all imports work."""
    from midas.model_loader import default_models, load_model
    from midas.dpt_depth import DPTDepthModel
    from midas.midas_net import MidasNet
    from midas.midas_net_custom import MidasNet_small
    from midas.transforms import NormalizeImage, PrepareForNet, Resize
    from midas.blocks import ResidualConvUnit
    from midas.backbones.beit import beit_forward_features
    from midas.base_model import BaseModel


if __name__ == "__main__":
    pytest.main([__file__])
