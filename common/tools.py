import torch.nn as nn


def init_ortho(module, gain=1):
    if isinstance(gain, str):
        gain = nn.init.calculate_gain(gain)
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module
