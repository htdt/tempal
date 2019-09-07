import torch
from torch import nn
import torch.nn.functional as F


class STDIM:
    def __init__(self, feature_size, device='cpu'):
        super().__init__()
        self.encoder = Conv(feature_size=feature_size).to(device)
        self.classifier1 = nn.Linear(feature_size, 128).to(device)
        self.classifier2 = nn.Linear(128, 128).to(device)
        self.encoder.train()
        self.classifier1.train()
        self.classifier2.train()
        self.device = device

    def get_loss(self, x1, x2):
        x1_loc, x1_glob = self.encoder(x1)
        x2_loc = self.encoder.block1(x2)

        batch_size = x1.shape[0]
        sy, sx = x1_loc.shape[2:]
        target = torch.arange(batch_size).to(self.device)

        loss = 0
        for y in range(sy):
            for x in range(sx):
                positive = x2_loc[:, :, y, x]

                predictions = self.classifier1(x1_glob)
                logits = torch.matmul(predictions, positive.t())
                loss += F.cross_entropy(logits, target)

                predictions = self.classifier2(x1_loc[:, :, y, x])
                logits = torch.matmul(predictions, positive.t())
                loss += F.cross_entropy(logits, target)

        return loss / (sx * sy)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def OrthoConv(in_channels, out_channels, kernel_size, stride):
    module = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

def OrthoLinear(in_channels, out_channels):
    module = nn.Linear(in_channels, out_channels)
    nn.init.orthogonal_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module

class Conv(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.block1 = nn.Sequential(
            OrthoConv(1, 32, 8, stride=4),
            nn.ReLU(),
            OrthoConv(32, 64, 4, stride=2),
            nn.ReLU(),
            OrthoConv(64, 128, 4, stride=2))
        
        self.block2 = nn.Sequential(
            nn.ReLU(),
            OrthoConv(128, 64, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            OrthoLinear(64 * 9 * 6, feature_size),
        )

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        return b1, b2
