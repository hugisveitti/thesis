# The discriminator networks use 70 x 70 PatchGANs
import torch
import torch.nn as nn
import numpy as np


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            Block(in_channels, 64, normalize=False),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


def test():

    x_img = torch.randn((1,3, 256, 256))
    d_img = Discriminator(3)
    print(d_img(x_img).shape)
    n_params = sum([np.prod(p.size()) for p in d_img.parameters()])
    print("number of parameters for image discriminator", n_params)

    x_classes = torch.randn((1, 14, 256, 256))
    d_classes = Discriminator(14)
    print(d_classes(x_classes).shape)
    n_params = sum([np.prod(p.size()) for p in d_classes.parameters()])
    print("number of parameters for classes discriminator", n_params)


    

if __name__ == "__main__":
    test()