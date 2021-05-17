import torch 
import torch.nn as nn
from torch.nn.modules.linear import Linear

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=5, stride=2,padding=2), # padding? I think 2 because if you think of a 5x5 kernel convoluting
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            Block(3, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            Block(512, 512), 
            Block(512, 512), # 512 x 4 x 4
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.Linear(1024, 1),
   # nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


def test_shape():

    imgs = torch.randn((1, 3, 256, 256))

    d = Discriminator()
    print(d(imgs))

    


if __name__ == "__main__":
    test_shape()