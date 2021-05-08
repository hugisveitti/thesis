import torch 
import torch.nn as nn
import config

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=5, stride=2,padding=2), # padding? I think 2 because if you think of a 5x5 kernel convoluting
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.network = nn.Sequential(
            Block(in_channels, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            Block(512, 512), 
            nn.Identity() if config.IMAGE_HEIGHT == 128 else Block(512, 512), # 512 x 4 x 4
            nn.Flatten(),
            nn.Linear(512 * 4*4, 1024),
            nn.Linear(1024, 1),
            # BCEWITHLOGITSLOSS performs a sigmoid and BCELoss
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
        


def test():
    img = torch.randn((2,3,config.IMAGE_WIDTH,config.IMAGE_HEIGHT))
    d = Discriminator()
    print(d(img).shape)


if __name__ == "__main__":
    test()