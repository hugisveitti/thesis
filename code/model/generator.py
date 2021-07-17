import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, down=True):
        super(Block, self).__init__()
        if down:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
        activation = nn.LeakyReLU(0.2) if down else nn.ReLU()

        self.block = nn.Sequential(
            self.conv_layer,
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.2) if not down else nn.Identity(), # Dropout acts as noise, see pix2pix
            activation,
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    def __init__(self, in_channels = 3 + 14 + 1):
        super(Generator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1,),
            nn.ReLU()
        )

        self.down1 = Block(64, 128)
        self.down2 = Block(128, 256)
        self.down3 = Block(256, 512)
        self.down4 = Block(512, 1024)
        
        bottleneck_features = 1024

        self.bottleneck = Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)
        self.bottleneck2 = Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)

        self.up4 = Block(bottleneck_features * 2, 512, down=False)
        self.up3 = Block(512 * 2, 256, down=False)
        self.up2 = Block(256 * 2, 128, down=False)
        self.up1 = Block(128 * 2, 64, down=False)
        
        self.up0 = Block(64 * 2, 64,  down=False)
        self.final = nn.Sequential(
            Block(64, 64, kernel_size=3, stride=1),
            nn.Conv2d(64, 3, kernel_size = 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, lc, lc_mask, rgb):
        x = torch.cat([lc, lc_mask, rgb], dim=1)
        d1 = self.first_layer(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        x = self.bottleneck(d5)
        
        x = self.bottleneck2(x)
        
        x = self.up4(torch.cat([d5, x], dim=1))
        x = self.up3(torch.cat([d4, x], dim=1))
        x = self.up2(torch.cat([d3, x], dim=1))
        x = self.up1(torch.cat([d2, x], dim=1))
        x = self.up0(torch.cat([d1, x], dim=1))
        return self.final(x)


def test():
    batch_size = 8
    g = Generator()
    rgb = torch.randn((batch_size, 3, 256, 256))
    lc = torch.randn((batch_size, 14, 256, 256))
    lc_mask = torch.randn((batch_size, 1, 256, 256))
    print(g(rgb, lc, lc_mask).shape)

    import numpy as np
    n_params = sum([np.prod(p.size()) for p in g.parameters()])
    print("number of parameters in generator", n_params)




if __name__ == "__main__":
    test()