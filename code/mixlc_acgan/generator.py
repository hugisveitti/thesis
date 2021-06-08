import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, down=True):
        super(Block, self).__init__()
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
        activation = nn.ReLU() # if down else nn.LeakyReLU(0.2)

        self.block = nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5), # Batch norm acts as noise
            activation,
        )

    def forward(self, x):
        return self.block(x)


class ChannelNetwork(nn.Module):

    def __init__(self, in_channels):
        super(ChannelNetwork, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.down1 = Block(64, 128)
        self.down2 = Block(128, 256)
        self.down3 = Block(256, 512)
        self.down4 = Block(512, 1024)
        
        bottleneck_features = 1024

        self.bottleneck1 = Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)
        self.bottleneck2 = Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)

        self.up4 = Block(1024 * 2, 512, down=False)
        self.up3 = Block(512 * 2, 256, down=False)
        self.up2 = Block(256 * 2, 128, down=False)
        self.up1 = Block(128 * 2, 64, down=False)
        self.final = Block(64 * 2, 64, stride=1)

    def forward(self, x):
        d1 = self.first_layer(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        x = self.bottleneck2(d5)
        x = self.bottleneck2(x)
        
        x = self.up4(torch.cat([d5, x], dim=1))
        x = self.up3(torch.cat([d4, x], dim=1))
        x = self.up2(torch.cat([d3, x], dim=1))
        x = self.up1(torch.cat([d2, x], dim=1))
        return self.final(torch.cat([d1, x], dim=1))

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.rgb_network = ChannelNetwork(in_channels = 3)
        self.lc_network = ChannelNetwork(in_channels = 14)

        self.output_network = nn.Sequential(
            Block(64 * 2, 64, down=False),
            nn.Conv2d(64, 3, kernel_size = 3, stride=1, padding=2),
            nn.Tanh()
        )
    
    def forward(self, rgb, lc):
        rgb = self.rgb_network(rgb)
        lc = self.lc_network(lc)
        return self.output_network(torch.cat([rgb, lc], dim=1))


def test():
    g = Generator()
    rgb = torch.randn((1, 3, 256, 256))
    lc = torch.randn((1, 14, 256, 256))
    print(g(rgb, lc).shape)



if __name__ == "__main__":
    test()