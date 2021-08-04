import torch
import torch.nn as nn
# import config
class Config:
    def __init__(self):
        self.num_classes = 9
config = Config()


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, down=True, dilation=1):
        super(Block, self).__init__()
        if down:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation)
        else:
            self.conv_layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
            ) 
        
        activation = nn.ELU() 

        self.block = nn.Sequential(
            self.conv_layer,
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.25) if not down else nn.Identity(), # Dropout acts as noise, see pix2pix
            activation,
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    def __init__(self, in_channels = 3 + config.num_classes):
        super(Generator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=in_channels+1, stride=1, padding=in_channels//2),
            nn.ELU()
        )
        bottleneck_features = 512

        self.down1 = Block(64, 128)
        self.down2 = Block(128, 256)
        self.down3 = Block(256, 512)
        self.down4 = Block(512, bottleneck_features)
    
        self.down5 = Block(bottleneck_features, bottleneck_features, stride=1, padding=2, kernel_size=3, dilation=2)
        self.bottleneck = Block(bottleneck_features, bottleneck_features, stride=1, padding=4, kernel_size=3, dilation=4)

        self.up5 = Block(bottleneck_features * 2, 512, down=False, stride=1, padding=1)
        self.up4 = Block(512 * 2, 256, down=False)
        self.up3 = Block(256 * 2, 128, down=False)
        self.up2 = Block(128 * 2, 64, down=False)
        
        self.up1 = Block(64 * 2, 64, stride=1)
        self.final = nn.Sequential(
            Block(64, 64, stride=1),
            nn.Conv2d(64, 3, kernel_size = 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_masked, lc):
        x = torch.cat([rgb_masked, lc], dim=1)
        
        d1 = self.first_layer(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        b = self.down5(d5)
        u5 = self.bottleneck(b)
        
        u4 = self.up5(torch.cat([d5, u5], dim=1))
        u3 = self.up4(torch.cat([d4, u4], dim=1))
        u2 = self.up3(torch.cat([d3, u3], dim=1))
        u1 = self.up2(torch.cat([d2, u2], dim=1))
        x = self.up1(torch.cat([d1, u1], dim=1))
        return self.final(x)

    


def test():
    batch_size = 8
    g = Generator()
    rgb_masked = torch.randn((batch_size, 3, 256, 256))
    lc = torch.randn((batch_size, config.num_classes, 256, 256))
    print(g(rgb_masked, lc).shape)

    import numpy as np
    n_params = sum([np.prod(p.size()) for p in g.parameters()])
    print("number of parameters in generator", n_params)




if __name__ == "__main__":
    test()