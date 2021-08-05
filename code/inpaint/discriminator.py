import torch
import torch.nn as nn
import config


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, down=True):
        super(Block, self).__init__()
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            conv_layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
            ) 
            
        activation = nn.ELU() 

        self.block = nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.ELU()
        )
        bottleneck_features = 512

        self.down1 = Block(64, 128)
        self.down2 = Block(128, 256)
        self.down3 = Block(256, 512)
        self.down4 = Block(512, bottleneck_features)
        

        self.down5 = Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)
        self.bottleneck = Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)

        self.up5 = Block(bottleneck_features * 2, 512, down=False)
        self.up4 = Block(512 * 2, 256, down=False)
        self.up3 = Block(256 * 2, 128, down=False)
        self.up2 = Block(128 * 2, 64, down=False)
        self.up1 = Block(64 * 2, 64, stride=1)

        self.final = nn.Sequential(
            Block(64, 64, kernel_size = 3, stride=1),
            nn.Conv2d(64, config.num_classes, kernel_size=3, stride=1, padding=1),
        )
        
        self.patch_gan_net = nn.Sequential(
            Block(bottleneck_features, bottleneck_features, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(bottleneck_features, 1, kernel_size=1, stride=1),
        )

    def forward(self, x):
        d1 = self.first_layer(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        patch_gan = self.patch_gan_net(d5)

        b = self.down5(d5)
        u5 = self.bottleneck(b)
        
        u4 = self.up5(torch.cat([d5, u5], dim=1))
        u3 = self.up4(torch.cat([d4, u4], dim=1))
        u2 = self.up3(torch.cat([d3, u3], dim=1))
        u1 = self.up2(torch.cat([d2, u2], dim=1))
        x = self.up1(torch.cat([d1, u1], dim=1))
        gen_lc = self.final(x)
        return gen_lc, patch_gan


def test():
    d = Discriminator()
    rgb = torch.randn((1, 3, 256, 256))
    gen_lc, patch_gan = d(rgb)

    print("gen lc shape", gen_lc.shape)
    print("patch_GAN shape", patch_gan.shape)
    import numpy as np
    n_params = sum([np.prod(p.size()) for p in d.parameters()])
    print("number of parameters in discriminator", n_params)


if __name__ == "__main__":
    test()