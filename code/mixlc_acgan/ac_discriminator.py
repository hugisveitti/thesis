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


class DownNetwork(nn.Module):

    def __init__(self, in_channels):
        super(DownNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            Block(512, 1024)
        )
        


    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.rgb_network = DownNetwork(in_channels = 3)
        self.lc_network = DownNetwork(in_channels = 14)

        bottleneck_features = 1024

        self.join_net = nn.Sequential(
            Block(bottleneck_features * 2, bottleneck_features, stride=1, padding=1, kernel_size=3),
            Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)
        )

        self.patch_classifier = nn.Sequential(
            nn.Conv2d(bottleneck_features, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.lc_gen_net = nn.Sequential(
            Block(1024, 1024, down=False),
            Block(1024, 512, down=False),
            Block(512, 256, down=False),
            Block(256, 128, down=False),
            Block(128, 64, down=False),
            Block(64, 14, stride=1, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )



    
    def forward(self, rgb, lc):
        rgb = self.rgb_network(rgb)
        lc = self.lc_network(lc)
        x = self.join_net(torch.cat([rgb, lc], dim=1))
        patch_class = self.patch_classifier(x)
        lc_gen = self.lc_gen_net(x)
        return lc_gen, patch_class


def test():
    g = Discriminator()
    rgb = torch.randn((1, 3, 256, 256))
    lc = torch.randn((1, 14, 256, 256))
    lc_gen, patch_class = g(rgb, lc)
    print("lc_gen", lc_gen.shape)
    print("patch_class", patch_class.shape)



if __name__ == "__main__":
    test()