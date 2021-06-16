import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, down=True, act="relu"):
        super(Block, self).__init__()
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
        activation = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)



        self.block = nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(out_channels),
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
        self.down_network = DownNetwork(in_channels = 3)

        bottleneck_features = 1024

        self.bottleneck = nn.Sequential(
            Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3),
            Block(bottleneck_features, bottleneck_features, stride=1, padding=1, kernel_size=3)
        )

        self.patch_classifier = nn.Sequential(
            Block(1024, 512, down=False, act="leaky"),
            Block(512, 256, down=False, act="leaky"),
            nn.Conv2d(256, 1, 3, padding=1),
            
           # nn.BatchNorm2d(512)
           # lsgan needs no sigmoid
            # nn.Sigmoid()
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



    
    def forward(self, rgb):
        x = self.down_network(rgb)
        x = self.bottleneck(x)
        patch_class = self.patch_classifier(x)
        lc_gen = self.lc_gen_net(x)
        return lc_gen, patch_class


def test():
    device = "cuda"
    adv_loss_fn = nn.MSELoss()
    g = Discriminator().to(device)
    rgb = torch.randn((1, 3, 256, 256)).to(device)
    lc_gen, patch_class = g(rgb)
    with torch.cuda.amp.autocast():
        print("lc_gen", lc_gen.shape)
        print("patch_class", patch_class.shape)
        print(patch_class)
        print(adv_loss_fn(patch_class, torch.ones_like(patch_class)))
        


if __name__ == "__main__":
    test()