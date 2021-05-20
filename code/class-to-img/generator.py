import torch
import torch.nn as nn
import config

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, stride=2, down=True):
        super(Block, self).__init__()
        padding = 1
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels,kernel_size=kernel, stride=stride, padding=padding,)
        else:
            conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding=padding,)
        
        self.block = nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    

    def __init__(self, in_channels, out_channels, features = 64):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.down1 = Block(features, features*2)
        self.down2 = Block(features*2, features*4)
        self.down3 = Block(features*4, features*8)
        self.down4 = Block(features*8, features*8)
        self.down5 = Block(features*8, features*8)
        self.down6 = Block(features*8, features*8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8, kernel_size=4,stride=2, padding=1),
            nn.ReLU()
        )

        self.up1 = Block(features*8, features*8, down=False, )
        self.up2 = Block(features*8*2, features*8, down=False,)
        self.up3 = Block(features*8*2, features*8, down=False,)
        self.up4 = Block(features*8*2, features*8, down=False,)
        self.up5 = Block(features*8*2, features*4, down=False,)
        self.up6 = Block(features*4*2, features*2, down=False,)
        self.up7 = Block(features*2*2, features, down=False, )

        self.output_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4,2,1),
            nn.Tanh()
         #   nn.Softmax(dim=1)
        )



    def forward(self, x):
        d1 = self.first_layer(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1)) # Skip layers in u-net, #3.2.1 in pix2pix paper
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.output_up(torch.cat([up7, d1], 1))
        


def test():
    device = "cuda"
    g = Generator(14, 3).to(device)
    img = torch.randn((4, 14, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)).to(device)
    with torch.cuda.amp.autocast():
        fake = g(img)
        print(img.shape)
        print(fake.shape)
        print(fake)


if __name__ == "__main__":
    test()