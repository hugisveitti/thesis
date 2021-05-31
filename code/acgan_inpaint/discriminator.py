import torch 
import torch.nn as nn
import numpy as np

if __name__ == "discriminator" or __name__ == "__main__":
    from config import torch_type
else:
    from .config import torch_type


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
            #nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ACDiscriminator(nn.Module):

    def __init__(self, in_channels=3, features = 64):
        super(ACDiscriminator, self).__init__()
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

        num_classes = 14
        self.class_generator = nn.Sequential(
            nn.ConvTranspose2d(features*2, num_classes, 4,2,1),
           # nn.Tanh()
            nn.Softmax(dim=1)
        )

        # last layer be very large kernel?
        self.final_layer = nn.Sequential(
            Block(num_classes, 1, 17, 16),
          #  Block()
            nn.Softmax(dim=1)
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

        gen_lc = self.class_generator(torch.cat([up7, d1], 1))
        outp = self.final_layer(gen_lc)
        
        return gen_lc, outp
        

def test_shape():

    imgs = torch.randn((1, 3, 256, 256))

    d = ACDiscriminator(3)
    
    adv_loss = nn.MSELoss()
    gen_lc, outp = d(imgs)
    print("gen lc shape", gen_lc.shape)
    print("outp shape", outp.shape)

    target_class = torch.tensor(np.random.random_integers(0, 13, (1, 256, 256))).to(torch.long)

    print(outp)
    print("ones like",adv_loss(outp, torch.ones_like(outp)))

    print("ones like",nn.CrossEntropyLoss()(gen_lc, target_class))
    



if __name__ == "__main__":
    test_shape()