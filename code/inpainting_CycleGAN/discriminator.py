import torch 
import torch.nn as nn
from torch.nn.modules.linear import Linear
import numpy as np

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=5, stride=2,padding=2), # padding? I think 2 because if you think of a 5x5 kernel convoluting
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
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
            #Block(512, 512), 
            #Block(512, 256), # 512 x 4 x 4
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, 4, padding=1)
            # nn.Flatten(),
            # nn.Linear(512 * 4 * 4, 1024),
            # nn.Linear(1024, 1),
        )
    
    def forward(self, x):
        return self.network(x)


def test_shape():

    imgs = torch.randn((1, 3, 256, 256))

    d = Discriminator()
    
    adv_loss = nn.MSELoss()
    d_test = d(imgs)
    print("d test shape",d_test.shape)
    d_test_mse = torch.sigmoid(d_test)
    print("ones like",adv_loss(d_test_mse, torch.ones_like(d_test_mse)))
    print("ones like",nn.BCEWithLogitsLoss()(d_test, torch.ones_like(d_test)))
    print("zeros like",adv_loss(d_test_mse, torch.zeros_like(d_test_mse)))
    print("zeros like",nn.BCEWithLogitsLoss()(d_test, torch.zeros_like(d_test)))
    

    patch = (1,256 // 2 ** 4,256 // 2 ** 4)
    print("patch",patch)
    Tensor = torch.FloatTensor
    valid = torch.autograd.Variable(Tensor(np.ones((imgs.size(0), *patch))), requires_grad=False)
    print(valid.shape)
    print(adv_loss(d_test_mse, valid))


if __name__ == "__main__":
    test_shape()