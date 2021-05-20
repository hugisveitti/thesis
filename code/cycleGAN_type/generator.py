import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):

    def __init__(self, in_channels, out_channels, num_features=64, num_residuals=9):
        super(Generator, self).__init__()

        layers = []
        out_features = num_features
        channels = in_channels
        padding = 3

        layers += [
            nn.Conv2d(channels, out_features, kernel_size=7, padding=padding),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            ]
        

        in_features = out_features
        # Down sample
        for _ in range(2):
            out_features = out_features * 2
            layers += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                ]
            in_features = out_features


        for _ in range(num_residuals):
            layers += [
                ResidualBlock(out_features)
                ]
            

        # Up sample
        for _ in range(2):
            out_features = out_features // 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features,out_features, 3, 1, 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
                ]
            
            in_features = out_features
        # final layer
        padding = 3
        

        layers += [
            nn.Conv2d(out_features, out_channels, kernel_size=7, padding=padding, padding_mode="reflect"),
            nn.Tanh() if out_channels == 3 else nn.Softmax(dim=1)
            ]
        

        self.model = nn.Sequential(*layers)

  
    def forward(self, x):
        # todo: add skip layers?
        return self.model(x)

def test():
    img = torch.randn((5, 3, 256, 256))
    class_g = Generator(3, 14, num_features=64, num_residuals=9)
    print("gen class shape", class_g(img).shape)
    n_params = sum([np.prod(p.size()) for p in class_g.parameters()])
    print("number of parameters for classes generator", n_params)


    in_c = 14
    out_c = 3
    classes = torch.randn((5, in_c, 256, 256))
    img_g = Generator(in_c, out_c, 64, 9)
    print("gen img shape", img_g(classes).shape)
    n_params = sum([np.prod(p.size()) for p in img_g.parameters()])
    print("number of parameters for image generator", n_params)


if __name__ == "__main__":
    test()