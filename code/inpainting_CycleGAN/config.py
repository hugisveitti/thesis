import torch

torch_type = torch.FloatTensor

# size of the cutout
size = 96
start = (256 - size) // 2

# This is to elminate border between the input areas
buffer_zone = 12