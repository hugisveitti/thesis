import torch

tensor_type = torch.FloatTensor
local_area_margin = 8
num_inpaints = 4
device = "cuda"
num_classes = 9

cross_entropy_weights = torch.tensor([1.75827705e-05, 2.63985891e-02, 4.93802954e-01, 0.00000000e+00,
       9.56997597e-02, 6.52813402e-02, 2.43301976e-01, 2.19168076e-02,
       0.00000000e+00, 2.51651604e-02, 2.09771106e-02, 3.17152767e-03,
       0.00000000e+00, 4.26719143e-03]).to(device)
