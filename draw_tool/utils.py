import matplotlib.pyplot as plt
from code.model.old_generator import Generator
from code.inpaint.generator import Generator as InpaintGenerator
import os
import torch
import numpy as np

possible_lc_classes = {
    str([255, 255, 255]): 0,
    str([210, 0, 0]): 1,
    str([253, 211, 39]): 2,
    str([176, 91, 16]): 3,
    str([35, 152, 0]): 4,
    str([8, 98, 0]): 5,
    str([249, 150, 39]): 6,
    str([141, 139, 0]): 7,
    str([95, 53, 6]): 8,
    str([149, 107, 196,]): 9,
    str([77, 37, 106,]): 10,
    str([154, 154, 154,]): 11,
    str([106, 255, 255,]): 12,
    str([20, 69, 249,]): 13,
}

tensor_type = torch.FloatTensor

def get_onehot(num):
    arr = np.zeros(14)
    arr[num] = 1
    return arr

def convert_to_classes(lc):
    lc_classes = np.zeros((lc.shape[0],lc.shape[1],14))
    for i in range(lc.shape[0]):
        for j in range(lc.shape[1]):
            lc_classes[i,j, possible_lc_classes[str(lc[i,j].tolist())]] = 1
    return lc_classes

def process(ma):
    ma = torch.tensor(ma)
    ma = torch.movedim(ma, -1, 0)
    ma = ma.reshape((1,ma.shape[0],ma.shape[1],ma.shape[2]))
    ma = ma.type(tensor_type)
    return ma

generator_file = "code/model/results/run1/models/generator.pt"
generator = Generator()
generator.load_state_dict(torch.load(generator_file))

inpaint_generator_file = "code/inpaint/results/inpaint_run1/models/generator.pt"
inp_generator = InpaintGenerator()
inp_generator.load_state_dict(torch.load(inpaint_generator_file))

c_dir = "draw_tool/testsetup"
if not os.path.exists(c_dir):
    os.mkdir(c_dir)

def handle_images(d, use_inpaint=True):
    rgb = d["rgb"]
    lc = d["lc"]
    binary_mask = d["binaryMask"]

    binary_mask = np.array(binary_mask)
    binary_mask = binary_mask.reshape(( 256, 256))


    rgb = np.array(rgb)
    rgb = rgb.reshape((256,256,3))

    if use_inpaint:
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if binary_mask[i,j] == 1:
                    rgb[i,j] = np.zeros(3)

    lc = np.array(lc)
    lc = lc.reshape((256,256,3))

    plt.imshow(rgb)
    plt.savefig(os.path.join(c_dir,"modrgb.png"))
    plt.close()

    plt.imshow(lc)
    plt.savefig(os.path.join(c_dir,"modlc.png"))
    plt.close()


    plt.imshow(binary_mask)
    plt.savefig(os.path.join(c_dir,"binmask.png"))
    plt.close()

    rgb = torch.tensor(rgb / 255)
    lc_classes = convert_to_classes(lc)

    rgb = process(rgb)
    lc_classes = process(lc_classes)

    binary_mask = torch.tensor(binary_mask)
    binary_mask = binary_mask.reshape((1, 1, binary_mask.shape[0], binary_mask.shape[1]))
    binary_mask = binary_mask.type(tensor_type)


    if not use_inpaint:
        fake_img = generator(rgb, lc_classes, binary_mask)
    else:
        fake_img = inp_generator(rgb, lc_classes)

    fake_img = fake_img.cpu().detach().numpy()[0]
    fake_img = np.moveaxis(fake_img, 0, -1)
    fake_img = np.array(fake_img*255, dtype=np.uint8)
    plt.imshow(fake_img)
    plt.savefig(os.path.join(c_dir,"fake_img.png"))
    plt.close()

    return fake_img