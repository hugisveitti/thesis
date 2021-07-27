import matplotlib.pyplot as plt
from code.model.generator import Generator
from code.model.old_generator import Generator as OldGenerator
from code.inpaint.generator import Generator as InpaintGenerator
import os
import torch
import numpy as np
from draw_tool.draw_models_utils import  num_classes, tensor_type, possible_lc_classes


def get_onehot(num):
    arr = np.zeros(num_classes)
    arr[num] = 1
    return arr

def convert_to_classes(lc):
    lc_classes = np.zeros((lc.shape[0],lc.shape[1], num_classes))
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

generator_file1 = "code/model/results/run12/models/generator.pt"
generator1 = Generator()
generator1.load_state_dict(torch.load(generator_file1))

generator_file2 = "code/model/results/run13/models/generator.pt"
generator2 = Generator()
generator2.load_state_dict(torch.load(generator_file2))


# old_generator_file = "code/model/results/run1/models/generator.pt"
# old_generator = OldGenerator()
# old_generator.load_state_dict(torch.load(old_generator_file))
# generator = old_generator


inpaint_generator_file = "code/inpaint/results/inpaint_run3/models/generator.pt"
inpaint_generator = InpaintGenerator()
inpaint_generator.load_state_dict(torch.load(inpaint_generator_file))

c_dir = "draw_tool/testsetup"
if not os.path.exists(c_dir):
    os.mkdir(c_dir)

# model_name can be 'inpaint', mixlc1, mixlc2
def handle_images(d, model_name):
    use_inpaint = model_name == 'inpaint'
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


    if model_name == 'mixlc1':
        fake_img = generator1(rgb, lc_classes, binary_mask)
    elif model_name == 'mixlc2':
        fake_img = generator2(rgb, lc_classes, binary_mask)
    elif model_name == 'inpaint':
        fake_img = inpaint_generator(rgb, lc_classes)
    else:
        print("unsupported model name")
        exit()

    fake_img = fake_img.cpu().detach().numpy()[0]
    fake_img = np.moveaxis(fake_img, 0, -1)
    fake_img = np.array(fake_img*255, dtype=np.uint8)
    plt.imshow(fake_img)
    plt.savefig(os.path.join(c_dir,"fake_img.png"))
    plt.close()

    return fake_img