import matplotlib.pyplot as plt
from code.model.generator import Generator
from code.model.old_generator import Generator as OldGenerator
from code.inpaint.generator import Generator as InpaintGenerator
import os
import torch
import numpy as np
from draw_tool.draw_models_utils import num_classes, tensor_type, possible_lc_classes

device = "cpu"

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
    ma = ma.to(device)
    return ma

generator_file1 = "code/model/results/run35/models/generator.pt"
generator1 = Generator().to(device)
generator1.load_state_dict(torch.load(generator_file1))
generator1.eval()
for m in generator1.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()


generator_file2 = "code/model/results/run35/models/generator.pt"
generator2 = Generator().to(device)
# generator2.load_state_dict(torch.load(generator_file2))
generator2.eval()
for m in generator2.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()


# old_generator_file = "code/model/results/run1/models/generator.pt"
# old_generator = OldGenerator().to(device)
# old_generator.load_state_dict(torch.load(old_generator_file))
# generator2 = old_generator


inpaint_generator_file = "code/inpaint/results/inpaint_run6/models/generator.pt"
inpaint_generator = InpaintGenerator().to(device)
inpaint_generator.load_state_dict(torch.load(inpaint_generator_file))
inpaint_generator.eval()
for m in inpaint_generator.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()


c_dir = "draw_tool/testsetup"
if not os.path.exists(c_dir):
    os.mkdir(c_dir)

# model_name can be 'inpaint', mixlc1, mixlc2
def handle_images(d, model_name):
    use_inpaint = model_name == 'inpaint'
    rgb = d["rgb"]
    lc = d["lc"]
    binary_mask = d["binaryMask"]

    with torch.no_grad() and torch.cuda.amp.autocast():

        binary_mask = np.array(binary_mask)
        binary_mask = binary_mask.reshape((256, 256))


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

        rgb = rgb / 255
        lc_classes = convert_to_classes(lc)

        rgb = process(rgb)
        lc_classes = process(lc_classes)

        binary_mask = torch.tensor(binary_mask)
        binary_mask = binary_mask.reshape((1, 1, binary_mask.shape[0], binary_mask.shape[1]))
        binary_mask = binary_mask.type(tensor_type)
        binary_mask = binary_mask.to(device)

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