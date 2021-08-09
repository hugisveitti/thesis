import matplotlib.pyplot as plt
from code.model.generator import Generator
from code.model.old_generator import Generator as OldGenerator
from code.inpaint.generator import Generator as InpaintGenerator
from code.landcover_model.landcover_model import LandcoverModel
import os
import torch
import numpy as np
from draw_tool.draw_models_utils import num_classes, tensor_type, possible_lc_classes, unprocess, create_img_from_classes
import time

device = "cpu"

save_dir = "test_images/drawtool"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

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


generator_file1 = "code/model/models/generator1.pt"
generator1 = Generator().to(device)
generator1.load_state_dict(torch.load(generator_file1))
generator1.eval()
for m in generator1.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()


generator_file2 = "code/model/models/generator2.pt"
generator2 = Generator().to(device)
generator2.load_state_dict(torch.load(generator_file2))
generator2.eval()
for m in generator2.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()


inpaint_generator_file = "code/inpaint/models/generator.pt"
inpaint_generator = InpaintGenerator().to(device)
inpaint_generator.load_state_dict(torch.load(inpaint_generator_file))
inpaint_generator.eval()
for m in inpaint_generator.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()

landcover_model_file =  "code/landcover_model/models/lc_model.pt"
landcover_model = LandcoverModel().to(device)
landcover_model.load_state_dict(torch.load(landcover_model_file))
landcover_model.eval()


c_dir = "draw_tool/testsetup"
if not os.path.exists(c_dir):
    os.mkdir(c_dir)

# model_name can be 'inpaint', model1, model2
def handle_images(d, model_name, unchanged_lc):
    use_inpaint = model_name == 'inpaint'
    rgb = d["rgb"]
    lc = d["lc"]
    binary_mask = d["binaryMask"]

  
    save_all = d["saveImages"]
    save_name = d["saveName"]
    start_time = time.time()
    
    if save_name != "":
        curr_save_dir = os.path.join(save_dir, save_name)
        if not os.path.exists(curr_save_dir):
            os.mkdir(curr_save_dir)
    else:
        curr_save_dir = save_dir
        
    with torch.no_grad() and torch.cuda.amp.autocast():

        binary_mask = np.array(binary_mask)
        binary_mask = binary_mask.reshape((256, 256))

        rgb = np.array(rgb)
        rgb = rgb.reshape((256,256,3))

        rgb_masked = np.copy(rgb)

        if use_inpaint or save_all:
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    if binary_mask[i,j] == 1:
                        rgb_masked[i,j] = np.zeros(3)

        lc = np.array(lc)
        lc = lc.reshape((256,256,3))


        if save_all:
            plt.imsave(os.path.join(curr_save_dir, f"rgb.png"), np.array(rgb, dtype=np.uint8))
            plt.imsave(os.path.join(curr_save_dir, f"lc.png"), np.array(lc, dtype=np.uint8))
            plt.imsave(os.path.join(curr_save_dir, f"binary_mask.png"), np.array(binary_mask, dtype=np.uint8))
            plt.imsave(os.path.join(curr_save_dir, f"rgb_masked.png"), np.array(rgb_masked, dtype=np.uint8))
            plt.imsave(os.path.join(curr_save_dir, f"unchanged_lc.png"), np.array(unchanged_lc, dtype=np.uint8))
            
        rgb = rgb / 255
        rgb_masked = rgb_masked / 255
        lc_classes = convert_to_classes(lc)

        rgb = process(rgb)
        rgb_masked = process(rgb_masked)
        lc_classes = process(lc_classes)

        binary_mask = torch.tensor(binary_mask)
        binary_mask = binary_mask.reshape((1, 1, binary_mask.shape[0], binary_mask.shape[1]))
        binary_mask = binary_mask.type(tensor_type)
        binary_mask = binary_mask.to(device)

        if save_all:
            fake_img1 = generator1(rgb, lc_classes, binary_mask)
            lc_fake_img1 = landcover_model(fake_img1)
            fake_img1 = fake_img1.cpu().detach().numpy()[0]
            fake_img1 = np.moveaxis(fake_img1, 0, -1)
            fake_img1 = np.array(fake_img1*255, dtype=np.uint8)

            fake_img2 = generator2(rgb, lc_classes, binary_mask)
            lc_fake_img2 = landcover_model(fake_img2)
            fake_img2 = fake_img2.cpu().detach().numpy()[0]
            fake_img2 = np.moveaxis(fake_img2, 0, -1)
            fake_img2 = np.array(fake_img2*255, dtype=np.uint8)

            inpaint_img = inpaint_generator(rgb_masked, lc_classes)
            lc_inpaint_img1 = landcover_model(inpaint_img)
            inpaint_img = inpaint_img.cpu().detach().numpy()[0]
            inpaint_img = np.moveaxis(inpaint_img, 0, -1)
            inpaint_img = np.array(inpaint_img*255, dtype=np.uint8)

            plt.imsave(os.path.join(curr_save_dir, f"fake_img1.png"), fake_img1)
            plt.imsave(os.path.join(curr_save_dir, f"fake_img2.png"), fake_img2)
            plt.imsave(os.path.join(curr_save_dir, f"inpaint_img.png"), inpaint_img)

            lc_fake_img1 = create_img_from_classes(unprocess(lc_fake_img1.cpu().detach()))
            lc_fake_img2 = create_img_from_classes(unprocess(lc_fake_img2.cpu().detach()))
            lc_inpaint_img = create_img_from_classes(unprocess(lc_inpaint_img1.cpu().detach()))

            lc_gen_rgb = landcover_model(rgb)
            lc_gen_rgb = create_img_from_classes(unprocess(lc_gen_rgb.cpu().detach()))

            plt.imsave(os.path.join(curr_save_dir, f"lc_fake_img1.png"), lc_fake_img1)
            plt.imsave(os.path.join(curr_save_dir, f"lc_fake_img2.png"), lc_fake_img2)
            plt.imsave(os.path.join(curr_save_dir, f"lc_inpaint_img.png"), lc_inpaint_img)
            plt.imsave(os.path.join(curr_save_dir, f"lc_gen_rgb.png"), lc_gen_rgb)


        # could optimize by using already created picture.
        if model_name == 'model1':
            fake_img = generator1(rgb, lc_classes, binary_mask)
        elif model_name == 'model2':
            fake_img = generator2(rgb, lc_classes, binary_mask)
        elif model_name == 'inpaint':
            fake_img = inpaint_generator(rgb_masked, lc_classes)
        else:
            print("unsupported model name")
            exit()

    fake_img = fake_img.cpu().detach().numpy()[0]
    fake_img = np.moveaxis(fake_img, 0, -1)
    fake_img = np.array(fake_img*255, dtype=np.uint8)

    end_time = time.time()
    print("time ellapsed", end_time - start_time)

    return fake_img