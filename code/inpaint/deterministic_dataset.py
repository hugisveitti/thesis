# This is currently only used for evaluating the validation dataset.
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np 
import os

import config


# always use the same places
mask_size_ws = [53, 50, 63, 45, 62, 63, 33, 43]
mask_size_hs = [52, 44, 47, 56, 49, 60, 44, 58]

#r_ws = [72, 144, 139, 143, 148, 148, 76, 149]
#r_hs = [155, 21, 29, 84, 78, 160, 35, 48]

r_ws = [20, 30, 100, 100, 190, 148, 70, 190]
r_hs = [20, 190, 20, 100, 45, 50, 160, 130]


def normalize(img):
    return img 
toTensor = T.ToTensor()

class DeterministicSatelliteDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        super(DeterministicSatelliteDataset, self).__init__()
        self.root_dir = root_dir
        self.rgb_files = os.listdir(os.path.join(self.root_dir, "rgb"))
        self.len = num_samples if num_samples else len(self.rgb_files)

    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]
            img = normalize(img)
        return img

    def open_classes(self, idx):
        fn = self.rgb_files[idx].split(".")[0]
        with np.load(os.path.join(self.root_dir, "lc_classes", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = toTensor(classes)
            classes = classes.type(config.tensor_type)
        return classes

    def create_mask(self, lc_a, lc_b, lc_ab, binary_mask, rgb_b, rgb_ab, index):
        
        mask_size_w = mask_size_ws[index]
        mask_size_h = mask_size_hs[index]
        
        r_w = r_ws[index]
        r_h = r_hs[index]

        for i in range(r_w, r_w + mask_size_w):
            for j in range(r_h, r_h + mask_size_h):
                # wont create a rgb_ab anymore
                if not torch.equal(lc_a[:,i,j], lc_b[:,i,j]):
                    binary_mask[0, i, j] = 1
                    lc_ab[:,i,j] = lc_b[:,i,j]
                    rgb_ab[:, i, j] = rgb_b[:, i, j]
        


        return [r_w, r_h, mask_size_w, mask_size_h]
        # Just for illustration
        # put boxes around changed area.
        for i in range(r_w, r_w + mask_size_w):
            rgb_b[:,i,r_h] = torch.tensor([1,1,1])
            rgb_b[:,i,r_h + mask_size_h] = torch.tensor([1,1,1])
            rgb_ab[:, r_w, j] = torch.tensor([1,1,1])
            rgb_ab[:, r_w + mask_size_w, j] = torch.tensor([1,1,1])
            lc_ab[:,i,r_h] = torch.zeros(14)
            lc_ab[0,i,r_h] = 1
            lc_ab[:,i,r_h+mask_size_h] = torch.zeros(14)
            lc_ab[0,i,r_h + mask_size_h] = 1
        
        for j in range(r_h, r_h + mask_size_h):
            rgb_b[:, r_w, j] = torch.tensor([1,1,1])
            rgb_b[:, r_w + mask_size_w, j] = torch.tensor([1,1,1])
            rgb_ab[:, r_w, j] = torch.tensor([1,1,1])
            rgb_ab[:, r_w + mask_size_w, j] = torch.tensor([1,1,1])
            lc_ab[:,r_w,j] = torch.zeros(14)
            lc_ab[0,r_w,j] = 1
            lc_ab[:,r_w+mask_size_w, j] = torch.zeros(14)
            lc_ab[0,r_w + mask_size_w,j] = 1

        return [r_w, r_h, mask_size_w, mask_size_h]



    def __len__(self):
        return self.len 

    def __getitem__(self, idx_a):
        # select one image to create samples from
        # 100 is randomly chosen right now
        idx_b = 100 

        rgb_a = self.open_img(idx_a)
        lc_a = self.open_classes(idx_a)

        rgb_b = self.open_img(idx_b)
        lc_b = self.open_classes(idx_b)

        binary_mask = torch.zeros(1, lc_b.shape[1], lc_b.shape[2])
       
        lc_ab = lc_a.clone()
        rgb_ab = rgb_a.clone()
        num_inpaints = config.num_inpaints
        masked_areas = []
        for index in range(num_inpaints):
            masked_area = self.create_mask(lc_a, lc_b, lc_ab, binary_mask, rgb_b, rgb_ab, index)
            masked_areas.append(masked_area)

        return rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab, masked_areas


def test():
    from datautils import unprocess, create_img_from_classes
    import matplotlib.pyplot as plt

    ds = DeterministicSatelliteDataset("../../data/val")
    loader = DataLoader(ds, 1)
    i = 0
    num_examples = 5
    
    for rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab, masked_areas in loader:
        rgb_a = unprocess(rgb_a)
        rgb_ab = unprocess(rgb_ab)
        lc_a = unprocess(lc_a)
        lc_b = unprocess(lc_b)
        lc_ab = unprocess(lc_ab)
        binary_mask = np.array(binary_mask)[0][0]
        lc_a = create_img_from_classes(lc_a)
        lc_b = create_img_from_classes(lc_b)
        lc_ab = create_img_from_classes(lc_ab)

        fig, ax = plt.subplots(2,3, figsize=(12,8))
        fig.tight_layout()
        
        ax[0,0].imshow(rgb_a)
        ax[0,0].set_title("rgb_a (input)")
        
        ax[0,1].imshow(rgb_ab)
        ax[0,1].set_title("rgb_ab")
        
        ax[0,2].imshow(lc_ab)
        ax[0,2].set_title("lc_ab (input)")

        ax[1,0].imshow(lc_a)
        ax[1,0].set_title("lc_a")

        ax[1,1].imshow(lc_b)
        ax[1,1].set_title("lc_b")

        ax[1,2].imshow(binary_mask, cmap="gray")
        ax[1,2].set_title("binary mask (input)")

        folder = "testsetup"
        if not os.path.exists(folder):
            os.mkdir(folder)

        plt.savefig(folder + f"/det_input_example_{i}.png")

        i += 1
        if i == num_examples:
            break

def test_utils():
    from utils import save_example
    from generator import Generator
    from discriminator import Discriminator
    device = "cpu"

    d = DeterministicSatelliteDataset("../../data/train")
    l = DataLoader(d, 3)
    g = Generator().to(device)
    discriminator = Discriminator().to(device)

    save_example(g, discriminator, "testsetup", 0, l, device)

if __name__ == "__main__":
    test()
    #test_utils()