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
        with np.load(os.path.join(self.root_dir, "lc_sieve", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = toTensor(classes)
            classes = classes.type(config.tensor_type)
        return classes

    def create_mask(self, rgb_masked):
        
        mask_size_w = np.random.randint(32, 64)
        mask_size_h = np.random.randint(32, 64)
        
        r_w = np.random.randint(config.local_area_margin, rgb_masked.shape[1] - mask_size_w - config.local_area_margin)
        r_h = np.random.randint(config.local_area_margin, rgb_masked.shape[2] - mask_size_h - config.local_area_margin)

        for i in range(r_w, r_w + mask_size_w):
            for j in range(r_h, r_h + mask_size_h):
                # wont create a rgb_ab anymore
                rgb_masked[:, i, j] = torch.tensor([0,0,0])

        return [r_w, r_h, mask_size_w, mask_size_h]



    def __len__(self):
        return self.len 

    def __getitem__(self, idx):

        rgb = self.open_img(idx)
        lc = self.open_classes(idx)

        rgb_masked = rgb.clone()
        num_inpaints = config.num_inpaints
        masked_areas = []
        for _ in range(num_inpaints):
            masked_area = self.create_mask(rgb_masked)
            masked_areas.append(masked_area)

        return rgb, lc, rgb_masked, masked_areas


def test():
    from datautils import unprocess, create_img_from_classes
    import matplotlib.pyplot as plt

    ds = DeterministicSatelliteDataset("../../data/grid_dir/val")
    loader = DataLoader(ds, 1)
    i = 0
    num_examples = 5
    
    for rgb_a, lc_a, rgb_a_masked, masked_areas in loader:
        rgb_a = unprocess(rgb_a)
        rgb_a_masked = unprocess(rgb_a_masked)
        lc_a = unprocess(lc_a)
        lc_a = create_img_from_classes(lc_a)

        fig, ax = plt.subplots(1,3, figsize=(12,4))
        fig.tight_layout()
        
        ax[0].imshow(rgb_a_masked)
        ax[0].set_title("rgb_a_masked (input)")
        
        ax[1].imshow(lc_a)
        ax[1].set_title("lc_a (input)")

        ax[2].imshow(rgb_a)
        ax[2].set_title("rgb_a (target)")

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

    d = DeterministicSatelliteDataset("../../data/grid_dir/train")
    l = DataLoader(d, 3)
    g = Generator().to(device)
    discriminator = Discriminator().to(device)

    save_example(g, discriminator, "testsetup", 0, l, device)

if __name__ == "__main__":
    test()
    #test_utils()