import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np 
import os

import config

flip_horizontal = T.RandomHorizontalFlip(p=1)
flip_vertical = T.RandomVerticalFlip(p=1)
rand_rotation_90 = T.RandomRotation((90,90)) # -90 or 90
rand_rotation_180 = T.RandomRotation((180,180)) # -180 or 180

transf_types = [flip_horizontal, flip_vertical, rand_rotation_90, rand_rotation_180]

toTensor = T.ToTensor()

class SatelliteDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        super(SatelliteDataset, self).__init__()
        self.root_dir = root_dir
        self.rgb_files = os.listdir(os.path.join(self.root_dir, "rgb"))
        self.len = num_samples if num_samples else len(self.rgb_files)

    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]
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
                # wont create a rgbb anymore
                rgb_masked[:, i, j] = torch.tensor([0,0,0])

        return [r_w, r_h, mask_size_w, mask_size_h]


    def __len__(self):
        return self.len 

    def __getitem__(self, idx):

        rgb = self.open_img(idx)
        lc = self.open_classes(idx)

        for transf in transf_types:
            if np.random.random() < 0.25:
                rgb = transf(rgb)
                lc = transf(lc)


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

    ds = SatelliteDataset("../../data/grid_dir/val")
    loader = DataLoader(ds, 4)
    i = 0
    num_examples = 5
    
    for rgb, lc, rgb_masked, masked_areas in loader:
        rgb = unprocess(rgb)
        rgb_masked = unprocess(rgb_masked)
        lc = unprocess(lc)
        lc = create_img_from_classes(lc)

        fig, ax = plt.subplots(1,3, figsize=(12,4))
        fig.tight_layout()
        
        ax[0].imshow(rgb_masked)
        ax[0].set_title("rgb_masked (input)")
        
        ax[1].imshow(lc)
        ax[1].set_title("lc (input)")

        ax[2].imshow(rgb)
        ax[2].set_title("rgb (target)")

        folder = "testsetup"
        if not os.path.exists(folder):
            os.mkdir(folder)

        plt.savefig(folder + f"/input_example_{i}.png")

        plt.imsave(os.path.join(folder, f"{i}_rgb.png"), rgb)
        plt.imsave(os.path.join(folder, f"{i}_lc.png"), lc)
        plt.imsave(os.path.join(folder, f"{i}_rgb_masked.png"), rgb_masked)

        i += 1
        if i == num_examples:
            break

def test_utils():
    from utils import save_example
    from generator import Generator
    from discriminator import Discriminator
    device = "cpu"

    d = SatelliteDataset("../../data/grid_dir/train")
    l = DataLoader(d, 3)
    g = Generator().to(device)
    discriminator = Discriminator().to(device)

    save_example(g, discriminator, "testsetup", 0, l, device, 1)

if __name__ == "__main__":
    test()
    # test_utils()