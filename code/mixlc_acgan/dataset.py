import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np 
import os

import config

flip_horizontal = T.RandomHorizontalFlip(p=1)

flip_vertical = T.RandomVerticalFlip(p=1)


normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
            img = normalize(img)
            # img = img.type(config.tensor_type)
        return img

    def open_classes(self, idx):
        fn = self.rgb_files[idx].split(".")[0]
        with np.load(os.path.join(self.root_dir, "lc_classes", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = toTensor(classes)
            classes = classes.type(config.tensor_type)
        return classes

    def create_modification(self, rgb_b, rgb_ab, lc_a, lc_b, lc_ab, lc_b_mask):
        
        mask_size_w = np.random.randint(32, 64)
        mask_size_h = np.random.randint(32, 64)
        
        r_w = np.random.randint(rgb_ab.shape[1] - mask_size_w)
        r_h = np.random.randint(rgb_ab.shape[2] - mask_size_h)

        for i in range(r_w, r_w + mask_size_w):
            for j in range(r_h, r_h + mask_size_h):
                if not torch.equal(lc_a[:,i,j] , lc_b[:,i,j]):
                    lc_b_mask[:,i,j] = lc_b[:,i,j]
                    rgb_ab[:,i,j] = rgb_b[:,i,j]
                    lc_ab[:,i,j] = lc_b[:,i,j]
        


        return [r_w, r_h, mask_size_w, mask_size_h]
        # Just for illustration
        # put boxes around changed area.
        for i in range(r_w, r_w + mask_size_w):
            rgb_ab[:,i,r_h] = torch.tensor([1,1,1])
            rgb_ab[:,i,r_h + mask_size_h] = torch.tensor([1,1,1])
            lc_ab[:,i,r_h] = torch.zeros(14)
            lc_ab[0,i,r_h] = 1
            lc_ab[:,i,r_h+mask_size_h] = torch.zeros(14)
            lc_ab[0,i,r_h + mask_size_h] = 1
        
        for j in range(r_h, r_h + mask_size_h):
            rgb_ab[:, r_w, j] = torch.tensor([1,1,1])
            rgb_ab[:, r_w + mask_size_w, j] = torch.tensor([1,1,1])
            lc_ab[:,r_w,j] = torch.zeros(14)
            lc_ab[0,r_w,j] = 1
            lc_ab[:,r_w+mask_size_w, j] = torch.zeros(14)
            lc_ab[0,r_w + mask_size_w,j] = 1



    def __len__(self):
        return self.len 

    def __getitem__(self, idx_a):
        idx_b = np.random.randint(self.len)

        rgb_a = self.open_img(idx_a)
        lc_a = self.open_classes(idx_a)

        rgb_b = self.open_img(idx_b)
        lc_b = self.open_classes(idx_b)

        if np.random.random() < 0.5:
            rgb_a = flip_horizontal(rgb_a)
            rgb_b = flip_horizontal(rgb_b)
            lc_a = flip_horizontal(lc_a)
            lc_b = flip_horizontal(lc_b)

        if np.random.random() < 0.5:
            rgb_a = flip_vertical(rgb_a)
            rgb_b = flip_vertical(rgb_b)
            lc_a = flip_vertical(lc_a)
            lc_b = flip_vertical(lc_b)

        lc_b_mask = torch.zeros_like(lc_b)
        rgb_ab = rgb_a.clone()
        lc_ab = lc_a.clone()
        num_inpaints = 8 #np.random.randint(5,15)
        masked_areas = []
        for _ in range(num_inpaints):
            masked_area = self.create_modification(rgb_b, rgb_ab, lc_a, lc_b, lc_b_mask, lc_ab)
            masked_areas.append(masked_area)

      
        # use a third sample as the "real" sample for the discriminator ?
     #   return rgb_a, lc_a, lc_b_mask, rgb_ab

        return rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_b_mask, lc_ab, masked_areas


def test():
    from datautils import unprocess, create_img_from_classes
    import matplotlib.pyplot as plt

    ds = SatelliteDataset("../../data/train")
    loader = DataLoader(ds, 1)
    i = 0
    num_examples = 5
    
    for rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_b_mask, masked_areas in loader:
        rgb_a = unprocess(rgb_a)
        rgb_b = unprocess(rgb_b)
        rgb_ab = unprocess(rgb_ab)
        lc_a = unprocess(lc_a, False)
        lc_b = unprocess(lc_b, False)
        lc_b_mask = unprocess(lc_b_mask, False)
        lc_a = create_img_from_classes(lc_a)
        lc_b = create_img_from_classes(lc_b)
        lc_b_mask = create_img_from_classes(lc_b_mask)

        fig, ax = plt.subplots(2,3, figsize=(12,8))
        fig.tight_layout()
        
        ax[0,0].imshow(rgb_a)
        ax[0,0].set_title("rgb_a (input)")
        
        ax[0,1].imshow(rgb_b)
        ax[0,1].set_title("rgb_b")
        
        ax[0,2].imshow(rgb_ab)
        ax[0,2].set_title("rgb_b (target)")

        ax[1,0].imshow(lc_a)
        ax[1,0].set_title("lc_a (input)")

        ax[1,1].imshow(lc_b)
        ax[1,1].set_title("lc_b")

        ax[1,2].imshow(lc_b_mask)
        ax[1,2].set_title("lc_b_mask (input)")

        folder = "testsetup"
        if not os.path.exists(folder):
            os.mkdir("images")

        plt.savefig(folder + f"/input_example_{i}.png")

        i += 1
        if i == num_examples:
            break

def test_utils():
    from utils import save_example
    from generator import Generator
    from ac_discriminator import Discriminator
    device = "cuda"

    d = SatelliteDataset("../../data/train")
    l = DataLoader(d, 1)
    g = Generator().to(device)
    discriminator = Discriminator().to(device)

    save_example(g,discriminator, "testsetup", 0, l, device)

if __name__ == "__main__":
    #test()
    test_utils()