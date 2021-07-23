import torch 
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import config
import numpy as np
from PIL import Image
import json
import torchvision.transforms as T
from datautils import create_img_from_classes, unprocess


flip_horizontal = T.RandomHorizontalFlip(p=1)
flip_vertical = T.RandomVerticalFlip(p=1)
color_jitter = T.ColorJitter()
toTensor = T.ToTensor()

class SatelliteDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        self.num_samples = num_samples
        self.root_dir = root_dir
        self.rgb_files = os.listdir(os.path.join(root_dir, "rgb"))

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        else:
            return len(self.rgb_files)

    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]
        return img

    def open_classes(self, idx):
        fn = self.rgb_files[idx].split(".")[0]
        with np.load(os.path.join(self.root_dir, "reduced_classes", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = toTensor(classes)
            classes = classes.type(config.tensor_type)
        return classes
    
    def __getitem__(self, idx):


        rgb = self.open_img(idx)
        lc = self.open_classes(idx)


        if np.random.random() < 0.5:
            rgb = flip_horizontal(rgb)
            lc = flip_horizontal(lc)

        if np.random.random() < 0.5:
            rgb = flip_vertical(rgb)
            lc = flip_vertical(lc)

        # if np.random.random() < 0.5:
        #     rgb = color_jitter(rgb)

        return rgb, lc

def test():
    ds = SatelliteDataset("../../data/grid_dir/val/")
    loader = DataLoader(ds,4)
    for rgb, lc in loader:
        
        rgb = unprocess(rgb)
        lc = unprocess(lc)
        lc = create_img_from_classes(lc)

        fig,ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(rgb)
        ax[0].set_title("rgb (input)")

        ax[1].imshow(lc)
        ax[1].set_title("lc (target)")

        folder = "testsetup"

        if not os.path.exists(folder):
            os.mkdir(folder)
        
        plt.savefig(os.path.join(folder, "input_example.png"))
        plt.close

        
        break


if __name__ == "__main__":
    test()



