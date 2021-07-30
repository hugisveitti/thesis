import torch 
from torch.utils.data import Dataset, DataLoader
import os
import config
import numpy as np
from PIL import Image
import torchvision.transforms as T



flip_horizontal = T.RandomHorizontalFlip(p=1)
flip_vertical = T.RandomVerticalFlip(p=1)
rand_rotation_90 = T.RandomRotation(90) # -90 or 90
rand_rotation_180 = T.RandomRotation(180) # -180 or 180


transf_types = [flip_horizontal]#, flip_vertical, rand_rotation_90, rand_rotation_180]

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
        with np.load(os.path.join(self.root_dir, "lc_sieve", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = toTensor(classes)
            classes = classes.type(config.tensor_type)
        return classes
    
    def __getitem__(self, idx):

        rgb = self.open_img(idx)
        lc = self.open_classes(idx)

        images = [rgb, lc] 

        for transf in transf_types:
            if np.random.random() < 0.25:
                rgb = transf(rgb)
                lc = transf(lc)

        return rgb, lc

def test():
    import matplotlib.pyplot as plt
    from datautils import create_img_from_classes, unprocess
    ds = SatelliteDataset("../../data/grid_dir/val/")
    loader = DataLoader(ds,1)
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



