import torch 
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
import numpy as np
from PIL import Image
import json


transform_image = A.Compose(
    [
      # A.Resize(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT),
      A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255,),
      # A.HorizontalFlip(p=0.5),
      # A.VerticalFlip(p=1),
    ]
)

transform_both = A.Compose(
    [
       ToTensorV2()
    ]
)

class MyDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        self.num_samples = num_samples
        self.root_dir = root_dir
        with open(root_dir + "rgb_files.json") as f:
            self.files = json.load(f)
        self.classes = np.load(root_dir + "classes.npz")

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        else:
            return len(self.files)
    
    def __getitem__(self, idx):

        name = self.files[idx]
        input_name = os.path.join(self.root_dir, "rgb", name)

        with Image.open(input_name) as img:
            input_img = np.array(img)[:,:,:3]
        fn = name.split(".")[0]
 
        target = self.classes[fn]
 
        input_img = transform_image(image=input_img)["image"]

        input_img = transform_both(image=input_img)["image"]
        target = torch.from_numpy(target).type(torch.HalfTensor)
        target = torch.movedim(target, 2, 0)
        return input_img, target


def test():
    ds = MyDataset("../../data/val/")
    loader = DataLoader(ds,4)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        
        break


if __name__ == "__main__":
    test()



