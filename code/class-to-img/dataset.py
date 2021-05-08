import torch 
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import torchvision.transforms as T


transform_image = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5], ),
        # Or use ImageNets mean and std?
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MyDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        self.num_samples = num_samples
        self.root_dir = root_dir
        with open(root_dir + "rgb_files.json") as f:
            self.files = json.load(f)
       # self.classes = np.load(root_dir + "classes.npz")

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        else:
            return len(self.files)
    
    def __getitem__(self, idx):

        name = self.files[idx]
        target_name = os.path.join(self.root_dir, "rgb", name)

        with Image.open(target_name) as img:
            target_img = img 
            target_img = transform_image(target_img)[:3,:,:]
        
 
        # Is this stupid, since I don't want to use RGBA, just RGB, maybe not use .png?
        
        fn = name.split(".")[0]
        with np.load(os.path.join(self.root_dir,"lc_classes",fn + ".npz")) as c:
            input_classes =  c["arr_0"]  #self.classes[fn]
            input_classes = torch.from_numpy(input_classes).type(torch.HalfTensor)
            input_classes = torch.movedim(input_classes, 2, 0)
            
        return input_classes, target_img


def test():
    ds = MyDataset("../../data/val/")
    loader = DataLoader(ds,4)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        
        break


if __name__ == "__main__":
    test()



