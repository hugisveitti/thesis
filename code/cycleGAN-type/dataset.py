import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T
from PIL import Image
import numpy as np



flip_horizontal = T.RandomHorizontalFlip(p=1)

flip_vertical = T.RandomVerticalFlip(p=1)


normalize = T.Normalize(mean=(0.5,0.5,0.5), std=(0.5, 0.5, 0.5))

class SatelliteDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        super().__init__()
        self.num_samples = num_samples
        self.root_dir = root_dir
        self.rgb_files = os.listdir(os.path.join(self.root_dir, "rgb"))
        

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_fn = self.rgb_files[idx]
        fn = rgb_fn.split(".")[0]
        rgb_fn = os.path.join(self.root_dir, "rgb", fn + ".png")
        lc_fn = os.path.join(self.root_dir, "lc_classes", fn + ".npz")

        with np.load(lc_fn) as classes:
            classes = torch.tensor(classes["arr_0"]).type(torch.HalfTensor)
            

        with Image.open(rgb_fn) as img:
            img = torch.tensor(np.array(img)[:,:,:3] / 255).type(torch.HalfTensor) # transform_classes(img)[:3,:,:]
        
        img = torch.movedim(img, -1, 0)
        classes = torch.movedim(classes, -1, 0)
        if np.random.random() < 0.5:
            img = flip_horizontal(img)
            classes = flip_horizontal(classes)

        if np.random.random() < 0.5:
            img = flip_vertical(img)
            classes = flip_vertical(classes)

        img = normalize(img)

        return img, classes

def test():
    d = SatelliteDataset("../../data/train")
    loader = DataLoader(d, 2)
    for img, classes in loader:

        print(img.shape)
        print(classes.shape)

        break


if __name__ == "__main__":
    test()


