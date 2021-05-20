import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from utils import create_img_from_classes, save_example
import config
from generator import Generator



normalize = T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5), inplace=True)
toTensor = T.ToTensor()
size = config.size
start = config.start
num_classes = 14


class SatelliteDataset(Dataset):

    def __init__(self, root_dir, num_samples = None) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.root_dir = root_dir
        self.rgb_files = os.listdir(os.path.join(self.root_dir, "rgb"))
        if num_samples:
            self.len = self.num_samples
        else:
            self.len = len(self.rgb_files)
        self.create_mapping()

    def __len__(self):
        return self.len

    def create_mapping(self):
        self.class_map = {}

        classes = list(range(1, num_classes))
        # don't use these classes in the mapping
        self.class_map[0] = 0
        for i in range(1, num_classes):
            r = random.choice(classes)
            # Possibility of mapping onto itself
            # while r == i and i < num_classes - 1:
            #     r = random.choice(classes)
            self.class_map[i] = r
            classes.remove(r)


    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]
            img = normalize(img)
        return img

    def open_classes(self, idx):
        fn = self.rgb_files[idx].split(".")[0]
        with np.load(os.path.join(self.root_dir, "lc_classes", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = torch.Tensor(classes)
            classes = torch.movedim(classes, -1, 0)
        return classes

    def map_classes(self, classes):
        _, m, n = classes.shape
        for i in range(m):
            for j in range(n):
                c = torch.argmax(classes[:,i,j]).item()
                r = self.class_map[c] 
                classes[:, i, j] = torch.zeros(14)

                classes[r, i, j] = 1
        return classes
        

    def __getitem__(self, idx):
        if idx == 0:
            # Create new mapping each epoch
            self.create_mapping()

        img = self.open_img(idx)
        classes = self.open_classes(idx)
 
        # inpainted region is 64x64 in the middle
        w_s = np.random.randint(256 - size)
        h_s = np.random.randint(256 - size)
        mod_classes = classes.clone()
        self.map_classes(mod_classes[:, w_s:w_s+size, h_s:h_s+size])  
        
        return img, classes, mod_classes, (w_s, h_s)

def pp(ma, is_img=True):
    ma = np.array(ma)[0]
    ma = np.moveaxis(ma, 0, -1)
    if is_img:
        ma = ma*0.5 + 0.5
        ma = np.array(ma * 255, dtype=np.uint8)
    return ma

def test():
    d = SatelliteDataset("../../data/train")
    loader = DataLoader(d, 1)
    for img, classes, mod_classes, _ in loader:
        print(img.shape)
        print(f"classes shape, {classes.shape}")
        img = pp(img)
        classes = pp(classes, False)
        mod_classes = pp(mod_classes, False)

        classes = create_img_from_classes(classes)
        mod_classes = create_img_from_classes(mod_classes)
        
        fig, ax = plt.subplots(1, 3, figsize=(12,4))

        ax[0].imshow(img)
        ax[0].set_title("rgb")
        ax[1].imshow(classes)
        ax[1].set_title("lc")
        ax[2].imshow(mod_classes)
        ax[2].set_title("mod lc")

        if not os.path.exists("images"):
            os.mkdir("images")
        plt.suptitle("input example")
        plt.savefig("images/input_example.png")

        plt.show()
        
        break
    print(d.class_map)


def test_utils():
    print("Testing utils")
    d = SatelliteDataset("../../data/train")
    l = DataLoader(d, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = Generator(True).to(device)
    save_example(g, l, 1, "testsetup", device, True)




if __name__ == "__main__":
    test()
    #test_utils()
    
