import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchvision.transforms as T
from PIL import Image
from utils import create_img_from_classes, save_example
import matplotlib.pyplot as plt
from generator import Generator
import config


normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
toTensor = T.ToTensor()


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

    def __len__(self):
        return self.len
    
    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]#.type(torch.HalfTensor)
            #img = torch.Tensor(np.array(img)[:,:,:3])
            #img = torch.movedim(img, -1, 0)
            img = normalize(img)
            img = img.type(config.torch_type)
        return img

    def open_classes(self, idx):
        fn = self.rgb_files[idx].split(".")[0]
        with np.load(os.path.join(self.root_dir, "lc_classes", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = torch.from_numpy(classes).type(config.torch_type)
            classes = torch.movedim(classes, -1, 0)
        return classes

    def __getitem__(self, a_idx):

        # random index
        b_idx = np.random.randint(self.len)

        rgb_a = self.open_img(a_idx)
        lc_a = self.open_classes(a_idx)
        rgb_b = self.open_img(b_idx)
        lc_b = self.open_classes(b_idx)
        
        # inpainted region is 64x64 in the middle

        size = config.size
        start = config.start

        rgb_ab = rgb_a.clone()
        rgb_ab[:, start:start+size, start:start+size] = rgb_b[:, start:start+size, start:start+size].clone()

        lc_ab = lc_a.clone()
        lc_ab[:, start:start+size, start:start+size] = lc_b[:, start:start+size, start:start+size].clone()
        

        return rgb_a, rgb_b, lc_ab, rgb_ab  

def pp(ma, is_img=True):
    ma = np.array(ma)[0]
    ma = np.moveaxis(ma, 0, -1)
    if is_img:
        ma = ma*0.5 + 0.5
        ma = np.array(ma * 255, dtype=np.uint8)
    return ma

def test():
    d = SatelliteDataset("../../data/train")
    loader = DataLoader(d, 2)
    for rgb_a, rgb_b, lc_ab, rgb_ab in loader:
        print(rgb_a.shape)
        print(f"classes shape, {lc_ab.shape}")
        rgb_a = pp(rgb_a)
        rgb_b = pp(rgb_b)
        lc_ab = pp(lc_ab, False)
        rgb_ab = pp(rgb_ab)

        i_lc_ab = create_img_from_classes(lc_ab)
        
        fig, ax = plt.subplots(2, 2, figsize=(10,10))

        ax[0,0].imshow(rgb_a)
        ax[0,0].set_title("rgb_a")
        ax[0,1].imshow(rgb_b)
        ax[0,1].set_title("rgb_b")
        ax[1,0].imshow(i_lc_ab)
        ax[1,0].set_title("lc_ab")
        ax[1,1].imshow(rgb_ab)
        ax[1,1].set_title("rgb_ab")
        

        # plt.show()
        if not os.path.exists("images"):
            os.mkdir("images")
        plt.savefig("images/input_example.png")
        
        break


def test_utils():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = Generator().to(device)
    g.load_state_dict(torch.load("models/generator9.pt"))
    ds = SatelliteDataset("../../data/train")
    loader = DataLoader(ds, 1)
    save_example(g, loader, 0, "testsetup", device)


if __name__ == "__main__":
    # test_utils()
    test()
    
