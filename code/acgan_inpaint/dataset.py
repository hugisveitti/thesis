import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import config

from datautils import unprocess, create_img_from_classes
from utils import save_example
from generator import Generator
from discriminator import ACDiscriminator

inpaint_size = 32
# num_inpaints = 5
masked_pixels = torch.tensor([1.,1.,1.])


normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
toTensor = T.ToTensor()

class SatelliteDataset(Dataset):

    def __init__(self, root_dir, num_samples=None):
        self.rgb_files = os.listdir(os.path.join(root_dir, "rgb"))
        self.len = num_samples if num_samples else len(self.rgb_files)
        self.root_dir = root_dir

    def __len__(self):
        return self.len

    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]
            img = normalize(img)
            img = img.type(config.torch_type)
        return img

    def open_classes(self, idx):     
        with np.load(os.path.join(self.root_dir, "lc_classes", self.rgb_files[idx].split(".")[0] + ".npz" )) as classes:
            classes = classes["arr_0"]
            classes = torch.from_numpy(classes).type(config.torch_type)
            classes = torch.movedim(classes, -1, 0)
        return classes

    def calc_most_common_class(self, classes):
        b = torch.flatten(classes, 1)
        c = torch.argmax(b,0)
        split = []
        for cla in range(14):
            split.append(c[c == cla].size()[0] / c.size()[0])
        return np.argmax(split)

    def mask(self, rgb_masked, lc):
        r_w = np.random.randint(rgb_masked.shape[1] - inpaint_size)
        r_h = np.random.randint(rgb_masked.shape[2] - inpaint_size)

        # do something more efficient?
        # while self.calc_most_common_class(lc_ab[:,r_w:r_w+inpaint_size, r_h:r_h + inpaint_size]) == self.calc_most_common_class(lc_b[:, r_w:r_w+inpaint_size, r_h:r_h + inpaint_size]):
        #     r_w = np.random.randint(rgb_a.shape[1] - inpaint_size)
        #     r_h = np.random.randint(rgb_a.shape[2] - inpaint_size)

        most_common = self.calc_most_common_class(lc[:, r_w: r_w + inpaint_size, r_h: r_h + inpaint_size])
        m_classes = torch.zeros(14)
        m_classes[most_common] = 1


        for i in range(r_w, r_w + inpaint_size):
            for j in range(r_h, r_h + inpaint_size):
                if torch.equal(lc[:,i,j], m_classes):
                    rgb_masked[:,i,j] = masked_pixels
          
        
        # cloud = torch.zeros(14)
        # cloud[0] = 1
        
        # for i in range(r_w, r_w + inpaint_size):
        #     rgb_masked[:,i,r_h] = torch.tensor([1,0,1])
        #     rgb_masked[:,i,r_h + inpaint_size-1] = torch.tensor([1,0,1])
        #     lc[:,i,r_h] = cloud
        #     lc[:,i,r_h + inpaint_size-1] = cloud
            

        # for j in range(r_h, r_h + inpaint_size):
        #     rgb_masked[:,r_w,j] = torch.tensor([1,0,1])
        #     rgb_masked[:,r_w + inpaint_size - 1, j] = torch.tensor([1,0,1])
        #     lc[:,r_w,j] = cloud
        #     lc[:,r_w + inpaint_size - 1, j] = cloud
  

    def __getitem__(self, a_idx):

        rgb = self.open_img(a_idx)

        lc = self.open_classes(a_idx)

        rgb_masked = rgb.clone()

        # should be between 3 and 17 % of area masked, use more?
        # There is randomness, could prove how many it will be at least
        num_inpaints = np.random.randint(5, 15)

        
        for _ in range(num_inpaints):
            self.mask(rgb_masked, lc)

        # c = 0
        # for i in range(256):
        #     for j in range(256):
        #         if torch.equal(rgb_masked[:,i,j], masked_pixels):
        #            c += 1
        # print(f"Ratio of masked pixels: {c / (256*256)}") 
        # print(f"c {c}")

        return rgb_masked, rgb, lc


def test():
    d = SatelliteDataset("../../data/train")
    loader = DataLoader(d, 1)
    
    for rgb_masked, rgb, lc in loader:

        rgb = unprocess(rgb)
        rgb_masked = unprocess(rgb_masked)
        lc = unprocess(lc, False)

        lc = create_img_from_classes(lc)

        fig, ax = plt.subplots(1,3, figsize=(15,10))
        fig.tight_layout()

        ax[0].imshow(rgb_masked)
        ax[0].set_title("rgb masked (input)")

        ax[1].imshow(lc)
        ax[1].set_title("lc (input)")

        ax[2].imshow(rgb)
        ax[2].set_title("rgb (target)")

        test_dir = "images"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        plt.savefig(test_dir + "/input_example.png")

        plt.show()

        break
    

def test_utils():
    ds = SatelliteDataset("../../data/train")
    l = DataLoader(ds, 1)
    g = Generator().to("cuda")
    d = ACDiscriminator(3).to("cuda")
    save_example(g, d, l, 0, "test_setup", "cuda")

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == "u":
        test_utils()
    else:
        test()