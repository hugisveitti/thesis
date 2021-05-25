import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import config

from datautils import unprocess, create_img_from_classes

inpaint_size = 32
# num_inpaints = 5


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

    def inpaint(self, rgb_ab, rgb_b, lc_ab, lc_b, rgb_a):
        r_w = np.random.randint(rgb_a.shape[1] - inpaint_size)
        r_h = np.random.randint(rgb_a.shape[2] - inpaint_size)

        # do something more efficient?
        # while self.calc_most_common_class(lc_ab[:,r_w:r_w+inpaint_size, r_h:r_h + inpaint_size]) == self.calc_most_common_class(lc_b[:, r_w:r_w+inpaint_size, r_h:r_h + inpaint_size]):
        #     r_w = np.random.randint(rgb_a.shape[1] - inpaint_size)
        #     r_h = np.random.randint(rgb_a.shape[2] - inpaint_size)

        for i in range(r_w, r_w + inpaint_size):
            for j in range(r_h, r_h + inpaint_size):
                if not torch.equal(lc_ab[:,i,j], lc_b[:, i,j]):
                    rgb_ab[:, i, j] = rgb_b[:, i, j]
                    lc_ab[:, i, j] = lc_b[:, i, j]
                    rgb_a[:,i,j] = torch.tensor([1,1,1])
  

    def __getitem__(self, a_idx):


        b_idx = np.random.randint(self.len)

        rgb_a = self.open_img(a_idx)
        rgb_b = self.open_img(b_idx)

        lc_a = self.open_classes(a_idx)
        lc_b = self.open_classes(b_idx)

        rgb_ab = rgb_a.clone()
        lc_ab = lc_a.clone()

        num_inpaints = np.random.randint(5, 10)
        
        for _ in range(num_inpaints):
            self.inpaint(rgb_ab, rgb_b, lc_ab, lc_b, rgb_a)

 
        # use rgb_b as the real image for the disciminator training.
        return rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_ab 


def test():
    d = SatelliteDataset("../../data/train")
    loader = DataLoader(d, 1)
    
    for rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_ab in loader:

        rgb_a = unprocess(rgb_a)
        rgb_b = unprocess(rgb_b)
        rgb_ab = unprocess(rgb_ab)
        lc_a = unprocess(lc_a, False)
        lc_b = unprocess(lc_b, False)
        lc_ab = unprocess(lc_ab, False)

        lc_a = create_img_from_classes(lc_a)
        lc_b = create_img_from_classes(lc_b)
        lc_ab = create_img_from_classes(lc_ab)

        fig, ax = plt.subplots(2,3, figsize=(15,10))

        ax[0,0].imshow(rgb_a)
        ax[0,0].set_title("rgb_a")

        ax[0,1].imshow(rgb_b)
        ax[0,1].set_title("rgb_b")

        ax[0,2].imshow(rgb_ab)
        ax[0,2].set_title("rgb_ab")

        ax[1,0].imshow(lc_a)
        ax[1,0].set_title("lc_a")

        ax[1,1].imshow(lc_b)
        ax[1,1].set_title("lc_b")

        ax[1,2].imshow(lc_ab)
        ax[1,2].set_title("lc_ab")

        test_dir = "images"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        plt.savefig(test_dir + "/input_example.png")



        plt.show()



        break
    

if __name__ == "__main__":
    test()