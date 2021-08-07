from tqdm import tqdm
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms as T


from code.model.generator import Generator
from code.inpaint.generator import Generator as InpaintGenerator
from .datautils import unprocess, create_img_from_classes, calc_all_IoUs
from code.landcover_model.landcover_model import LandcoverModel

toTensor = T.ToTensor()

tensor_type = torch.FloatTensor
local_area_margin = 8
num_inpaints = 4
device = "cuda"
num_classes = 9

other_class = torch.zeros(num_classes)
other_class[0] = 1


class MyDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rgb_files = os.listdir(os.path.join(self.root_dir, "rgb"))

    def open_img(self, idx):
        with Image.open(os.path.join(self.root_dir, "rgb", self.rgb_files[idx])) as img:
            img = toTensor(img)[:3,:,:]
        return img

    def open_classes(self, idx):
        fn = self.rgb_files[idx].split(".")[0]
        with np.load(os.path.join(self.root_dir, "lc_sieve", fn + ".npz")) as classes:
            classes = classes["arr_0"]
            classes = toTensor(classes)
            classes = classes.type(tensor_type)
        return classes

    def __len__(self):
        return len(self.rgb_files)


    def create_mask(self, lc_a, lc_b, lc_ab, binary_mask, rgb_b, rgb_ab, rgb_a_masked):
        
        mask_size_w = np.random.randint(32, 64)
        mask_size_h = np.random.randint(32, 64)
        
        r_w = np.random.randint(local_area_margin, lc_ab.shape[1] - mask_size_w - local_area_margin)
        r_h = np.random.randint(local_area_margin, lc_ab.shape[2] - mask_size_h - local_area_margin)

        # if squared mask then the binary mask will always 1
        # if not squared then the binary mask will 0 if lc_a[i,j] == lc_b[i,j]
        # That is we will sometimes ask for the generated image not to be changed under the mask
        # if the lc's match. I believe this will make the model more robust to different inpainting shapes.
        squared_mask = np.random.random() < 0.5

        for i in range(r_w, r_w + mask_size_w):
            for j in range(r_h, r_h + mask_size_h):
                if not torch.equal(lc_b[:,i,j], other_class):
                    if squared_mask:
                        lc_ab[:,i,j] = lc_b[:,i,j]
                        binary_mask[0, i, j] = 1
                        rgb_a_masked[:, i, j] = torch.zeros(3)
                        if not torch.equal(lc_a[:,i,j], lc_b[:,i,j]):
                            rgb_ab[:, i, j] = rgb_b[:, i, j]
                    elif not torch.equal(lc_a[:,i,j], lc_b[:,i,j]):
                        lc_ab[:,i,j] = lc_b[:,i,j]
                        binary_mask[0, i, j] = 1
                        rgb_ab[:, i, j] = rgb_b[:, i, j]
                        rgb_a_masked[:, i, j] = torch.zeros(3)
        



    def __getitem__(self, idx_a):
        idx_b = (idx_a + 200) % self.__len__()

        rgb_a = self.open_img(idx_a)
        lc_a = self.open_classes(idx_a)

        rgb_b = self.open_img(idx_b)
        lc_b = self.open_classes(idx_b)

        rgb_a_masked = rgb_a.clone()

        lc_ab = lc_a.clone()
        rgb_ab = rgb_a.clone()
        binary_mask = torch.zeros(1, lc_b.shape[1], lc_b.shape[2])

        for _ in range(num_inpaints):
            self.create_mask(lc_a, lc_b, lc_ab, binary_mask, rgb_b, rgb_ab, rgb_a_masked)

        return rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab, rgb_a_masked

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data/grid_dir/val")
parser.add_argument("--inpaint_dir", type=str, default="./code/inpaint/results/inpaint_run6")
parser.add_argument("--model_dir1", type=str, default="./code/model/results/run35")
parser.add_argument("--model_dir2", type=str, default="./code/model/results/run36")
parser.add_argument("--landcover_model_dir", type=str, default="./code/landcover_model/")
parser.add_argument("--gen1_name", type=str, default="1")
parser.add_argument("--gen2_name", type=str, default="2")
parser.add_argument("--num_save_imgs", type=int, default=10)
parser.add_argument("--shuffle_dataset", dest="shuffle_dataset", action="store_true")
parser.set_defaults(shuffle_dataset=False)

args = parser.parse_args()
print(args)
def run_test():
    save_dir = "test_images"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ds = MyDataset(args.data_dir)
    # shuffle true to get random values?
    loader = DataLoader(ds, 1, shuffle=args.shuffle_dataset)

    landcover_model = LandcoverModel().to(device)
    landcover_model.load_state_dict(torch.load(os.path.join(args.landcover_model_dir, "models","lc_model.pt")))

    inpaint_generator = InpaintGenerator().to(device) 
    inpaint_generator.load_state_dict(torch.load(os.path.join(args.inpaint_dir, "models","generator.pt")))


    generator1 = Generator().to(device)
    generator1.load_state_dict(torch.load(os.path.join(args.model_dir1, "models","generator.pt")))


    generator2 = Generator().to(device)
    generator2.load_state_dict(torch.load(os.path.join(args.model_dir2, "models","generator.pt")))
    
    inpaint_generator.eval()
    generator1.eval()
    generator2.eval()
    landcover_model.eval()

    for m in inpaint_generator.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    for m in generator1.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    for m in generator2.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    gen1_IoU = []
    gen2_IoU = []
    inpaint_IoU = []
    total = 0

    example_n = 0
    for rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab, rgb_a_masked in tqdm(loader):
        rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab, rgb_a_masked = rgb_a.to(device), rgb_ab.to(device), lc_a.to(device), lc_b.to(device), binary_mask.to(device), lc_ab.to(device), rgb_a_masked.to(device)


        with torch.no_grad():
            inpaint_img = inpaint_generator(rgb_a_masked, lc_ab)
            fake_img1 = generator1(rgb_a, lc_ab, binary_mask)
            fake_img2 = generator2(rgb_a, lc_ab, binary_mask)

            gen_lc1 = landcover_model(fake_img1)
            gen_lc2 = landcover_model(fake_img2)
            gen_lc_inpaint = landcover_model(inpaint_img)
            
            gen1_IoU.append(calc_all_IoUs(gen_lc1, lc_ab))
            gen2_IoU.append(calc_all_IoUs(gen_lc2, lc_ab))
            inpaint_IoU.append(calc_all_IoUs(gen_lc_inpaint, lc_ab))
           
         

            if example_n < args.num_save_imgs:
                fake_img1 = unprocess(fake_img1)
                fake_img2 = unprocess(fake_img2)

                rgb_a_masked = unprocess(rgb_a_masked)
                rgb_a = unprocess(rgb_a)
                rgb_ab = unprocess(rgb_ab)
                inpaint_img = unprocess(inpaint_img)
                lc_ab = create_img_from_classes(unprocess(lc_ab))
                lc_a = create_img_from_classes(unprocess(lc_a))
                binary_mask = np.array(binary_mask[0][0].cpu())

                plt.imsave(os.path.join(save_dir, f"{example_n}_lc_a.png"), lc_a)
                plt.imsave(os.path.join(save_dir, f"{example_n}_lc_ab.png"), lc_ab)
                plt.imsave(os.path.join(save_dir, f"{example_n}_rgb_a.png"), rgb_a)
                plt.imsave(os.path.join(save_dir, f"{example_n}_rgb_ab.png"), rgb_ab)
                plt.imsave(os.path.join(save_dir, f"{example_n}_binary_mask.png"), binary_mask)
                plt.imsave(os.path.join(save_dir, f"{example_n}_fake_img1.png"), fake_img1)
                plt.imsave(os.path.join(save_dir, f"{example_n}_fake_img2.png"), fake_img2)
                plt.imsave(os.path.join(save_dir, f"{example_n}_inpaint_img.png"), inpaint_img)
                plt.imsave(os.path.join(save_dir, f"{example_n}_rgb_a_masked.png"), rgb_a_masked)
                
                
                fig, ax = plt.subplots(1,3, figsize=(12,4))
                fig.tight_layout()
                ax[0].imshow(rgb_a_masked)
                ax[1].imshow(lc_ab)
                ax[2].imshow(inpaint_img)
                ax[0].set_title("rgb_a_masked")
                ax[1].set_title("lc_ab")
                ax[2].set_title("inpaint_img")

                plt.savefig(os.path.join(save_dir, f"{example_n}_inpaint.png"))
                plt.close()

                fig, ax = plt.subplots(2,2, figsize=(8,8))
                fig.tight_layout()
                ax[0,0].imshow(rgb_a)
                ax[0,1].imshow(lc_ab)
                ax[1,0].imshow(binary_mask)
                ax[1,1].imshow(fake_img1)
                ax[0,0].set_title("rgb_a")
                ax[0,1].set_title("lc_ab")
                ax[1,0].set_title("binary_mask")
                ax[1,1].set_title("fake_img")

                plt.savefig(os.path.join(save_dir, f"{example_n}_{args.gen1_name}.png"))
                plt.close()

                fig, ax = plt.subplots(2,2, figsize=(8,8))
                fig.tight_layout()
                ax[0,0].imshow(rgb_a)
                ax[0,1].imshow(lc_ab)
                ax[1,0].imshow(binary_mask)
                ax[1,1].imshow(fake_img2)
                ax[0,0].set_title("rgb_a")
                ax[0,1].set_title("lc_ab")
                ax[1,0].set_title("binary_mask")
                ax[1,1].set_title("fake_img")

                plt.savefig(os.path.join(save_dir, f"{example_n}_{args.gen2_name}.png"))
                plt.close()

                fig, ax = plt.subplots(1,3, figsize=(12,4))
                fig.tight_layout()
                ax[0].imshow(inpaint_img)
                ax[0].set_title("inpaint")
                ax[1].imshow(fake_img1)
                ax[1].set_title(args.gen1_name)
                ax[2].imshow(fake_img2)
                ax[2].set_title(args.gen2_name)

                plt.savefig(os.path.join(save_dir, f"{example_n}_all.png"))
                plt.close()

                gen_lc1 = create_img_from_classes(unprocess(gen_lc1))
                gen_lc2 = create_img_from_classes(unprocess(gen_lc2))
                gen_lc_inpaint = create_img_from_classes(unprocess(gen_lc_inpaint))

                fig, ax = plt.subplots(1,4, figsize=(16,4))
                fig.tight_layout()
                ax[0].imshow(gen_lc_inpaint)
                ax[0].set_title("inpaint")
                ax[1].imshow(gen_lc1)
                ax[1].set_title(args.gen1_name)
                ax[2].imshow(gen_lc2)
                ax[2].set_title(args.gen2_name)
                ax[3].imshow(lc_ab)
                ax[3].set_title("lc_ab")

                plt.savefig(os.path.join(save_dir, f"{example_n}_all_lc.png"))
                plt.close()

            

            example_n += 1
        if args.num_save_imgs == example_n:
            break

    print("#### IoUs ####")
    print(args.gen1_name, np.mean(gen1_IoU) )
    print(args.gen2_name, np.mean(gen2_IoU))
    print("inpaint", np.mean(inpaint_IoU))
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    fig.tight_layout()
    ax[0].hist(gen1_IoU, 20)
    ax[0].set_title(args.gen1_name)
    ax[1].hist(gen2_IoU, 20)
    ax[1].set_title(args.gen2_name)
    ax[2].hist(inpaint_IoU, 20)
    ax[2].set_title("inpaint")

    plt.savefig(os.path.join(save_dir, "Iou_hist.pdf"))




    


if __name__ == "__main__":
    run()