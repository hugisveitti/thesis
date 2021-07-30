import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from generator import Generator
from discriminator import Discriminator
from datautils import unprocess, create_img_from_classes
from dataset import SatelliteDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def save_two_samples(generator, discriminator, folder, loader, device):
    # In pix2pix they talk about using the dropout as the random noise
    generator.eval()
    for m in generator.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    discriminator.eval()
    for m in discriminator.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    if not os.path.exists(folder):
        os.mkdir(folder)
        os.mkdir(os.path.join(folder,"generator"))
        os.mkdir(os.path.join(folder,"discriminator"))
        start = 0
    else:
        start = len(os.listdir(os.path.join(folder,"generator")))
    example = start
    loop = tqdm(loader)
    for rgb_a, rgb_b, lc_a, lc_b, binary_mask, lc_ab, masked_areas in loop:
        example += 1
        rgb_a, rgb_b, lc_a, lc_b, binary_mask, lc_ab, masked_areas = rgb_a.to(device), rgb_b.to(device), lc_a.to(device), lc_b.to(device), binary_mask.to(device), lc_ab.to(device), masked_areas

        with torch.cuda.amp.autocast():
            fake_img = generator(rgb_a, lc_ab, binary_mask)
            gen_lc, _ = discriminator(rgb_a)
            fake_gen_lc, _ = discriminator(fake_img)
            fake_img = fake_img.cpu()
            gen_lc = gen_lc.cpu()
            fake_gen_lc = fake_gen_lc.cpu()
            if torch.isnan(fake_img).any():
                print("fake rgb has nan") 

            fake_img = unprocess(fake_img.detach())
            fake_img = np.array(fake_img, dtype=np.float32)
            rgb_a = unprocess(rgb_a.cpu())
            lc_a = unprocess(lc_a.cpu())
            lc_a = create_img_from_classes(lc_a)
            lc_ab = unprocess(lc_ab.cpu())
            lc_ab = create_img_from_classes(lc_ab)
            binary_mask = np.array(binary_mask.cpu())[0][0]

            gen_lc = unprocess(gen_lc.detach())
            gen_lc = create_img_from_classes(gen_lc)

            fake_gen_lc = unprocess(fake_gen_lc.detach())
            fake_gen_lc = create_img_from_classes(fake_gen_lc)

            fig, ax = plt.subplots(2,2,figsize=(16,16))
            fig.tight_layout()


            ax[0,0].imshow(lc_ab)
            ax[0,0].set_title("lc_ab (input)")

            ax[0,1].imshow(binary_mask)
            ax[0,1].set_title("binary_mask (input)")

            ax[1,0].imshow(rgb_a)
            ax[1,0].set_title("rgb_a (input)")

            ax[1,1].imshow(fake_img)
            ax[1,1].set_title("generated image")
           

            plt.savefig(os.path.join(folder, "generator", f"example_{example}.png"))
            plt.close()

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            ax[0].imshow(rgb_a)
            ax[0].set_title("rgb_a (input)")
            
            ax[1].imshow(gen_lc)
            ax[1].set_title("generated lc")

            ax[2].imshow(lc_a)
            ax[2].set_title("target")

            plt.savefig(os.path.join(folder, "discriminator", f"{example}_real.png"))
            plt.close()

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            ax[0].imshow(fake_img)
            ax[0].set_title("generated img (input)")
            
            ax[1].imshow(fake_gen_lc)
            ax[1].set_title("generated lc from gen img")

            ax[2].imshow(lc_ab)
            ax[2].set_title("lc_ab (target)")

            plt.savefig(os.path.join(folder, "discriminator", f"{example}_fake.png"))
            plt.close()


device = "cpu"
num_samples = 15
ds = SatelliteDataset("../../data/grid_dir/val", num_samples)
loader = DataLoader(ds, 1)
for i in range(17, 18):
    
    results_dir = f"results/run{i}"
    print(f"results dir: {results_dir}")
    if os.path.exists(results_dir):
        os.system(f"rm -rf {results_dir}/random_examples_two_images/")

        g = Generator().to(device)
        d = Discriminator().to(device)
        try:
            g.load_state_dict(torch.load(f"{results_dir}/models/generator.pt"))
            d.load_state_dict(torch.load(f"{results_dir}/models/discriminator.pt"))

            


            save_two_samples(g, d, f"{results_dir}/random_examples_two_images/", loader, device)
        except Exception as exception:
            print("state dict probs dont fit", exception)