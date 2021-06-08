import torch
import os
import matplotlib.pyplot as plt

from datautils import unprocess, create_img_from_classes

def save_example(generator, discriminator, folder, epoch, loader, device):
    generator.eval()
    discriminator.eval()
    if not os.path.exists(folder):
        os.mkdir(folder)

    rgb_a, lc_a, lc_ab, rgb_ab = next(iter(loader))
    rgb_a, lc_a, lc_ab, rgb_ab = rgb_a.to(device), lc_a.to(device), lc_ab.to(device), rgb_ab.to(device)

    with torch.cuda.amp.autocast():
        fake_img = generator(rgb_a, lc_ab)
        gen_lc, _ = discriminator(rgb_a, lc_a)
        fake_gen_lc, _ = discriminator(fake_img, lc_ab)
        fake_img = fake_img.cpu()
        gen_lc = gen_lc.cpu()
        fake_gen_lc = fake_gen_lc.cpu()
        if torch.isnan(fake_img).any():
            print("fake rgb has nan") 

        fake_img = unprocess(fake_img.detach())
        rgb_a = unprocess(rgb_a.cpu())
        rgb_ab = unprocess(rgb_ab.cpu())
        lc_ab = unprocess(lc_ab.cpu(), False)
        lc_ab = create_img_from_classes(lc_ab)

        gen_lc = unprocess(gen_lc.detach(), False)
        gen_lc = create_img_from_classes(gen_lc)

        fake_gen_lc = unprocess(fake_gen_lc.detach(), False)
        fake_gen_lc = create_img_from_classes(fake_gen_lc)

        fig, ax = plt.subplots(3,2, figsize=(8,12))
        fig.tight_layout()

        ax[0,0].imshow(rgb_a)
        ax[0,0].set_title("rgb_a (input)")

        ax[0,1].imshow(lc_ab)
        ax[0,1].set_title("lc_ab (input)")

        ax[1,0].imshow(rgb_ab)
        ax[1,0].set_title("rgb_ab (target)")

        ax[1,1].imshow(fake_img)
        ax[1,1].set_title("generate image")

        ax[2,0].imshow(gen_lc)
        ax[2,0].set_title("gen lc of rgb_a img")

        ax[2,1].imshow(fake_gen_lc)
        ax[2,1].set_title("gen lc of gen img")

        plt.savefig(f"{folder}/gen_{epoch}.png")

        plt.close()

    generator.train() 
    discriminator.train()