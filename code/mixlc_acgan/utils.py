import torch
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datautils import unprocess, create_img_from_classes

def save_example(generator, discriminator, folder, epoch, loader, device):
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

    rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_b_mask,lc_ab, masked_areas = next(iter(loader))
    rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_b_mask = rgb_a.to(device), rgb_b.to(device), rgb_ab.to(device), lc_a.to(device), lc_b.to(device), lc_b_mask.to(device)

    with torch.cuda.amp.autocast():
        fake_img = generator(rgb_a, lc_a, lc_b_mask)
        gen_lc, _ = discriminator(rgb_a)
        fake_gen_lc, _ = discriminator(fake_img)
        fake_img = fake_img.cpu()
        gen_lc = gen_lc.cpu()
        fake_gen_lc = fake_gen_lc.cpu()
        if torch.isnan(fake_img).any():
            print("fake rgb has nan") 

        fake_img = unprocess(fake_img.detach())
        rgb_a = unprocess(rgb_a.cpu())
        rgb_ab = unprocess(rgb_ab.cpu())
        lc_a = unprocess(lc_a.cpu(), False)
        lc_a = create_img_from_classes(lc_a)
        lc_b_mask = unprocess(lc_b_mask.cpu(), False)
        lc_b_mask = create_img_from_classes(lc_b_mask)

        gen_lc = unprocess(gen_lc.detach(), False)
        gen_lc = create_img_from_classes(gen_lc)

        fake_gen_lc = unprocess(fake_gen_lc.detach(), False)
        fake_gen_lc = create_img_from_classes(fake_gen_lc)

        fig = plt.figure(figsize=(16,16))
        gs = GridSpec(3,6, figure=fig)
        fig.tight_layout()

        ax1 = fig.add_subplot(gs[0,0:2])
        ax2 = fig.add_subplot(gs[0,2:4])
        ax3 = fig.add_subplot(gs[0,4:6])

        ax1.imshow(rgb_a)
        ax1.set_title("rgb_a (input)")

        ax2.imshow(lc_a)
        ax2.set_title("lc_a (input)")

        ax3.imshow(lc_b_mask)
        ax3.set_title("lc_b_mask (input)")

        ax4 = fig.add_subplot(gs[1, :3])
        ax5 = fig.add_subplot(gs[1,3:])

        ax4.imshow(fake_img)
        ax4.set_title("generate image")

        ax5.imshow(rgb_ab)
        ax5.set_title("rgb_ab (target)")

        ax6 = fig.add_subplot(gs[2, :3])
        ax7 = fig.add_subplot(gs[2, 3:])

        ax6.imshow(fake_gen_lc)
        ax6.set_title("gen lc of gen img")

        ax7.imshow(gen_lc)
        ax7.set_title("gen lc of rgb_a")

        plt.savefig(f"{folder}/gen_{epoch}.png")

        plt.close()

    generator.train() 
    discriminator.train()