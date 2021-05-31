import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from datautils import unprocess, create_img_from_classes



def save_example(gen, disc, loader, epoch, folder, device, print_img_stats=False):
    if not os.path.exists(folder):
        os.mkdir(folder)

    rgb_masked, rgb, lc = next(iter(loader))
    rgb_masked = rgb_masked.to(device)
    rgb = rgb.to(device)
    lc = lc.to(device)

    gen.eval()
    disc.eval()

    fake_img = gen(rgb_masked, lc)

    fake_gen_lc, _ = disc(fake_img)
    gen_lc, _ = disc(rgb)

    fake_img = fake_img.cpu()

    if torch.isnan(fake_img).any():
        print("fake img has nan")

    if print_img_stats:
        print("max value img",torch.max(img),"min" ,torch.min(img))
        print("max value fake img",torch.max(fake_img),"min", torch.min(fake_img))
        print("mean img", torch.mean(img),"mean fake", torch.mean(fake_img))
        print("std img", torch.std(img),"std fake", torch.std(fake_img))

    with torch.no_grad():
        fake_img = unprocess(fake_img.detach())
        rgb = unprocess(rgb.cpu())
        lc = unprocess(lc.cpu(), False)
        fake_gen_lc = unprocess(fake_gen_lc.cpu(), False)
        gen_lc = unprocess(gen_lc.cpu(), False)
        rgb_masked = unprocess(rgb_masked.cpu())

    lc = create_img_from_classes(lc)
    fake_gen_lc = create_img_from_classes(fake_gen_lc)
    gen_lc = create_img_from_classes(gen_lc)
    

    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    fig.tight_layout()

    ax[0,0].imshow(rgb_masked)
    ax[0,0].set_title("rgb_masked (input)")

    ax[0,1].imshow(lc)
    ax[0,1].set_title("lc (input)")

    ax[1,0].imshow(rgb)
    ax[1,0].set_title("rgb (target)")

    ax[1,1].imshow(fake_img)
    ax[1,1].set_title("generated image")
    
    ax[2,0].imshow(gen_lc)
    ax[2,0].set_title("generated lc from target rgb")
    
    ax[2,1].imshow(fake_gen_lc)
    ax[2,1].set_title("generated lc from gen img")

    plt.savefig(f"{folder}/gen_{epoch}.png")
    plt.close()

    gen.train()
    disc.train()

def plot_losses(losses, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

    plt.figure(figsize=(8,8))

    for (key, value) in losses.items():
        plt.plot(value, label=key)

    plt.legend()
    plt.savefig(f"{folder}/losses.png")
    plt.close()