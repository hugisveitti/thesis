import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from datautils import unprocess, create_img_from_classes



def save_example(gen, loader, epoch, folder, device, print_img_stats=False):
    if not os.path.exists(folder):
        os.mkdir(folder)

    rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_ab = next(iter(loader))
    rgb_a = rgb_a.to(device)
    rgb_ab = rgb_ab.to(device)
    lc_ab = lc_ab.to(device)

    gen.eval()

    fake_img = gen(rgb_a, lc_ab)
    fake_img = fake_img.cpu()

    if torch.isnan(fake_img).any():
        print("fake img has nan")

    if print_img_stats:
        print("max value img",torch.max(img),"min" ,torch.min(img))
        print("max value fake img",torch.max(fake_img),"min", torch.min(fake_img))
        print("mean img", torch.mean(img),"mean fake", torch.mean(fake_img))
        print("std img", torch.std(img),"std fake", torch.std(fake_img))

    fake_img = unprocess(fake_img.detach())
    

    rgb_a = unprocess(rgb_a.cpu())
    lc_ab = unprocess(lc_ab.cpu(), False)
    lc_ab = create_img_from_classes(lc_ab)
    
    rgb_ab = unprocess(rgb_ab.cpu())

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0,0].imshow(rgb_a)
    ax[0,0].set_title("rgb_a")

    ax[0,1].imshow(lc_ab)
    ax[0,1].set_title("lc_ab")

    ax[1,0].imshow(rgb_ab)
    ax[1,0].set_title("rgb_ab (target)")

    ax[1,1].imshow(fake_img)
    ax[1,1].set_title("generated img")


    plt.savefig(f"{folder}/gen_{epoch}.png")
    plt.close()

    gen.train()

