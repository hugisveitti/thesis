import numpy as np
import ast
import os
import torch
import matplotlib.pyplot as plt

lc_pixels = {
    0: str([255, 255, 255, 255]),
    1: str([210, 0, 0, 255]),
    2: str([253, 211, 39, 255]),
    3: str([176, 91, 16, 255]),
    4: str([35, 152, 0, 255]),
    5: str([8, 98, 0, 255]),
    6: str([249, 150, 39, 255]),
    7: str([141, 139, 0, 255]),
    8: str([95, 53, 6, 255]),
    9: str([149, 107, 196, 255]),
    10: str([77, 37, 106, 255]),
    11: str([154, 154, 154, 255]),
    12: str([106, 255, 255, 255]),
    13: str([20, 69, 249, 255]),
}

# to revert to an image
def create_img_from_classes(img_classes):
    img = np.zeros((img_classes.shape[0], img_classes.shape[1], 4), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ast.literal_eval(lc_pixels[np.argmax(img_classes[i,j])])
    return img

def pp(ma, is_img=True):
    ma = np.array(ma)[0]
    ma = np.moveaxis(ma, 0, -1)
    if is_img:
        ma = ma*0.5 + 0.5
        ma = np.array(ma * 255, dtype=np.uint8)
    return ma

def save_example(gen, loader, epoch, folder, device):
    if not os.path.exists(folder):
        os.mkdir(folder)

    rgb_a, rgb_b, lc_ab, rgb_ab = next(iter(loader))
    rgb_a, lc_ab, rgb_ab = rgb_a.to(device), lc_ab.to(device), rgb_ab.to(device)

    gen.eval()
    #with torch.cuda.amp.autocast():
    fake_rgb = gen(rgb_a, lc_ab).cpu()
    if torch.isnan(fake_rgb).any():
        print("fake rgb has nan")

    fake_rgb = pp(fake_rgb.detach())
    rgb_a = pp(rgb_a.cpu())
    lc_ab = pp(lc_ab.cpu(), False)
    lc_ab = create_img_from_classes(lc_ab)

    rgb_ab = pp(rgb_ab.cpu())

    fig, ax = plt.subplots(2,2, figsize=(10,10))

    ax[0,0].imshow(rgb_a)
    ax[0,0].set_title("rgb_a")

    ax[0,1].imshow(lc_ab)
    ax[0,1].set_title("lc_ab")

    ax[1,0].imshow(rgb_ab)
    ax[1,0].set_title("rgb_ab")

    ax[1,1].imshow(fake_rgb)
    ax[1,1].set_title("fake rgb")

    plt.savefig(f"{folder}/gen_{epoch}.png")


    gen.train()


def save_models(generator, discriminator, epoch):
    folder = "models/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    torch.save(generator.state_dict(), f"{folder}generator{epoch}.pt")
    torch.save(discriminator.state_dict(), f"{folder}discriminator{epoch}.pt")

def plot_losses(losses, folder):
    plt.figure(figsize=(8,8))
    for key in losses.keys():
        plt.plot(losses[key], label=key)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig(f"{folder}/losses.png")
    plt.close()
   
    