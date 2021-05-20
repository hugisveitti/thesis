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
        ma = ma * 0.5 + 0.5
        ma = np.array(ma * 255, dtype=np.uint8)
    return ma

def save_example(gen, loader, epoch, folder, device, print_img_stats=False):
    if not os.path.exists(folder):
        os.mkdir(folder)

    img, classes, mod_classes, _ = next(iter(loader))
    img, classes, mod_classes = img.to(device), classes.to(device), mod_classes.to(device)

    gen.eval()

    fake_img = gen(img, mod_classes)

    cycle_img = gen(fake_img, classes).cpu()
    id_img = gen(img, classes).cpu()
    fake_img = fake_img.cpu()
    if torch.isnan(fake_img).any():
        print("fake img has nan")

    if print_img_stats:
        print("max value img",torch.max(img),"min" ,torch.min(img))
        print("max value fake img",torch.max(fake_img),"min", torch.min(fake_img))
        print("mean img", torch.mean(img),"mean fake", torch.mean(fake_img))
        print("std img", torch.std(img),"std fake", torch.std(fake_img))

    fake_img = pp(fake_img.detach())
    
    cycle_img = pp(cycle_img.detach())
    id_img = pp(id_img.detach())

    img = pp(img.cpu())
    classes = pp(classes.cpu(), False)
    mod_classes = pp(mod_classes.cpu(), False)

    classes = create_img_from_classes(classes)
    mod_classes = create_img_from_classes(mod_classes)


    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0,0].imshow(img)
    ax[0,0].set_title("img")

    ax[0,1].imshow(classes)
    ax[0,1].set_title("classes")

    ax[0,2].imshow(mod_classes)
    ax[0,2].set_title("mod classes")

    ax[1,0].imshow(fake_img)
    ax[1,0].set_title("fake img")

    ax[1,1].imshow(cycle_img)
    ax[1,1].set_title("cycle img")

    ax[1,2].imshow(id_img)
    ax[1,2].set_title("id img")

    plt.savefig(f"{folder}/gen_{epoch}.png")
    plt.close()

    gen.train()


def save_models(generator, discriminator, epoch):
    folder = "models/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    torch.save(generator.state_dict(), f"{folder}generator_{epoch}.pt")
    torch.save(discriminator.state_dict(), f"{folder}discriminator_{epoch}.pt")

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
   
    