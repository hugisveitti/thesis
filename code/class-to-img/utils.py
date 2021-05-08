import matplotlib.pyplot as plt
import torch
import numpy as np
from generator import Generator
from dataset import MyDataset
from torch.utils.data import DataLoader

import ast
import os

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


def save_example(gen, val_loader, epoch, folder, device):
    """
    save first image in the val_loader into folder
    where input target and generated are all together

    """
    input_classes, target_img = next(iter(val_loader))
    input_classes, target_img = input_classes.to(device), target_img.to(device)
    gen.eval()
    with torch.no_grad() and torch.cuda.amp.autocast():
  
        fake_img = gen(input_classes).cpu()

        target_img = np.array(target_img.cpu())[0]
        target_img = np.moveaxis(target_img, 0, -1)
        input_classes = input_classes.cpu().numpy()[0]
        input_classes = np.moveaxis(input_classes, 0, -1)
        input_img = create_img_from_classes(np.array(input_classes))[:,:,:3]

        fake_img = fake_img.detach().numpy()[0]
        fake_img = np.moveaxis(fake_img, 0, -1)

        # undo normalization
        target_img = np.array((target_img * 0.5 + 0.5) * 255, dtype=np.uint8)
        fake_img = np.array((fake_img * 0.5 + 0.5) * 255, dtype=np.uint8)


        #input_img = np.array(input_img.cpu())[0]

        
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(input_img)
        ax[0].set_title("input")
        ax[0].axis("off")

        ax[1].imshow(target_img)
        ax[1].set_title("target")
        ax[1].axis("off")

        ax[2].imshow(fake_img)
        ax[2].set_title("generated image")
        ax[2].axis("off")

        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(f"{folder}/gen_{epoch}.png")
        plt.close()

    gen.train()


def plot_losses(losses, folder):
    plt.figure(figsize=(8,8))
    for key in losses.keys():
        plt.plot(losses[key], label=key)

    plt.legend()
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig(f"{folder}/losses.png")
    plt.close()

def save_models(generator, discriminator, epoch):
    folder = "models/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(generator.state_dict(), f"{folder}generator{epoch}.pt")
    torch.save(discriminator.state_dict(), f"{folder}discriminator{epoch}.pt")

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = Generator(14, 3).to(device)
    dataset = MyDataset("../../data/val/")
    loader = DataLoader(dataset, 1)

    save_example(g, loader, 0, "testsetup", device)



if __name__ == "__main__":
    test()