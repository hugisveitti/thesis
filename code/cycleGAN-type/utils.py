import matplotlib.pyplot as plt
import torch
import numpy as np
from generator import Generator
from dataset import SatelliteDataset
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


def save_example(classes_gen, img_gen, loader, epoch, folder, device):
    """
    save first image in the val_loader into folder
    where input target and generated are all together

    """
    img, classes = next(iter(loader))
    img, classes = img.to(device), classes.to(device)
    classes_gen.eval()
    img_gen.eval()
    with torch.no_grad() and torch.cuda.amp.autocast():
  
        fake_img = img_gen(classes).cpu()
        fake_classes = classes_gen(img).cpu()

        img = np.array(img.cpu())[0]
        img = np.moveaxis(img, 0, -1)

        classes = classes.cpu().numpy()[0]
        classes = np.moveaxis(classes, 0, -1)
        classes_img = create_img_from_classes(np.array(classes))[:,:,:3]

        fake_img = fake_img.detach().numpy()[0]
        fake_img = np.moveaxis(fake_img, 0, -1)

        # undo normalization
        img = np.array((img * 0.5 + 0.5) * 255, dtype=np.uint8)
        fake_img = np.array((fake_img * 0.5 + 0.5) * 255, dtype=np.uint8)

        fake_classes = fake_classes.detach().numpy()[0]
        fake_classes = np.moveaxis(fake_classes, 0, -1)
        fake_classes_img = create_img_from_classes(np.array(fake_classes))[:,:,:3]


        #input_img = np.array(input_img.cpu())[0]

        
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        ax[0,0].imshow(img)
        ax[0,0].set_title("input rgb")
        ax[0,0].axis("off")

        ax[1,1].imshow(fake_img)
        ax[1,1].set_title("generated rgb")
        ax[1,1].axis("off")

        ax[1,0].imshow(classes_img)
        ax[1,0].set_title("input lc")
        ax[1,0].axis("off")

        ax[0,1].imshow(fake_classes_img)
        ax[0,1].set_title("generated lc classes")
        ax[0,1].axis("off")

        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(f"{folder}/gen_{epoch}.png")
        plt.close()

    classes_gen.train()
    img_gen.train()


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

def save_models(classes_generator, img_generator, classes_discriminator, img_discriminator, epoch):
    folder = "models/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    torch.save(classes_generator.state_dict(), f"{folder}classes_generator_{epoch}.pt")
    torch.save(img_generator.state_dict(), f"{folder}img_generator_{epoch}.pt")
    torch.save(classes_discriminator.state_dict(), f"{folder}classes_discriminator_{epoch}.pt")
    torch.save(img_discriminator.state_dict(), f"{folder}img_discriminator_{epoch}.pt")
    

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_g = Generator(14, 3).to(device)
    classes_g = Generator(3, 14).to(device)
    dataset = SatelliteDataset("../../data/val/")
    loader = DataLoader(dataset, 1)

    save_example(classes_g, img_g, loader, 1, "testsetup", device)



if __name__ == "__main__":
    test()