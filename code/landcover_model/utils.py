import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import SatelliteDataset
from torch.utils.data import DataLoader
from landcover_model import LandcoverModel
import ast
import os
from datautils import create_img_from_classes, unprocess

def save_example(lc_model, val_loader, epoch, folder, device):
    input_img, target_c = next(iter(val_loader))
    input_img, target_c = input_img.to(device), target_c.to(device)
    lc_model.eval()
    with torch.no_grad():
  
        fake_c = lc_model(input_img).cpu()

        target_c = unprocess(target_c.cpu().detach())
        target_img = create_img_from_classes(target_c)

        fake_c = unprocess(fake_c)
        fake_c = create_img_from_classes(fake_c)

        input_img = unprocess(input_img.cpu().detach())

        
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        fig.tight_layout()

        ax[0].imshow(input_img)
        ax[0].set_title("input")
        ax[0].axis("off")

        ax[1].imshow(target_img)
        ax[1].set_title("target")
        ax[1].axis("off")

        ax[2].imshow(fake_c)
        ax[2].set_title("generated image")
        ax[2].axis("off")

        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(f"{folder}/gen_{epoch}.png")
        plt.close()

    lc_model.train()


def plot_losses(losses, folder):
    plt.figure(figsize=(8,8))
    plt.plot(losses["mse"], label="mse")
    plt.plot(losses["g"], label="gen")
    plt.plot(losses["d"], label="disc")

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

    lc_model = LandcoverModel().to(device)
    dataset = SatelliteDataset("../../data/grid_dir/val/")
    loader = DataLoader(dataset, 1)

    save_example(lc_model, loader, 4, "testsetup", device)



if __name__ == "__main__":
    test()