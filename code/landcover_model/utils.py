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


def IoU(lc_a, lc_b, cla):
    union = lc_a[(lc_a == cla) | (lc_b == cla)].shape[0]
    if union == 0:
        return None
    return len(((lc_a == cla) | (lc_b == cla))[(lc_a == cla) & (lc_b == cla)]) / union
num_classes = 14
def calc_single_IoUs(lc_a, lc_b):
    """
    Calculates the mean IoU,
    the weights of each class depend on their total ratio in lc_a
    """
    c_ratio = []
    ious = []
    for c in range(num_classes):
        iou = IoU(lc_a, lc_b, c)
        n = lc_a.shape[0] * lc_a.shape[1]
        if iou != None:
            ious.append(iou)
            c_ratio.append(lc_a[lc_a==c].shape[0] / n)
    return np.sum(np.array(ious) * np.array(c_ratio))


def calc_all_IoUs(lc_a, lc_b):
    lc_a = torch.argmax(lc_a, dim=1)
    lc_b = torch.argmax(lc_b, dim=1)
    # lc_a.shape[0] are the batches
    return np.mean([calc_single_IoUs(lc_a[i], lc_b[i]) for i in range(lc_a.shape[0])])



def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lc_model = LandcoverModel().to(device)
    dataset = SatelliteDataset("../../data/grid_dir/val/")
    loader = DataLoader(dataset, 1)

    save_example(lc_model, loader, 4, "testsetup", device)



if __name__ == "__main__":
    test()