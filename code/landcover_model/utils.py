import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from dataset import SatelliteDataset
from torch.utils.data import DataLoader
from landcover_model import LandcoverModel
import ast
import os
from datautils import create_img_from_classes, unprocess

def save_example(lc_model, val_loader, epoch, folder, device, num_examples=1):
    if not os.path.exists(folder):
        os.mkdir(folder)
    example = 0
    loop = tqdm(val_loader)
    
    for input_img, target_c in loop:
   
        if example >= num_examples:
            break
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

            
            plt.savefig(f"{folder}/gen_{epoch}_{example}.png")
            plt.close()
        example += 1

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
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "results/landcover_run2/models/lc_model.pt"
    lc_model = LandcoverModel().to(device)
    lc_model.load_state_dict(torch.load(model_dir))
    num_examples = 50
    dataset = SatelliteDataset("../../data/grid_dir/val/", num_examples)
    loader = DataLoader(dataset, 1)

    save_example(lc_model, loader, 50, "results/landcover_run2/random_examples", device, num_examples)



if __name__ == "__main__":
    test()


def calc_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    target = torch.argmax(target, dim=1)
    pred = pred.flatten()
    target = target.flatten() 
    return torch.tensor(target[target == pred].shape[0] / target.shape[0], requires_grad=True)