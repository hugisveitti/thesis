import torch
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator
from datautils import unprocess, create_img_from_classes
from dataset import SatelliteDataset
from utils import save_example




def run_test():

    device = "cuda"

    g = Generator().to(device)
    d = Discriminator().to(device)

    g.load_state_dict(torch.load("results4/models/generator.pt"))
    d.load_state_dict(torch.load("results4/models/discriminator.pt"))

    ds = SatelliteDataset("../../data/val")
    loader = DataLoader(ds, 1, shuffle=True)
    
    save_example(g, d, "results4/random_examples/", 0, loader, device)


if __name__ == "__main__":
    run_test()