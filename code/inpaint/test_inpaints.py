from generator import Generator
from discriminator import Discriminator
from dataset import SatelliteDataset
from utils import save_example
from torch.utils.data import DataLoader
import torch

device = "cpu"

g = Generator().to(device)
g.load_state_dict(torch.load("results/inpaint_run6/models/generator.pt"))


d = Discriminator().to(device)
d.load_state_dict(torch.load("results/inpaint_run6/models/discriminator.pt"))

ds = SatelliteDataset("../../data/grid_dir/val")
loader = DataLoader(ds, 1, shuffle=True)

folder = "images"

save_example(g, d, folder, 0, loader, device)

