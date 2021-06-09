import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json

from generator import Generator
from ac_discriminator import Discriminator
from dataset import SatelliteDataset
from utils import save_example


device = "cuda" 
LEARNING_RATE = 0.0002
scaler = torch.cuda.amp.GradScaler()

pixel_loss_fn = nn.L1Loss()
adv_loss_fn = nn.MSELoss()
# style_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()

STYLE_LAMBDA = 0.45
ADV_LAMBDA = 0.05
PIXEL_LAMBDA = 0.5
ID_LAMBDA = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../../data")
parser.add_argument("--load_models", type=bool, default=False)
parser.add_argument("--eval_dir", type=str, default="eval")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--losses_dir", type=str, default="losses")
parser.add_argument("--models_dir", type=str, default="models/")

args = parser.parse_args()
losses_names =  ["d_fake_loss","d_real_loss","d_lc_fake_loss","d_lc_real_loss","g_adv_loss","g_pixel_loss","g_style_loss","g_pixel_id_loss","g_style_id_loss"]

def style_loss_fn(phi1, phi2):
    batch_size, c, h, w = phi1.shape
    psi1 = phi1.reshape((batch_size, c, w*h))
    psi2 = phi2.reshape((batch_size, c, w*h))
    
    gram1 = torch.matmul(psi1, torch.transpose(psi1, 1, 2)) / (c*h*w)
    gram2 = torch.matmul(psi2, torch.transpose(psi2, 1, 2)) / (c*h*w)
    # as described in johnson et al.
    return torch.sum(torch.norm(gram1 - gram2, p = "fro", dim=(1,2))) / batch_size
class Train:

    def __init__(self):
        data_dir = args.data_dir


        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        # betas?
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

        d = SatelliteDataset(os.path.join(data_dir,"train"))
        self.loader = DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        d_val = SatelliteDataset(os.path.join(data_dir,"val"))
        print(f"{len(os.listdir(os.path.join(data_dir,'train/rgb')))} files in train/rgb")
        print(f"{len(os.listdir(os.path.join(data_dir,'train/lc_classes')))} files in train/lc_classes")
        self.val_loader = DataLoader(d_val, 1)

        vgg_model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True).to(device)
        self.relu3_3 = torch.nn.Sequential(*vgg_model.features[:16])

        self.models_dir = args.models_dir
        self.loop_description = ""
        print(args.load_models)
        if args.load_models:
            self.load_models()

        self.losses = {}
        for name in losses_names:
            self.losses[name] = []

        if not os.path.exists(args.losses_dir):
            os.mkdir(args.losses_dir)
        
        self.losses_file = os.path.join(args.losses_dir,"losses.json")
        if os.path.exists(self.losses_file):
            with open(self.losses_file, "r") as f:
                self.losses = json.load(f)

    def epoch(self):

        loop = tqdm(self.loader, position=0)
        loop.set_description(self.loop_description)
        self.discriminator.train()
        self.generator.train()

        epoch_losses = {}
        for name in losses_names:
            epoch_losses[name] = 0
        total = 0

        for rgb_a, lc_a, lc_ab, rgb_ab in loop:
            rgb_a, lc_a, lc_ab, rgb_ab = rgb_a.to(device), lc_a.to(device), lc_ab.to(device), rgb_ab.to(device)

            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()

            with torch.cuda.amp.autocast():
                fake_img = self.generator(rgb_a, lc_ab)

                fake_gen_lc, d_fake = self.discriminator(fake_img, lc_ab)
                # maybe use random image as rgb_a
                gen_lc_a, d_real = self.discriminator(rgb_a, lc_a)

                d_fake_loss = adv_loss_fn(d_fake, torch.zeros_like(d_fake))
                d_real_loss = adv_loss_fn(d_real, torch.ones_like(d_real))

                # we want the fake one to still have correct classes?
                d_lc_fake_loss = class_loss_fn(fake_gen_lc, torch.argmax(lc_ab, 1))
                d_lc_real_loss = class_loss_fn(gen_lc_a, torch.argmax(lc_a, 1))

                d_lc_loss = (d_lc_fake_loss + d_lc_real_loss) / 2
                d_adv_loss = (d_fake_loss + d_real_loss) / 2

                d_loss = (d_lc_loss + d_adv_loss)
            
            
            scaler.scale(d_loss).backward()
            scaler.step(self.disc_opt)
            scaler.update()

            with torch.cuda.amp.autocast():
                fake_img = self.generator(rgb_a, lc_ab)
                _, d_fake = self.discriminator(fake_img, lc_ab)

                g_adv_loss = adv_loss_fn(d_fake, torch.ones_like(d_fake))
                g_pixel_loss = pixel_loss_fn(fake_img, rgb_ab)

                g_style_loss = style_loss_fn(fake_img, rgb_ab)

                id_img = self.generator(rgb_a, lc_a)
                g_pixel_id_loss = pixel_loss_fn(id_img, rgb_a)
                g_style_id_loss = style_loss_fn(id_img, rgb_a)

                g_loss = (
                    (g_adv_loss * ADV_LAMBDA)
                    + (g_style_loss * STYLE_LAMBDA)
                    + (g_pixel_loss * PIXEL_LAMBDA)
                    + (
                        (g_pixel_id_loss + g_style_id_loss) * ID_LAMBDA
                    )
                )

            scaler.scale(g_loss).backward()
            scaler.step(self.gen_opt)
            scaler.update()

            total += 1
            for name in losses_names:
                epoch_losses[name] += eval(name).item()
        
        for name in losses_names:
            self.losses[name].append(epoch_losses[name] / total)


    def save_models(self):
        
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        torch.save(self.generator.state_dict(), f"{self.models_dir}/generator.pt")
        torch.save(self.discriminator.state_dict(), f"{self.models_dir}/discriminator.pt")

    def load_models(self):

        self.generator.load_state_dict(torch.load(f"{self.models_dir}/generator.pt"))
        self.discriminator.load_state_dict(torch.load(f"{self.models_dir}/discriminator.pt"))

    def save_losses(self):
        with open(self.losses_file, "w") as f:
            json.dump(self.losses, f)


    def train(self):

        start_epoch = 1
        eval_dir = args.eval_dir
        if os.path.exists(eval_dir):
            start_epoch = len(os.listdir(eval_dir)) + 1
        num_epochs = args.num_epochs
        print("arguments", args)

        save_example(self.generator, self.discriminator, eval_dir, 0, self.val_loader, device)
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.loop_description = f"{epoch} / {num_epochs + start_epoch - 1}"
            self.epoch()
            save_example(self.generator, self.discriminator, eval_dir, epoch, self.val_loader, device)
            self.save_models()
            self.save_losses()


def run():
    t = Train()
    
    t.train()

if __name__ == "__main__":
    run()