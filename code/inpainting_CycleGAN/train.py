import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

from generator import Generator
from discriminator import Discriminator
from dataset import SatelliteDataset
from utils import save_example, save_models, plot_losses
import config

LEARNING_RATE = 0.0002
using_mse = False
if using_mse:
    adv_loss = nn.MSELoss()
else:
    adv_loss = nn.BCEWithLogitsLoss()
pixel_loss = nn.L1Loss()
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
# These are random numbers
LAMBDA_ID = .3
LAMBDA_CYCLE = .33
LAMBDA_ANTI_LOCAL = .3
LAMBDA_GAN = 0.01

start = config.start
size = config.size

scaler = torch.cuda.amp.GradScaler()

class Train:

    def __init__(self):
        self.generator = Generator(True).to(device)
        self.discriminator = Discriminator().to(device)

        self.disc_opt = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE)
        self.gen_opt = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)

        ds = SatelliteDataset("../../data/train", 10)
        self.loader = DataLoader(ds, batch_size=2, num_workers=1)
        val_ds = SatelliteDataset("../../data/val")
        self.val_loader = DataLoader(val_ds, 1)
        torch.cuda.empty_cache()
        self.losses = {}
        self.losses["d_loss"] = []
        self.losses["g_loss"] = []
        self.losses["cycle_loss"] = []
        self.losses["id_loss"] = []
        self.current_epoch = 0



    def do_epoch(self):
        # torch.cuda.empty_cache()
        self.disc_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.generator.train()
        self.discriminator.train()
        total_g_loss = 0
        total_d_loss = 0
        total_cycle_loss = 0
        total_id_loss = 0
        total = 0
        loop = tqdm(self.loader)
        for img, classes, mod_classes, (w_s, h_s) in loop:
            img, classes, mod_classes = img.to(device), classes.to(device), mod_classes.to(device)

            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            with torch.cuda.amp.autocast():


                fake_img = self.generator(img, mod_classes)
                if torch.isnan(fake_img).any():
                    print("generated image contains nan")
                d_real = self.discriminator(img)
                if using_mse:
                    d_real = torch.sigmoid(d_real)
                d_fake = self.discriminator(fake_img)
                if using_mse:
                    d_fake = torch.sigmoid(d_fake)
                
                d_real_loss = adv_loss(d_real, torch.ones_like(d_real))
                d_fake_loss = adv_loss(d_fake, torch.zeros_like(d_fake))

                d_loss = (d_fake_loss + d_fake_loss) / 2



               # fake_img = self.generator(img, mod_classes)
                cycle_img = self.generator(fake_img, classes)
                id_img = self.generator(img, classes)

                # d_fake = self.discriminator(fake_img)
                # if using_mse:
                #     d_fake = torch.sigmoid(d_fake)

                g_gan_loss = adv_loss(d_fake, torch.ones_like(d_fake))

                g_cycle_loss = pixel_loss(cycle_img, img)
                g_id_loss = pixel_loss(id_img, img)

          
                local_img = torch.zeros((img.shape[0], 3, size, size))
                local_fake = torch.zeros((img.shape[0], 3, size, size))
                
                w_s = np.array(w_s)
                h_s = np.array(h_s)
                w_slice = slice(list(w_s), list(w_s+size))
                h_slice = slice(list(h_s), list(h_s + size))
                print(img.shape[0])
                print(slice(img.shape[0]))
                print(slice(3))
                print("w slice", w_slice)
                print("h slice", h_slice)
                batch_slice = slice(0,img.shape[0])
                print("batch slice", batch_slice)
                c = slice(0,3)
                print(c)
                local_fake = fake_img[batch_slice, c, w_slice, h_slice]
                print("local fake shape", local_fake.shape)
                for _idx in range(img.shape[0]):
                    w_s_i = w_s[_idx].item()
                    h_s_i = h_s[_idx].item()
                    for i in range(w_s_i, w_s_i + size):
                        for j in range(h_s_i, h_s_i + size):
                            local_img[_idx,:,i - w_s_i, j - h_s_i] = img[_idx, :, i, j]
                            local_fake[_idx,:, i - w_s_i, j - h_s_i] = fake_img[_idx, :, i, j]


                anti_local_loss = 1 - pixel_loss(local_fake, local_img)

                g_loss = (
                    (g_gan_loss * LAMBDA_GAN) 
                        + (LAMBDA_CYCLE * g_cycle_loss) 
                        + (LAMBDA_ID * g_id_loss)
                        + (LAMBDA_ANTI_LOCAL * anti_local_loss)
                )
            

            # Update models
            scaler.scale(d_loss).backward(retain_graph=True)
            scaler.scale(g_loss).backward()
            scaler.step(self.disc_opt)
            scaler.step(self.gen_opt)
            scaler.update()
            total_d_loss += d_loss.item()
            

            total_g_loss += g_loss.item()
            total_cycle_loss += g_cycle_loss.item()
            total_id_loss += g_id_loss.item()
               

            total += 1
            

        self.losses["g_loss"].append(total_g_loss)
        self.losses["d_loss"].append(total_d_loss)
        self.losses["cycle_loss"].append(total_cycle_loss)
        self.losses["id_loss"].append(total_id_loss)


    def train(self, num_epochs, start_epoch=0):
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            print(f"\nepoch {epoch + 1} / {num_epochs}")
            self.do_epoch()
            save_example(self.generator, self.val_loader, epoch+1, "eval", device)
            plot_losses(self.losses, "losses")
            save_models(self.generator, self.discriminator, epoch+1)


    def load_models(self, epoch_num):
        self.generator.load_state_dict(torch.load(f"models/generator{epoch_num}.pt"))
        self.discriminator.load_state_dict(torch.load(f"models/discriminator{epoch_num}.pt"))




def run():
    t = Train()
    t.train(3)

if __name__ == "__main__":
    run()