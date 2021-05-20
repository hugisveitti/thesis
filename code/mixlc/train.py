from generator import Generator
from discriminator import Discriminator
from dataset import SatelliteDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import save_example, save_models, plot_losses
import config


LEARNING_RATE = 1e-3
BATCH_SIZE = 1
ALPHA = 0.5
PIX_LAMBDA = 10
LOCAL_PIX_LAMBDA = 5

use_ba = False
use_mse = True

start = config.start
size = config.size
bz = config.buffer_zone
scaler = torch.cuda.amp.GradScaler()
if use_mse:
    adv_loss = nn.MSELoss() 
else:
    adv_loss = nn.BCEWithLogitsLoss()
pixel_loss = nn.L1Loss()
device = "cuda" if torch.cuda.is_available() else "cpu"


class Train:

    def __init__(self):
        torch_type = config.torch_type
        self.generator = Generator().type(torch_type).to(device)
        self.discriminator = Discriminator().type(torch_type).to(device)
        d = SatelliteDataset("../../data/train")
        self.loader = DataLoader(d, BATCH_SIZE, num_workers=1, pin_memory=True)
        self.dict_opt = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE)
        d_val = SatelliteDataset("../../data/val")
        self.val_loader = DataLoader(d_val, 1)
        self.losses = {}
        self.losses["d_loss"] = []
        self.losses["g_adv_loss"] = []
        self.losses["g_pix_loss"] = [] 
        self.losses["g_loss"] = []
        self.losses["g_local_pix_loss"] = []

        torch.cuda.empty_cache()


    def epoch(self):

        loop = tqdm(self.loader, position=0)
        total_d_loss = 0
        total_g_adv_loss = 0
        total_g_pix_loss = 0
        total_g_loss = 0
        total_g_local_pix_loss = 0
        total = 0
        cpu = "cpu"
        for rgb_a, rgb_b, lc_ab, rgb_ab in loop:
            rgb_a = rgb_a.to(device)
            if use_ba:
                rgb_b = rgb_b.to(device)
            lc_ab = lc_ab.to(device)
            rgb_ab = rgb_ab.to(device)
            self.dict_opt.zero_grad()
            self.gen_opt.zero_grad()


            # having some casting problems, cant use DoubleTensor since that is too much memory but got many nan-s when using only HalfTensor
           # with torch.cuda.amp.autocast():

            rgb_ab_fake = self.generator(rgb_a, lc_ab)
            if use_ba:
                rgb_ba_fake = self.generator(rgb_b, lc_ab)
            d_fake_ab = self.discriminator(rgb_ab_fake)
            if use_ba:
                d_fake_ba = self.discriminator(rgb_ba_fake)
            d_real_a = self.discriminator(rgb_a)
            if use_ba:
                d_real_b = self.discriminator(rgb_b)


            # Discriminator_loss
            
            d_fake_ab_loss = adv_loss(d_fake_ab, torch.zeros_like(d_fake_ab))
            d_real_a_loss = adv_loss(d_real_a, torch.ones_like(d_real_a))
            if use_mse:
                d_fake_ab_loss = torch.sigmoid(d_fake_ab_loss)
                d_real_a_loss = torch.sigmoid(d_real_a_loss)

            if use_ba:
                d_fake_ba_loss = adv_loss(d_fake_ba, torch.zeros_like(d_fake_ba))
                d_real_b_loss = adv_loss(d_real_b, torch.ones_like(d_real_b))
                if use_mse:
                    d_fake_ba_loss = torch.sigmoid(d_fake_ba_loss)
                    d_real_b_loss = torch.sigmoid(d_real_b_loss)

            if use_ba:
                d_loss = (d_fake_ab_loss + d_fake_ba_loss + d_real_a_loss + d_real_b_loss) / 4
            else:
                d_loss = (d_fake_ab_loss + d_real_a_loss) / 2

            scaler.scale(d_loss).backward()
            scaler.step(self.dict_opt)
            scaler.update()


            
            # Generator loss


            rgb_ab_fake = self.generator(rgb_a, lc_ab)
            if use_ba:
                rgb_ba_fake = self.generator(rgb_b, lc_ab)

            d_fake_ab = self.discriminator(rgb_ab_fake)
            d_fake_ab_loss = adv_loss(d_fake_ab, torch.zeros_like(d_fake_ab))
            g_adv_ab_loss = adv_loss(d_fake_ab_loss, torch.ones_like(d_fake_ab_loss))
            if use_mse:
                d_fake_ab_loss = torch.sigmoid(d_fake_ab_loss)
                g_adv_ab_loss = torch.sigmoid(g_adv_ab_loss)

            if use_ba:
                d_fake_ba = self.discriminator(rgb_ba_fake)
                d_fake_ba_loss = adv_loss(d_fake_ba, torch.zeros_like(d_fake_ba))
                g_adv_ba_loss = adv_loss(d_fake_ba_loss, torch.ones_like(d_fake_ba_loss))
                if use_mse:
                    d_fake_ba_loss = torch.sigmoid(d_fake_ba_loss)
                    g_adv_ba_loss = torch.sigmoid(g_adv_ba_loss)


            g_pix_ab_loss = pixel_loss(rgb_ab_fake, rgb_ab)
            
            if use_ba:
                g_pix_ba_loss = pixel_loss(rgb_ba_fake, rgb_ab)

            local_rgb_ab_fake = rgb_ab_fake[:,:,start - bz : start + size + bz, start - bz:start+size + bz]
            if use_ba:
                local_rgb_ba_fake = rgb_ba_fake[:,:,start - bz : start + size + bz, start - bz:start+size + bz]

            local_rgb_real = rgb_ab[:,:, start - bz:start+size + bz, start - bz:start+size + bz]

            g_local_pix_loss_ab = pixel_loss(local_rgb_ab_fake, local_rgb_real)
            if use_ba:
                g_local_pix_loss_ba = pixel_loss(local_rgb_ba_fake, local_rgb_real)

            if use_ba:
                g_loss = (g_pix_ab_loss * PIX_LAMBDA) + (g_pix_ba_loss * PIX_LAMBDA) + (g_adv_ab_loss * ALPHA) + (g_adv_ba_loss * ALPHA) + (g_local_pix_loss_ab * LOCAL_PIX_LAMBDA) + (g_local_pix_loss_ba * LOCAL_PIX_LAMBDA)
            else:
                g_loss = (g_pix_ab_loss * PIX_LAMBDA) + (g_adv_ab_loss * ALPHA) + (g_local_pix_loss_ab * LOCAL_PIX_LAMBDA) 
    


            scaler.scale(g_loss).backward()
            scaler.step(self.gen_opt)
            scaler.update()

            total += 1
  
         
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

            if use_ba:
                total_g_adv_loss += g_adv_ab_loss.item() + g_adv_ba_loss.item()
            else:
                total_g_adv_loss += g_adv_ab_loss.item() 
           
            if use_ba:
                total_g_pix_loss += g_pix_ab_loss.item() + g_pix_ba_loss.item()
            else:
                total_g_pix_loss += g_pix_ab_loss.item() 
            if use_ba:
                total_g_local_pix_loss += g_local_pix_loss_ab.item() + g_local_pix_loss_ba.item()
            else:
                total_g_local_pix_loss += g_local_pix_loss_ab.item() 
        
        self.losses["g_adv_loss"].append(total_g_adv_loss / total)
        self.losses["d_loss"].append(total_d_loss / total)
        self.losses["g_pix_loss"].append(total_g_pix_loss / total)
        self.losses["g_loss"].append(total_g_loss / total)
        self.losses["g_local_pix_loss"].append(total_g_local_pix_loss / total)
       

                
    def run_epochs(self, num_epochs, start_epoch=0):
        print(f"device is {device}")
        for epoch in range(start_epoch, num_epochs + start_epoch):
            print(f"epoch {epoch}")
            self.epoch()
            save_example(self.generator, self.val_loader, epoch, "eval", device)
            plot_losses(self.losses, "losses/second")
            save_models(self.generator, self.discriminator, epoch)

    def load_models(self, epoch_num):
        self.generator.load_state_dict(torch.load(f"models/generator{epoch_num}.pt"))
        self.discriminator.load_state_dict(torch.load(f"models/discriminator{epoch_num}.pt"))

        
def run():
    t = Train()
    t.load_models(9)
    t.run_epochs(10, 10)

if __name__ == "__main__":
    run()