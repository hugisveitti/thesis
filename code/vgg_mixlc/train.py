import torch
from tqdm import tqdm
import os

from generator import Generator
from discriminator import Discriminator
from dataset import SatelliteDataset
from utils import save_example

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.cuda.amp.GradScaler()

# Coarse-to-fine use both VGG and l1 loss
# Coarse-to-fine calls this reconstruction loss
pixel_loss = torch.nn.L1Loss()
adv_loss_fn = torch.nn.MSELoss()
mse_loss_fn = torch.nn.MSELoss()

LEARNING_RATE = 0.0002
LAMDBA_PIXEL = 0.25
LAMBDA_ADV = 0.001
LAMBDA_FEATURE = 0.749


class Train():

    def __init__(self):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE, )
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)

        ds = SatelliteDataset("../../data/train")
        self.loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
        vgg_model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True).to(device)

        self.relu3_3 = torch.nn.Sequential(*vgg_model.features[:16])
        # self.relu4_3 = torch.nn.Sequential(*vgg_model.features[:23])
        self.models_dir = "models/"

        val_ds = SatelliteDataset("../../data/val")
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)
        self.tqdm_description = ""

    def epoch(self):
        loop = tqdm(self.loader, position=0)
        loop.set_description(self.tqdm_description)
        self.generator.train()
        self.discriminator.train()
        for rgb_a, rgb_b, rgb_ab, lc_a, lc_b, lc_ab in loop:
            
            rgb_a, rgb_ab, lc_a, lc_ab = rgb_a.to(device), rgb_ab.to(device), lc_a.to(device), lc_ab.to(device)
            rgb_b = rgb_b.to(device)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()


            with torch.cuda.amp.autocast():
                rgb_fake = self.generator(rgb_a, lc_ab)

                d_real = self.discriminator(rgb_b)
                d_fake = self.discriminator(rgb_fake)

                d_real_loss = adv_loss_fn(d_real, torch.ones_like(d_real))
                d_fake_loss = adv_loss_fn(d_fake, torch.zeros_like(d_fake))

                d_loss = (d_fake_loss + d_real_loss) / 2

            scaler.scale(d_loss).backward()
            scaler.step(self.disc_opt)
            scaler.update()

            
            with torch.cuda.amp.autocast():
                rgb_fake = self.generator(rgb_a, lc_ab)

                d_fake = self.discriminator(rgb_fake)

                target_features = self.relu3_3(rgb_ab)
                fake_features = self.relu3_3(rgb_fake)

                feature_loss = mse_loss_fn(fake_features, target_features)

                g_adv_loss = adv_loss_fn(d_fake, torch.ones_like(d_fake))
                
                g_pix_loss = pixel_loss(rgb_fake, rgb_ab)

                g_loss = (
                 (g_adv_loss * LAMBDA_ADV) 
                 + (feature_loss * LAMBDA_FEATURE) 
                 + (g_pix_loss * LAMDBA_PIXEL)
                )




            # will the g_loss also update the discriminator, since g_loss uses adv_loss which uses d_fake

            scaler.scale(g_loss).backward()
            scaler.step(self.gen_opt)
            scaler.update()


    def save_models(self, epoch):
        
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        torch.save(self.generator.state_dict(), f"{self.models_dir}generator_{epoch}.pt")
        torch.save(self.discriminator.state_dict(), f"{self.models_dir}discriminator_{epoch}.pt")

    def load_models(self, epoch):

        self.generator.load_state_dict(torch.load(f"{self.models_dir}/generator_{epoch}.pt"))
        self.discriminator.load_state_dict(torch.load(f"{self.models_dir}/discriminator_{epoch}.pt"))



    def train(self, num_epochs, start_epoch=1):
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.tqdm_description = f"epoch {epoch} / {num_epochs + start_epoch - 1}"
            self.epoch()
            save_example(self.generator, self.val_loader, epoch, "eval", device)
            self.save_models(epoch)
            

if __name__ == "__main__":
    t = Train()
    t.train(5)