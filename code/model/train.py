import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json

from generator import Generator
from discriminator import Discriminator
from dataset import SatelliteDataset
from utils import save_example
import config


device = "cuda"
LEARNING_RATE = 0.0002
scaler = torch.cuda.amp.GradScaler()

pixel_loss_fn = nn.L1Loss()
adv_loss_fn = nn.MSELoss()
style_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()

STYLE_LAMBDA = 0.25
ADV_LAMBDA = 0.10
PIXEL_LAMBDA = 0.50
ID_LAMBDA = 0.15
LOCAL_LAMBDA = 0.25
G_LC_LAMBDA = 0.10

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../../data")
parser.add_argument("--load_models", type=bool, default=False)
parser.add_argument("--eval_dir", type=str, default="eval")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--losses_dir", type=str, default="losses")
parser.add_argument("--models_dir", type=str, default="models/")
parser.add_argument("--use_sigmoid", type=bool, default=False)

args = parser.parse_args()
losses_names =  ["d_fake_loss","d_real_loss","d_lc_real_loss", "d_loss", "g_loss","g_adv_loss","g_pixel_loss","g_pixel_id_loss","g_style_id_loss", "g_gen_lc_loss", "local_loss"]

def style_loss_fn2(phi1, phi2, vgg_activation):

    if len(phi1.shape) < 4:
        phi1 = phi1.reshape(1, phi1.shape[0], phi1.shape[1], phi1.shape[2]) 
        phi2 = phi2.reshape(1, phi2.shape[0], phi2.shape[1], phi2.shape[2]) 

    phi1 = vgg_activation(phi1)
    phi2 = vgg_activation(phi2)

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
        print("arguments", args)

        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        print(self.generator)
        print(self.discriminator)
        # betas? 0.5 and 0.999 are used in pix2pix
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
        if args.load_models:
            self.load_models()

        self.losses = {}
        for name in losses_names:
            self.losses[name] = []

        if not os.path.exists(args.losses_dir):
            os.mkdir(args.losses_dir)
        
        self.losses_file = os.path.join(args.losses_dir, "losses.json")
        if os.path.exists(self.losses_file) and args.load_models:
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

        for rgb_a, rgb_b, lc_a, lc_b, binary_mask, lc_ab, masked_areas in loop:
            rgb_a, rgb_b, lc_a, lc_b, binary_mask = rgb_a.to(device), rgb_b.to(device), lc_a.to(device), lc_b.to(device), binary_mask.to(device)
            lc_ab = lc_ab.to(device)

            ## DISCRIMINATOR TRAIN

            self.disc_opt.zero_grad()
        
            with torch.cuda.amp.autocast():
                fake_img = self.generator(rgb_a, lc_ab, binary_mask)

                fake_gen_lc, d_fake = self.discriminator(fake_img)
                # use sigmoid ?
                # maybe use random image as rgb_a
                gen_lc_a, d_real = self.discriminator(rgb_a)

                if args.use_sigmoid:
                    d_fake = torch.sigmoid(d_fake)
                    d_real = torch.sigmoid(d_real)

                # is this the correct use of adv loss? PatchGAN https://github.com/znxlwm/pytorch-pix2pix/blob/3059f2af53324e77089bbcfc31279f01a38c40b8/pytorch_pix2pix.py 
                # uses BCE_Loss
                # LSGAN https://arxiv.org/abs/1611.04076 uses mse but no patches
                #
                # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
                # In this they say they use PatchGAN, they do it like this and dont use sigmoid in discriminator
                # and either use MSE or BCEWithLogits, and they do wgangp: see line 270 in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
                d_fake_loss = adv_loss_fn(d_fake, torch.zeros_like(d_fake))
                d_real_loss = adv_loss_fn(d_real, torch.ones_like(d_real))

                # we want the fake one to still have correct classes?
                # I think using the fake to
                # d_lc_fake_loss = class_loss_fn(fake_gen_lc, torch.argmax(lc_ab, 1))
                d_lc_real_loss = class_loss_fn(gen_lc_a, torch.argmax(lc_a, 1))

                d_lc_loss =  d_lc_real_loss # (d_lc_fake_loss + d_lc_real_loss) / 2
                d_adv_loss = (d_fake_loss + d_real_loss) / 2

                d_loss = (d_lc_loss + d_adv_loss)
            
            
            scaler.scale(d_loss).backward()
            scaler.step(self.disc_opt)
            scaler.update()

            ## GENERATOR TRAIN

            self.gen_opt.zero_grad()
            with torch.cuda.amp.autocast():
                fake_img = self.generator(rgb_a, lc_ab, binary_mask)
                lc_gen_fake, d_fake = self.discriminator(fake_img)
                if args.use_sigmoid:
                    torch.sigmoid(d_fake)

                g_adv_loss = adv_loss_fn(d_fake, torch.ones_like(d_fake))

                # use how bad the discriminator is at generating the fake lc_ab as a generator loss
                # btw does this work as I expect?
                # https://discuss.pytorch.org/t/optimizing-based-on-another-models-output/6935
                # Because fake_img, from self.generator is part of the computational graph of g_adv_loss, this does work in training the generator.
                g_gen_lc_loss = class_loss_fn(lc_gen_fake, torch.argmax(lc_ab, 1))
                

                id_img = self.generator(rgb_a, lc_a, torch.zeros_like(binary_mask).to(device))
                g_pixel_id_loss = pixel_loss_fn(id_img, rgb_a)
                feature_id_img = self.relu3_3(id_img)
                feature_rgb_a = self.relu3_3(rgb_a)
                g_style_id_loss = style_loss_fn(feature_id_img, feature_rgb_a)

                local_loss = 0

                # set changed places to 0 
                fake_img_unchanged_area = fake_img.clone()
                rgb_a_unchanged_area = rgb_a.clone()

                # look individually at changed area and do a pixel loss
                # Not sure what type of loss is best,
                # Maybe using some kind of local discriminator would be better, like in local and global inpainting paper
                for j in range(args.batch_size):
                    for i in range(len(masked_areas)):

                        r_w = masked_areas[i][0][j]
                        r_h = masked_areas[i][1][j]
                        mask_size_w = masked_areas[i][2][j]
                        mask_size_h = masked_areas[i][3][j]
                        
                        #r_w, r_h, mask_size_w, mask_size_h = masked_area
                        # maybe not use the rgb_a is the classes are the same.
                        local_gen_area = fake_img[j,:,r_w-config.local_area_margin:r_w + mask_size_w+config.local_area_margin, r_h-config.local_area_margin:r_h+mask_size_h+config.local_area_margin]
                        rgb_b_local_area = rgb_b[j,:,r_w-config.local_area_margin:r_w + mask_size_w+config.local_area_margin, r_h-config.local_area_margin:r_h+mask_size_h+config.local_area_margin]
                        feature_local_gen = self.relu3_3(local_gen_area.reshape(1, local_gen_area.shape[0], local_gen_area.shape[1], local_gen_area.shape[2]))
                        feature_local_rgb_b = self.relu3_3(rgb_b_local_area.reshape(1, rgb_b_local_area.shape[0], rgb_b_local_area.shape[1], rgb_b_local_area.shape[2]))
                        local_loss += style_loss_fn(feature_local_gen, feature_local_rgb_b)

                        fake_img_unchanged_area[j, :, r_w:r_w + mask_size_w, r_h:r_h + mask_size_h] = torch.zeros(3, mask_size_w, mask_size_h)
                        rgb_a_unchanged_area[j, :, r_w:r_w + mask_size_w, r_h:r_h + mask_size_h] = torch.zeros(3, mask_size_w, mask_size_h)


                g_pixel_loss = pixel_loss_fn(fake_img_unchanged_area, rgb_a_unchanged_area)



                local_loss = local_loss / args.batch_size

                g_loss = (
                    (g_adv_loss * ADV_LAMBDA)
                  #  + (g_style_loss * STYLE_LAMBDA)
                    + (g_gen_lc_loss * G_LC_LAMBDA)
                    + (g_pixel_loss * PIXEL_LAMBDA)
                    + (
                        (g_pixel_id_loss + g_style_id_loss) * ID_LAMBDA
                    )
                    + (local_loss * LOCAL_LAMBDA)
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

        
        eval_dir = args.eval_dir
        save_example(self.generator, self.discriminator, eval_dir, 0, self.val_loader, device)
        if os.path.exists(eval_dir) and args.load_models:
            start_epoch = len(os.listdir(eval_dir))
        else:
            start_epoch = 1
        num_epochs = args.num_epochs
        
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