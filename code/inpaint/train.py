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
from deterministic_dataset import DeterministicSatelliteDataset
from utils import save_example, calc_all_IoUs
import config
from plot_losses import plot_losses, plot_ious

device = "cuda"
LEARNING_RATE = 0.0002
scaler = torch.cuda.amp.GradScaler()

pixel_loss_fn = nn.L1Loss()
adv_loss_fn = nn.MSELoss()
style_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()

STYLE_LAMBDA = 0.0
ADV_LAMBDA = 0.10
PIXEL_LAMBDA = 0.50
ID_LAMBDA = 0.15
LOCAL_STYLE_LAMBDA = 0.25
LOCAL_PIXEL_LAMBDA = 0.25
G_LC_LAMBDA = 0.3


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../../data/grid_dir")
parser.add_argument("--load_models", type=bool, default=False)
parser.add_argument("--eval_dir", type=str, default="eval")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--losses_dir", type=str, default="losses")
parser.add_argument("--models_dir", type=str, default="models/")
parser.add_argument("--use_sigmoid", type=bool, default=False)
parser.add_argument("--log_file", type=str, default="training_log.txt")

args = parser.parse_args()
losses_names =  ["d_fake_loss","d_real_loss","d_lc_real_loss", "d_loss", "g_loss","g_adv_loss","g_pixel_loss", "g_pixel_id_loss","g_style_id_loss", "g_gen_lc_loss", "local_style_loss", "local_pixel_loss"]
possible_ious = ["iou_gen_lc_fake_a_vs_gen_lc_a","iou_gen_lc_a_vs_lc_a","iou_gen_lc_fake_a_vs_lc_a"]

log_file = args.log_file

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

        d = SatelliteDataset(os.path.join(data_dir,"train"), 10)
        self.loader = DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

        d_val = SatelliteDataset(os.path.join(data_dir,"val"), 10)
        self.val_loader = DataLoader(d_val, 1, num_workers=args.num_workers)
       
        # det_d_val = DeterministicSatelliteDataset(os.path.join(data_dir,"val"))
        # self.det_val_loader = DataLoader(det_d_val, 1)

        print(f"{len(os.listdir(os.path.join(data_dir,'train/rgb')))} files in train/rgb")
        print(f"{len(os.listdir(os.path.join(data_dir,'train/lc_classes')))} files in train/lc_classes")

        vgg_model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True).to(device)
        self.relu3_3 = torch.nn.Sequential(*vgg_model.features[:16])

        
        self.models_dir = args.models_dir
        self.loop_description = ""
        if args.load_models:
            self.load_models()

        self.losses = {}
        self.val_losses = {}
        for name in losses_names:
            self.losses[name] = []
            self.val_losses[name] = []

        if not os.path.exists(args.losses_dir):
            os.mkdir(args.losses_dir)

        
        self.losses_file = os.path.join(args.losses_dir, "train_losses.json")
        self.val_losses_file = os.path.join(args.losses_dir, "val_losses.json")
        if os.path.exists(self.losses_file) and args.load_models:
            with open(self.losses_file, "r") as f:
                self.losses = json.load(f)

        if os.path.exists(self.val_losses_file) and args.load_models:
            with open(self.val_losses_file, "r") as f:
                self.val_losses = json.load(f)

        self.ious = {}
        for iou in possible_ious:
            self.ious[iou] = []

        self.iou_file = os.path.join(args.losses_dir, "iou.json")
        if os.path.exists(self.iou_file) and args.load_models:
            with open(self.iou_file, "r") as f:
                self.ious = json.load(f) 

        print("\n########\nLambdas:")
        print("STYLE_LAMBDA",STYLE_LAMBDA)
        print("ADV_LAMBDA",ADV_LAMBDA)
        print("PIXEL_LAMBDA", PIXEL_LAMBDA)
        print("ID_LAMBDA", ID_LAMBDA)
        print("LOCAL_STYLE_LAMBDA",LOCAL_STYLE_LAMBDA)
        print("LOCAL_PIXEL_LAMBDA", LOCAL_PIXEL_LAMBDA)
        print("G_LC_LAMBDA", G_LC_LAMBDA)
        print("########\n")

        log_s = f"""
########\nLambdas:
STYLE_LAMBDA: {STYLE_LAMBDA}
ADV_LAMBDA: {ADV_LAMBDA}
PIXEL_LAMBDA: {PIXEL_LAMBDA}
ID_LAMBDA: {ID_LAMBDA}
LOCAL_STYLE_LAMBDA: {LOCAL_STYLE_LAMBDA}
LOCAL_PIXEL_LAMBDA: {LOCAL_PIXEL_LAMBDA}
G_LC_LAMBDA: {G_LC_LAMBDA}
########\n
        """

        with open(log_file, "a") as f:
            f.write(log_s)
        

    def epoch(self, evaluation=False):

        if evaluation:
            loop = tqdm(self.val_loader, position=0)
            loop.set_description(f"{self.loop_description} evaluation")
            self.generator.eval()
            self.discriminator.eval()
            for m in self.generator.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        else:
            loop = tqdm(self.loader, position=0)
            loop.set_description(self.loop_description)

            self.discriminator.train()
            self.generator.train()

        epoch_losses = {}
        for name in losses_names:
            epoch_losses[name] = 0

        if evaluation:
            epoch_ious = {}
            for iou in possible_ious:
                epoch_ious[iou] = 0

        total = 0

        for rgb_a, lc_a, rgb_a_masked, masked_areas in loop:
            rgb_a, lc_a, rgb_a_masked = rgb_a.to(device), lc_a.to(device), rgb_a_masked.to(device)


            ## DISCRIMINATOR TRAIN

            self.disc_opt.zero_grad()
        
            with torch.cuda.amp.autocast():
                fake_img = self.generator(rgb_a_masked, lc_a)

                _, d_fake = self.discriminator(fake_img)
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

               
                d_lc_real_loss = class_loss_fn(gen_lc_a, torch.argmax(lc_a, 1))
                d_loss = (d_fake_loss + d_real_loss + d_lc_real_loss) / 2
            
            if not evaluation:
                scaler.scale(d_loss).backward()
                scaler.step(self.disc_opt)
                scaler.update()

            ## GENERATOR TRAIN

            self.gen_opt.zero_grad()
            with torch.cuda.amp.autocast():
                fake_img = self.generator(rgb_a_masked, lc_a)

                lc_gen_fake, d_fake = self.discriminator(fake_img)
                if args.use_sigmoid:
                    torch.sigmoid(d_fake)

                if ADV_LAMBDA == 0:
                    g_adv_loss = torch.tensor(0)
                else:
                    g_adv_loss = adv_loss_fn(d_fake, torch.ones_like(d_fake))

                if G_LC_LAMBDA == 0:
                    g_gen_lc_loss = torch.tensor(0)
                else:
                    g_gen_lc_loss = class_loss_fn(lc_gen_fake, torch.argmax(lc_a, 1))
                
                if ID_LAMBDA == 0:
                    g_pixel_id_loss = torch.tensor(0)
                    g_style_id_loss = torch.tensor(0)
                else:
                    id_img = self.generator(rgb_a, lc_a)
                    g_pixel_id_loss = pixel_loss_fn(id_img, rgb_a)
                    feature_id_img = self.relu3_3(id_img)
                    feature_rgb_a = self.relu3_3(rgb_a)
                    g_style_id_loss = style_loss_fn(feature_id_img, feature_rgb_a)

                local_style_loss = 0
                local_pixel_loss = 0

                # set changed places to 0 
                fake_img_unchanged_area = fake_img.clone()
                rgb_a_unchanged_area = rgb_a.clone()

                # look individually at changed area and do a pixel loss
                # Not sure what type of loss is best,
                # Maybe using some kind of local discriminator would be better, like in local and global inpainting paper
                for j in range(len(masked_areas[0][0])):
                    for i in range(len(masked_areas)):

                        r_w = masked_areas[i][0][j]
                        r_h = masked_areas[i][1][j]
                        mask_size_w = masked_areas[i][2][j]
                        mask_size_h = masked_areas[i][3][j]
                        
                        #r_w, r_h, mask_size_w, mask_size_h = masked_area
                        # maybe not use the rgb_a is the classes are the same.
                        # the boarder margin is only constrained by the adversarial loss...
                        local_gen_area = fake_img[j,:,r_w:r_w + mask_size_w, r_h:r_h+mask_size_h]
                        rgb_a_local_area = rgb_a[j,:,r_w:r_w + mask_size_w, r_h:r_h+mask_size_h]
                        feature_local_gen = self.relu3_3(local_gen_area.reshape(1, local_gen_area.shape[0], local_gen_area.shape[1], local_gen_area.shape[2]))
                        feature_local_rgb_a = self.relu3_3(rgb_a_local_area.reshape(1, rgb_a_local_area.shape[0], rgb_a_local_area.shape[1], rgb_a_local_area.shape[2]))
                        
                        if LOCAL_STYLE_LAMBDA != 0:
                            local_style_loss += style_loss_fn(feature_local_gen, feature_local_rgb_a)
                        
                        if LOCAL_PIXEL_LAMBDA != 0:
                            local_pixel_loss += pixel_loss_fn(local_gen_area, rgb_a_local_area)


                if local_style_loss == 0:
                    local_style_loss = torch.tensor(0)
                if local_pixel_loss == 0:
                    local_pixel_loss = torch.tensor(0)

                # Don't calculate if lambda is 0, since
                if PIXEL_LAMBDA == 0:
                    g_pixel_loss = torch.tensor(0)
                else:
                    g_pixel_loss = pixel_loss_fn(fake_img, rgb_a)
                

                local_pixel_loss = local_pixel_loss / (len(masked_areas[0][0]) * config.num_inpaints)
                
                local_style_loss = local_style_loss / (len(masked_areas[0][0]) * config.num_inpaints)

                g_loss = (
                    (g_adv_loss * ADV_LAMBDA)
                    + (g_gen_lc_loss * G_LC_LAMBDA)
                    + (g_pixel_loss * PIXEL_LAMBDA)
                    + (
                        (g_pixel_id_loss + g_style_id_loss) * ID_LAMBDA
                    )
                    + (local_style_loss * LOCAL_STYLE_LAMBDA)
                    + (local_pixel_loss * LOCAL_PIXEL_LAMBDA)
                )

                if evaluation:
                    fake_a = self.generator(rgb_a, lc_a)
                    gen_lc_fake_a, _ = self.discriminator(fake_a)
                    gen_lc_a, _ = self.discriminator(rgb_a) 

                    iou_gen_lc_fake_a_vs_gen_lc_a = calc_all_IoUs(gen_lc_fake_a, gen_lc_a)
                    iou_gen_lc_a_vs_lc_a = calc_all_IoUs(gen_lc_a, lc_a)
                    iou_gen_lc_fake_a_vs_lc_a = calc_all_IoUs(gen_lc_fake_a, lc_a)
            
            if not evaluation:
                scaler.scale(g_loss).backward()
                scaler.step(self.gen_opt)
                scaler.update()

            total += 1
            for name in losses_names:
                epoch_losses[name] += eval(name).item()

            if evaluation:
                for iou in possible_ious:
                    epoch_ious[iou] += eval(iou)
        
        loss_dict = self.val_losses if evaluation else self.losses

        for name in losses_names:
            loss_dict[name].append(epoch_losses[name] / total)

        if evaluation:
            for iou in possible_ious:
                self.ious[iou].append(epoch_ious[iou] / total)
            with open(self.iou_file, "w") as f:
                json.dump(self.ious, f)

            plot_ious(self.ious, args.losses_dir)

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

        with open(self.val_losses_file, "w") as f:
            json.dump(self.val_losses, f)

    def train(self):
        eval_dir = args.eval_dir
        num_save_examples = 1
        # save deterministic samples
        save_example(self.generator, self.discriminator, eval_dir, 0, self.val_loader, device, num_save_examples)
        if os.path.exists(eval_dir) and args.load_models:
            start_epoch = int(len(os.listdir(eval_dir)) / num_save_examples)
        else:
            start_epoch = 1
        num_epochs = args.num_epochs
        with open(log_file, "a") as f:
            f.write(f"start epoch {start_epoch}\n\n\n")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.loop_description = f"{epoch} / {num_epochs + start_epoch - 1}"
            self.epoch(False)
            save_example(self.generator, self.discriminator, eval_dir, epoch, self.val_loader, device, num_save_examples)
            self.epoch(evaluation=True)
            plot_losses(args.losses_dir, self.losses, self.val_losses)

            self.save_models()
            self.save_losses()


def run():
    t = Train()
    t.train()

if __name__ == "__main__":
    run()