import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json
import config


from generator import Generator
from discriminator import Discriminator
from landcover_model import LandcoverModel
from dataset import SatelliteDataset
from deterministic_dataset import DeterministicSatelliteDataset
from utils import save_example, calc_all_IoUs, StyleLoss, calc_accuracy
import config
from plot_losses import plot_losses, plot_ious
from augmentations import apply_augmentations

device = "cuda"
LEARNING_RATE = 0.0002
scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser()
parser.add_argument("--load_models", dest="load_models", action="store_true")
parser.set_defaults(load_models=False)
parser.add_argument("--data_dir", type=str, default="../../data/grid_dir")
parser.add_argument("--eval_dir", type=str, default="eval")
parser.add_argument("--losses_dir", type=str, default="losses")
parser.add_argument("--models_dir", type=str, default="models/")
parser.add_argument("--log_file", type=str, default="training_log.txt")
parser.add_argument("--landcover_model_file", type=str, default="../landcover_model/models/lc_model.pt")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
# activate dynamic lambdas after certain epoch, if -1 then never
parser.add_argument("--dynamic_lambdas_epoch", type=int, default=-1)

parser.add_argument("--g_feature_lambda", type=float, default=0.)
parser.add_argument("--g_adv_lambda", type=float, default=1)
parser.add_argument("--g_pixel_lambda", type=float, default=1)
parser.add_argument("--id_lambda", type=float, default=0.15)
parser.add_argument("--local_g_style_lambda", type=float, default=1)
parser.add_argument("--local_g_feature_lambda", type=float, default=1)
parser.add_argument("--local_g_pixel_lambda", type=float, default=0.)
parser.add_argument("--g_gen_lc_lambda", type=float, default=1)
parser.add_argument("--smooth_l1_beta", type=float, default=0.1)

# Should the changed areas have the style or feature from the new area or the old area
parser.add_argument("--local_style_new", dest="local_style_new", action="store_true")
parser.add_argument("--local_style_old", dest="local_style_new", action="store_false")
parser.set_defaults(local_style_new=False)
parser.add_argument("--local_feature_new", dest="local_feature_new", action="store_true")
parser.add_argument("--local_feature_old", dest="local_feature_new", action="store_false")
parser.set_defaults(local_feature_new=False)

args = parser.parse_args()

g_feature_lambda = args.g_feature_lambda
g_adv_lambda = args.g_adv_lambda
g_pixel_lambda = args.g_pixel_lambda
id_lambda = args.id_lambda
local_g_style_lambda = args.local_g_style_lambda
local_g_pixel_lambda = args.local_g_pixel_lambda
local_g_feature_lambda = args.local_g_feature_lambda
g_gen_lc_lambda = args.g_gen_lc_lambda

min_lambda_value = 0.05

losses_names =  ["d_fake_loss","d_real_loss","d_lc_real_loss", "d_loss", "g_loss","g_adv_loss","g_pixel_loss", "g_feature_loss","g_pixel_id_loss", "g_feature_id_loss", "g_gen_lc_loss", "local_g_style_loss", "local_g_pixel_loss", "local_g_feature_loss"]
possible_ious = ["iou_gen_lc_fake_a_vs_gen_lc_a", "iou_gen_lc_fake_a_vs_lc_a", "iou_gen_lc_a_vs_lc_a"]


dynamic_lambdas_names = ["g_feature_loss_lambda", "g_adv_lambda", "g_pixel_lambda", "local_g_style_lambda", "local_g_pixel_lambda", "local_g_feature_lambda"]

# to dynamically change loss lambdas,
# Not include id or lc_gen_lambda
all_lambdas = {
    "g_feature_loss_lambda":[g_feature_lambda, "g_feature_loss"], 
    "g_adv_lambda": [g_adv_lambda, "g_adv_loss"], 
    "g_pixel_lambda":[g_pixel_lambda, "g_pixel_loss"], 
    "local_g_style_lambda":[local_g_style_lambda, "local_g_style_loss"], 
    "local_g_pixel_lambda":[local_g_pixel_lambda, "local_g_pixel_loss"], 
    "local_g_feature_lambda":[local_g_feature_lambda, "local_g_feature_loss"]
}

log_file = args.log_file


class Train:

    def __init__(self):
        data_dir = args.data_dir
        print("arguments", args)

        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        # betas? 0.5 and 0.999 are used in pix2pix
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

        d = SatelliteDataset(os.path.join(data_dir, "train"))
        self.loader = DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

        d_val = DeterministicSatelliteDataset(os.path.join(data_dir, "val"))
        self.val_loader = DataLoader(d_val, batch_size=args.val_batch_size, num_workers=args.num_workers)
       
        print(f"{len(os.listdir(os.path.join(data_dir,'train/rgb')))} files in train/rgb")
        print(f"{len(os.listdir(os.path.join(data_dir,'train/lc_sieve')))} files in train/lc_sieve")

        vgg_model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True).to(device)
        self.relu3_3 = torch.nn.Sequential(*vgg_model.features[:16])

        self.landcover_model = LandcoverModel().to(device)
        self.landcover_model.load_state_dict(torch.load(args.landcover_model_file))
        
        self.pixel_loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
        self.adv_loss_fn = nn.MSELoss()
        self.style_loss_fn = StyleLoss(self.relu3_3, device)
        # Don't use weighted class loss, since we don't have an unbalaced training set, just that the classes are unbalaced
        # but they have a similar balance in the validation set.
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.feature_loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
        
        self.models_dir = args.models_dir
        self.loop_description = "" 

        self.losses_file = os.path.join(args.losses_dir, "train_losses.json")
        self.val_losses_file = os.path.join(args.losses_dir, "val_losses.json")
        self.iou_file = os.path.join(args.losses_dir, "iou.json")

        # if load_models then the files should exist
        if args.load_models:
            self.load_models()
            with open(self.losses_file, "r") as f:
                self.losses = json.load(f)

            with open(self.val_losses_file, "r") as f:
                self.val_losses = json.load(f)

            with open(self.iou_file, "r") as f:
                self.ious = json.load(f) 
        else:
            if not os.path.exists(args.losses_dir):
                os.mkdir(args.losses_dir)

            self.losses = {}
            self.val_losses = {}
            for name in losses_names:
                self.losses[name] = []
                self.val_losses[name] = []

            self.ious = {}
            for iou in possible_ious:
                self.ious[iou] = []


        self.use_dynamic_lambdas = False

        self.add_lambdas_to_log()
        log_string = f"""
======= LAMBDAS =======
g_feature_lambda = {g_feature_lambda}
g_adv_lambda = {g_adv_lambda}
g_pixel_lambda = {g_pixel_lambda}
id_lambda = {id_lambda}
local_g_style_lambda = {local_g_style_lambda}
local_g_pixel_lambda = {local_g_pixel_lambda}
local_g_feature_lambda = {local_g_feature_lambda}
g_gen_lc_lambda = {g_gen_lc_lambda}
========================
        """
        print(log_string)
        with open(log_file, "a") as f:
            f.write(log_string)



    def add_lambdas_to_log(self):
        # These are only the possible dynamic lamdbas
        # might delete later 
        log_string = f"""
########\nLambdas:
epoch: {self.loop_description}
{str(all_lambdas)}
########\n
        """

        with open(log_file, "a") as f:
            f.write(log_string)
        

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
            self.add_lambdas_to_log()

        epoch_losses = {}
        for name in losses_names:
            epoch_losses[name] = 0

        if evaluation:
            epoch_ious = {}
            for iou in possible_ious:
                epoch_ious[iou] = 0

        total_batches = 0

        for rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab, masked_areas in loop:
            rgb_a, rgb_ab, lc_a, lc_b, binary_mask, lc_ab = rgb_a.to(device), rgb_ab.to(device), lc_a.to(device), lc_b.to(device), binary_mask.to(device), lc_ab.to(device)
            

            ## DISCRIMINATOR TRAIN

            self.disc_opt.zero_grad()

            flag_setting = torch.cuda.amp.autocast()
            if evaluation:
                flag_setting = torch.cuda.amp.autocast() and torch.no_grad()
        
            with flag_setting:
                fake_img = self.generator(rgb_a, lc_ab, binary_mask)
                fake_img_aug, _ = apply_augmentations(fake_img, None, types=['blit', 'noise'])

                _, d_patchGAN_fake = self.discriminator(fake_img_aug)

                rgb_a_aug, lc_a_aug = apply_augmentations(rgb_a, lc_a, types=['blit', 'noise'])
                gen_lc_a, d_patchGAN_real = self.discriminator(rgb_a_aug)


                # is this the correct use of adv loss? PatchGAN https://github.com/znxlwm/pytorch-pix2pix/blob/3059f2af53324e77089bbcfc31279f01a38c40b8/pytorch_pix2pix.py 
                # uses BCE_Loss
                # LSGAN https://arxiv.org/abs/1611.04076 uses mse but no patches
                # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
                # In this they say they use PatchGAN, they do it like this and dont use sigmoid in discriminator
                # and either use MSE or BCEWithLogits, and they do wgangp: see line 270 in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
                d_fake_loss = self.adv_loss_fn(d_patchGAN_fake, torch.zeros_like(d_patchGAN_fake))
                d_real_loss = self.adv_loss_fn(d_patchGAN_real, torch.ones_like(d_patchGAN_real))

               
                d_lc_real_loss = self.class_loss_fn(gen_lc_a, torch.argmax(lc_a_aug, 1))
                d_loss = (d_fake_loss + d_real_loss + d_lc_real_loss) 
            
            if not evaluation:
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # Uncomment this if you want to save the visualization of the computational graph for the discriminator #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # from torchviz import make_dot
                # if not os.path.exists("graphs"):
                #     os.mkdir("graphs")
                # for loss in losses_names:
                #     if loss[0] == "d":
                #         print(loss, eval(loss))
                #         scaler.scale(eval(loss)).backward(retain_graph=True)
                #         make_dot(eval(loss), show_attrs=True, show_saved=True).render(f"graphs/{loss}_graph") #.render(f"graphs/{loss}_graph", format="png")
                # exit()
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
                scaler.scale(d_loss).backward()
                scaler.step(self.disc_opt)
                scaler.update()

            ## GENERATOR TRAIN

            self.gen_opt.zero_grad()
            with flag_setting:
                fake_img = self.generator(rgb_a, lc_ab, binary_mask)
                fake_img_aug, _ = apply_augmentations(fake_img, None, types=['blit', 'noise'])
                _, g_patchGAN = self.discriminator(fake_img_aug)
                # Use landcover model for lc_gen_fake?
                lc_gen_fake = self.landcover_model(fake_img)
                # Not use softmax if use cross entropy
                # lc_gen_fake = torch.softmax(lc_gen_fake, dim=1)

                if all_lambdas["g_adv_lambda"][0] == 0:
                    g_adv_loss = torch.tensor(0., requires_grad=True).to(device)
                else:
                    g_adv_loss = self.adv_loss_fn(g_patchGAN, torch.ones_like(g_patchGAN))

                # use how bad the discriminator is at generating the fake lc_ab as a generator loss
                # btw does this work as I expect?
                # https://discuss.pytorch.org/t/optimizing-based-on-another-models-output/6935
                # Because fake_img, from self.generator is part of the computational graph of g_adv_loss, this does work in training the generator.
                if g_gen_lc_lambda == 0:
                    g_gen_lc_loss = torch.tensor(0., requires_grad=True).to(device)
                else:
                    # use accuracy?
                    g_gen_lc_loss = self.class_loss_fn(lc_gen_fake, torch.argmax(lc_ab, 1))


                if id_lambda == 0:
                    g_pixel_id_loss = torch.tensor(0., requires_grad=True).to(device)
                    g_feature_id_loss = torch.tensor(0., requires_grad=True).to(device)
                else:
                    id_img = self.generator(rgb_a, lc_a, torch.zeros_like(binary_mask).to(device))
                    g_pixel_id_loss = self.pixel_loss_fn(id_img, rgb_a)
                    feature_id_img = self.relu3_3(id_img)
                    feature_rgb_a = self.relu3_3(rgb_a)
                    g_feature_id_loss = self.feature_loss_fn(feature_id_img, feature_rgb_a)

                local_g_style_loss = 0
                local_g_pixel_loss = 0
                local_g_feature_loss = 0

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
                        gen_local_area = fake_img[j,:,r_w-config.local_area_margin:r_w + mask_size_w+config.local_area_margin, r_h-config.local_area_margin:r_h+mask_size_h+config.local_area_margin]
                        # new area
                        rgb_ab_local_area = rgb_ab[j,:,r_w-config.local_area_margin:r_w + mask_size_w + config.local_area_margin, r_h - config.local_area_margin:r_h+mask_size_h + config.local_area_margin]
                        # old area
                        rgb_a_local_area = rgb_a[j,:,r_w-config.local_area_margin:r_w + mask_size_w + config.local_area_margin, r_h-config.local_area_margin:r_h+mask_size_h+config.local_area_margin]
                    
                        if all_lambdas["local_g_style_lambda"][0] != 0:
                            # use old or new area to compare with the style loss
                            if args.local_style_new:
                                local_g_style_loss += self.style_loss_fn(gen_local_area, rgb_ab_local_area)
                            else:
                                local_g_style_loss += self.style_loss_fn(gen_local_area, rgb_a_local_area)

                        if all_lambdas["local_g_pixel_lambda"][0] != 0:
                            local_g_pixel_loss += self.pixel_loss_fn(gen_local_area, rgb_ab_local_area)

                        if all_lambdas["local_g_feature_lambda"][0] != 0:
                            c, w, h = gen_local_area.shape
                            # use old or new area for feature loss
                            if args.local_feature_new:
                                gen_local_feature = self.relu3_3(gen_local_area.reshape(1, c, w, h))
                                rgb_ab_local_feature = self.relu3_3(rgb_ab_local_area.reshape(1, c, w, h))  
                                local_g_feature_loss += self.feature_loss_fn(gen_local_feature, rgb_ab_local_feature)
                            else:
                                gen_local_feature = self.relu3_3(gen_local_area.reshape(1, c, w, h))
                                rgb_a_local_feature = self.relu3_3(rgb_a_local_area.reshape(1, c, w, h))  
                                local_g_feature_loss += self.feature_loss_fn(gen_local_feature, rgb_a_local_feature)
                                
                        fake_img_unchanged_area[j,:,r_w-config.local_area_margin:r_w + mask_size_w+config.local_area_margin, r_h-config.local_area_margin:r_h+mask_size_h+config.local_area_margin] = torch.zeros(3, mask_size_w + (config.local_area_margin * 2), mask_size_h + (config.local_area_margin * 2))
                        rgb_a_unchanged_area[j,:,r_w-config.local_area_margin:r_w + mask_size_w+config.local_area_margin, r_h-config.local_area_margin:r_h+mask_size_h+config.local_area_margin] = torch.zeros(3, mask_size_w + (config.local_area_margin * 2), mask_size_h + (config.local_area_margin * 2))
                        # fake_img_unchanged_area[j,:,r_w:r_w + mask_size_w, r_h:r_h+mask_size_h] = torch.zeros(3, mask_size_w, mask_size_h)
                        # rgb_a_unchanged_area[j,:,r_w:r_w + mask_size_w, r_h:r_h+mask_size_h] = torch.zeros(3, mask_size_w, mask_size_h)


                if local_g_style_loss == 0:
                    local_g_style_loss = torch.tensor(0., requires_grad=True).to(device)
                if local_g_pixel_loss == 0:
                    local_g_pixel_loss = torch.tensor(0., requires_grad=True).to(device)
                if local_g_feature_loss == 0:
                    local_g_feature_loss = torch.tensor(0., requires_grad=True).to(device)

                # Don't calculate if lambda is 0, since
                if all_lambdas["g_pixel_lambda"][0] == 0:
                    g_pixel_loss = torch.tensor(0., requires_grad=True).to(device)
                else:
                    g_pixel_loss = self.pixel_loss_fn(fake_img_unchanged_area, rgb_a_unchanged_area)
                
                if all_lambdas["g_feature_loss_lambda"][0] == 0:
                    g_feature_loss = torch.tensor(0., requires_grad=True).to(device)
                else:
                    fake_img_feature = self.relu3_3(fake_img_unchanged_area)
                    rgb_a_feature = self.relu3_3(rgb_a_unchanged_area)                    
                    g_feature_loss = self.feature_loss_fn(fake_img_feature, rgb_a_feature)
             
                local_g_pixel_loss = local_g_pixel_loss / (len(masked_areas[0][0]) * config.num_inpaints)
                local_g_style_loss = local_g_style_loss / (len(masked_areas[0][0]) * config.num_inpaints)
                local_g_feature_loss = local_g_feature_loss / (len(masked_areas[0][0]) * config.num_inpaints)


                g_loss = (
                    (g_adv_loss * all_lambdas["g_adv_lambda"][0])
                    + (g_feature_loss * all_lambdas["g_feature_loss_lambda"][0])
                    + (g_gen_lc_loss * g_gen_lc_lambda)
                    + (g_pixel_loss * all_lambdas["g_pixel_lambda"][0])
                    + (
                        (g_pixel_id_loss + g_feature_id_loss) * id_lambda
                    )
                    + (local_g_style_loss * all_lambdas["local_g_style_lambda"][0])
                    + (local_g_pixel_loss * all_lambdas["local_g_pixel_lambda"][0])
                    + (local_g_feature_loss * all_lambdas["local_g_feature_lambda"][0])
                )  



                if evaluation:
                    # Idea from Stefan, use rgb_ab and lc_a as input and rgb_a as targer
                    fake_a = self.generator(rgb_ab, lc_a, binary_mask)
                    gen_lc_fake_a = self.landcover_model(fake_a)
                    gen_lc_a = self.landcover_model(rgb_a) 

                    iou_gen_lc_fake_a_vs_gen_lc_a = calc_all_IoUs(gen_lc_fake_a, gen_lc_a)
                    iou_gen_lc_fake_a_vs_lc_a = calc_all_IoUs(gen_lc_fake_a, lc_a)
                    iou_gen_lc_a_vs_lc_a = calc_all_IoUs(lc_a, gen_lc_a)

            if not evaluation:

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                # Uncomment this if you want to save the visualization of the computational graph #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # from torchviz import make_dot
                # for loss in losses_names:
                #     if loss[0] != "d":
                #         print(loss, eval(loss))
                #         scaler.scale(eval(loss)).backward(retain_graph=True)
                #         make_dot(eval(loss)).render(f"graphs/{loss}_graph", format="png")
                # exit()
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
                scaler.scale(g_loss).backward()
                scaler.step(self.gen_opt)
                scaler.update()

            total_batches += 1
            for name in losses_names:
                epoch_losses[name] += eval(name).item()

            if evaluation:
                for iou in possible_ious:
                    epoch_ious[iou] += eval(iou)

        # dynamically change the lambdas, based off how bad their loss is
        if self.use_dynamic_lambdas:
            total_g_loss = 0 
            for lamdba_name in dynamic_lambdas_names:
                loss_name = all_lambdas[lamdba_name][1]
                total_g_loss += epoch_losses[loss_name]

            for lamdba_name in dynamic_lambdas_names:
                loss_name = all_lambdas[lamdba_name][1]
                # if lambda is 0 then it's value won't increase 
                if all_lambdas[lamdba_name][0] != 0:
                    all_lambdas[lamdba_name][0] = max((epoch_losses[loss_name]) / total_g_loss, min_lambda_value)

        
        loss_dict = self.val_losses if evaluation else self.losses

        for name in losses_names:
            loss_dict[name].append(epoch_losses[name] / total_batches)

        if evaluation:
            for iou in possible_ious:
                self.ious[iou].append(epoch_ious[iou] / total_batches)
            with open(self.iou_file, "w") as f:
                json.dump(self.ious, f)

            plot_ious(self.ious, args.losses_dir)

   
    def save_models(self):
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        torch.save(self.generator.state_dict(), os.path.join(self.models_dir, "generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.models_dir, "discriminator.pt"))

    def load_models(self):
        self.generator.load_state_dict(torch.load(os.path.join(self.models_dir, "generator.pt")))
        self.discriminator.load_state_dict(torch.load(os.path.join(self.models_dir, "discriminator.pt")))

    def save_losses(self):
        with open(self.losses_file, "w") as f:
            json.dump(self.losses, f)

        with open(self.val_losses_file, "w") as f:
            json.dump(self.val_losses, f)

    def train(self):
        eval_dir = args.eval_dir
        # cannot change this number between runs
        num_save_examples = 2
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
            if args.dynamic_lambdas_epoch != -1 and epoch >= args.dynamic_lambdas_epoch:
                self.use_dynamic_lambdas = True

            self.loop_description = f"{epoch} / {num_epochs + start_epoch - 1}"
            self.epoch(evaluation=False)
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