import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json

from plot_losses import plot_losses
from landcover_model import LandcoverModel

from dataset import SatelliteDataset
from utils import save_example, calc_all_IoUs, calc_accuracy
import config

device = "cuda"
LEARNING_RATE = 0.0002
scaler = torch.cuda.amp.GradScaler()



parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../../data/grid_dir")
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--eval_dir", type=str, default="eval")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--losses_dir", type=str, default="losses")
parser.add_argument("--models_dir", type=str, default="models/")
parser.add_argument("--log_file", type=str, default="training_log.txt")

args = parser.parse_args()

log_file = args.log_file

losses_names = ["loss", "iou_loss", "class_loss", "accuracy"]

class Train:

    def __init__(self):
        data_dir = args.data_dir
        print("arguments", args)

        self.lc_model = LandcoverModel().to(device)
        # betas? 0.5 and 0.999 are used in pix2pix
        self.opt = torch.optim.Adam(self.lc_model.parameters(), lr=LEARNING_RATE)

        d = SatelliteDataset(os.path.join(data_dir,"train"))
        self.loader = DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

        d_val = SatelliteDataset(os.path.join(data_dir,"val"))
        self.val_loader = DataLoader(d_val, batch_size=args.val_batch_size, num_workers=args.num_workers)
       
        print(f"{len(os.listdir(os.path.join(data_dir,'train/rgb')))} files in train/rgb")
        print(f"{len(os.listdir(os.path.join(data_dir,'train/lc_sieve')))} files in train/lc_sieve")

        self.class_loss_fn = nn.CrossEntropyLoss()

        self.models_dir = args.models_dir
        self.loop_description = ""
        if args.load_model:
            self.load_model()

        self.losses = {}
        self.val_losses = {}
        for name in losses_names:
            self.losses[name] = []
            self.val_losses[name] = []

        if not os.path.exists(args.losses_dir):
            os.mkdir(args.losses_dir)

        
        self.losses_file = os.path.join(args.losses_dir, "train_losses.json")
        self.val_losses_file = os.path.join(args.losses_dir, "val_losses.json")
        if os.path.exists(self.losses_file) and args.load_model:
            with open(self.losses_file, "r") as f:
                self.losses = json.load(f)

        if os.path.exists(self.val_losses_file) and args.load_model:
            with open(self.val_losses_file, "r") as f:
                self.val_losses = json.load(f)

        

    def epoch(self, evaluation=False):

        if evaluation:
            loop = tqdm(self.val_loader, position=0)
            loop.set_description(f"{self.loop_description} evaluation")
            self.lc_model.eval()
        else:
            loop = tqdm(self.loader, position=0)
            loop.set_description(self.loop_description)

            self.lc_model.train()

        epoch_losses = {}
        for name in losses_names:
            epoch_losses[name] = 0

        total = 0

        for rgb, lc in loop:
            rgb, lc = rgb.to(device), lc.to(device)

            flag_setting = torch.cuda.amp.autocast()
            if evaluation:
                flag_setting = torch.cuda.amp.autocast() and torch.no_grad()
        

            self.opt.zero_grad()
            with flag_setting:
                gen_lc = self.lc_model(rgb)
                class_loss = self.class_loss_fn(gen_lc, torch.argmax(lc, 1))
                iou_loss = 1 - calc_all_IoUs(gen_lc, lc)
                accuracy = calc_accuracy(gen_lc, lc)
            
            loss = class_loss + iou_loss

            if not evaluation:
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()

            total += 1
            for name in losses_names:
                epoch_losses[name] += eval(name).item()

        
        loss_dict = self.val_losses if evaluation else self.losses

        for name in losses_names:
            loss_dict[name].append(epoch_losses[name] / total)

   
    def save_model(self):
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        torch.save(self.lc_model.state_dict(), f"{self.models_dir}/lc_model.pt")

    def load_model(self):
        self.lc_model.load_state_dict(torch.load(f"{self.models_dir}/lc_model.pt"))

    def save_losses(self):
        with open(self.losses_file, "w") as f:
            json.dump(self.losses, f)

        with open(self.val_losses_file, "w") as f:
            json.dump(self.val_losses, f)

    def train(self):
        eval_dir = args.eval_dir

        # save deterministic samples
        save_example(self.lc_model, self.val_loader, 0, eval_dir, device)

        if os.path.exists(eval_dir) and args.load_model:
            start_epoch = int(len(os.listdir(eval_dir)) )
        else:
            start_epoch = 1
        num_epochs = args.num_epochs
        with open(log_file, "a") as f:
            f.write(f"start epoch {start_epoch}\n\n\n")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.loop_description = f"{epoch} / {num_epochs + start_epoch - 1}"
            self.epoch(False)
            save_example(self.lc_model, self.val_loader, epoch, eval_dir, device)
            self.epoch(evaluation=True)
            plot_losses(args.losses_dir, self.losses, self.val_losses)

            self.save_model()
            self.save_losses()


def run():
    t = Train()
    t.train()

if __name__ == "__main__":
    run()