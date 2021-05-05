import torch
from generator import Generator
from discriminator import Discriminator
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from utils import save_some_examples, plot_losses, save_models

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
PRETRAIN_EPOCH = 100
ALPHA = 0.0004

def train(discriminator, generator, loader, optimizer_disriminator, optimizer_generator, g_scaler, d_scaler, mse_loss_fn, bce_loss_fn):
    loop = tqdm(loader, leave=True, position=0)
    total_d_loss = 0
    total_g_loss = 0
    total_mse_loss = 0
    total = 0
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1

    for idx, (input_img, target_classes) in enumerate(loop):
        input_img = input_img.to(device)
        target_classes = target_classes.to(device)
        
        with torch.cuda.amp.autocast():
            fake_classes = generator(input_img)
            d_real = discriminator(target_classes)
            d_fake = discriminator(fake_classes.detach())

            d_fake_loss = bce_loss_fn(d_fake, torch.zeros_like(d_fake))
            d_real_loss = bce_loss_fn(d_real, torch.ones_like(d_real))
            
            d_loss = d_fake_loss + d_real_loss

            # is this correct WGAN loss?
            #d_fake = d_fake.mean()
            #d_real = d_real.mean()

            #d_loss = d_real - d_fake

            # need gradient penalty


        
        discriminator.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(optimizer_disriminator)
        d_scaler.update()
        total_d_loss += d_loss.item()

        with torch.cuda.amp.autocast():
            fake_classes = generator(input_img)
            d_real = discriminator(target_classes)
            d_fake = discriminator(fake_classes.detach())

            d_fake_loss = bce_loss_fn(d_fake, torch.ones_like(d_fake))
            d_real_loss = bce_loss_fn(d_real, torch.zeros_like(d_real))
            gan_loss = d_fake_loss + d_real_loss
           # gan_loss = d_fake.mean()
            
            # not sure mse is best for classes 
          #  target_classes = target_classes.clone().detach().requires_grad_(True).type(torch.long)
            target_classes = torch.squeeze(target_classes, dim=2)
            mse = mse_loss_fn(fake_classes, target_classes)
            g_loss = mse + gan_loss * ALPHA

        optimizer_generator.zero_grad()
        
        g_scaler.scale(g_loss).backward()
        g_scaler.step(optimizer_generator)
        g_scaler.update()
        total_g_loss += gan_loss.item()
        total_mse_loss += mse.item()

        total += 1
    
    return total_g_loss / total, total_d_loss / total, total_mse_loss / total




def main():
    
    discriminator = Discriminator(14).to(device=device)
    generator = Generator(3, 14).to(device=device)

    # betas?
    optimizer_disriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    dataset = MyDataset("../../data/train/", 100)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = MyDataset("../../data/val/")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    losses = {}
    losses["mse"] = []
    losses["g"] = []
    losses["d"] = []

    total_epochs = 30
    for epoch in range(total_epochs):
        print(f"\nepoch {epoch}")
        # mse loss probably not a good idea
        # CrossEntropyLoss probably better
        g_loss, d_loss, mse_loss = train(discriminator, generator, loader, optimizer_disriminator, optimizer_generator,g_scaler, d_scaler, nn.MSELoss(), nn.BCEWithLogitsLoss())
        losses["mse"].append(mse_loss)
        losses["g"].append(g_loss)
        losses["d"].append(d_loss)
        plot_losses(losses, "loss")
        save_some_examples(generator,val_loader,epoch,"eval",device)
    save_models(generator, discriminator, epoch)



if __name__ == "__main__":
    main()