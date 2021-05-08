import torch
import torch.optim as optim
from tqdm import tqdm

import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import save_example, plot_losses, save_models

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0002
BATCH_SIZE = 16
ALPHA = 0.001

def train(discriminator, generator, loader, optimizer_disriminator, optimizer_generator, scaler, generator_loss_fn, adversarial_loss):
    loop = tqdm(loader, leave=True, position=0)
    total_d_loss = 0
    total_g_loss = 0
    total_generator_loss = 0
    total = 0


    for idx, (input_classes, target_img) in enumerate(loop):
        input_classes = input_classes.to(device)
        target_img = target_img.to(device)
        
        with torch.cuda.amp.autocast():
            fake_img = generator(input_classes)
            d_real = discriminator(target_img)
            d_fake = discriminator(fake_img.detach())
            

            d_real_loss = adversarial_loss(d_real, torch.ones_like(d_real))
            d_fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_fake_loss + d_real_loss) / 2 


            g_loss = adversarial_loss(d_fake, torch.ones_like(d_fake))
            
            generator_loss = generator_loss_fn(fake_img, target_img)
            g_loss = generator_loss + (g_loss * ALPHA)


        
        optimizer_disriminator.zero_grad()
        optimizer_generator.zero_grad()

        scaler.scale(d_loss).backward(retain_graph=True)
        scaler.scale(g_loss).backward()

        scaler.step(optimizer_disriminator)
        scaler.step(optimizer_generator)
        
        scaler.update()


        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        total_generator_loss += generator_loss.item()

        total += 1
    
    return total_g_loss / total, total_d_loss / total, total_generator_loss / total




def main():
    
    discriminator = Discriminator(3).to(device=device)
    generator = Generator(14, 3).to(device=device)

    # betas?
    optimizer_disriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    dataset = MyDataset("../../data/train/", 100)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    scaler = torch.cuda.amp.GradScaler()

    val_dataset = MyDataset("../../data/val/")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    losses = {}
    losses["l1"] = []
    losses["g"] = []
    losses["d"] = []

    total_epochs = 10
    for epoch in range(total_epochs):
        print(f"\nepoch {epoch}")
        # mse loss probably not a good idea
        # CrossEntropyLoss probably better
        g_loss, d_loss, l1_loss = train(discriminator, generator, loader, optimizer_disriminator, optimizer_generator,scaler, nn.L1Loss(), nn.MSELoss())
        losses["l1"].append(l1_loss)
        losses["g"].append(g_loss)
        losses["d"].append(d_loss)
        plot_losses(losses, "loss mse")
        save_example(generator,val_loader,epoch,"eval",device)
    save_models(generator, discriminator, epoch)



if __name__ == "__main__":
    main()