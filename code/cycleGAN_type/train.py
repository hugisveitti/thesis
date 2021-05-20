from generator import Generator
from discriminator import Discriminator
import torch
from dataset import SatelliteDataset
from torch.utils.data import DataLoader
from utils import save_example, save_models, plot_losses
import torch.nn as nn
from tqdm import tqdm

LEARNING_RATE = 0.0002
BATCH_SIZE = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_CYCLE = 10

gan_loss = nn.MSELoss()
img_cycle_loss = nn.L1Loss()
classes_cycle_loss = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


def train(img_generator, classes_generator, img_discriminator, classes_discriminator, i_d_opt, i_g_opt, c_d_opt, c_g_opt, loader):
    
    loop = tqdm(loader, position=0, leave=True)

    total_d_i_loss = 0
    total_d_c_loss = 0

    total_g_i_gan_loss = 0
    total_g_c_gan_loss = 0

    total_cycle_c_loss = 0
    total_cycle_i_loss = 0
    
    total = 0

    for img, classes in loop:
        img, classes = img.to(device), classes.to(device)

        i_d_opt.zero_grad()
        i_g_opt.zero_grad()
        c_d_opt.zero_grad()
        c_g_opt.zero_grad()

        with torch.cuda.amp.autocast():
            fake_classes = classes_generator(img)
            fake_img = img_generator(classes)

            d_i_real = img_discriminator(img)
            d_i_fake = img_discriminator(fake_img)

            d_i_fake_loss = gan_loss(d_i_fake, torch.zeros_like(d_i_fake))
            d_i_real_loss = gan_loss(d_i_real, torch.ones_like(d_i_real))

            d_i_loss = (d_i_fake_loss + d_i_real_loss) / 2

            d_c_real = classes_discriminator(classes)
            d_c_fake = classes_discriminator(fake_classes)

            d_c_fake_loss = gan_loss(d_c_fake, torch.zeros_like(d_c_fake))
            d_c_real_loss = gan_loss(d_c_real, torch.ones_like(d_c_real))

            d_c_loss = (d_c_fake_loss + d_c_real_loss) / 2


            g_i_gan_loss = gan_loss(d_i_fake, torch.ones_like(d_i_fake))
            
            g_c_gan_loss = gan_loss(d_c_fake, torch.ones_like(d_c_fake))

            cycle_classes = classes_generator(fake_img)
            cycle_img = img_generator(fake_classes)

            classes_target = torch.argmax(classes, dim=1)
            g_c_cycle_loss = classes_cycle_loss(cycle_classes, classes_target)

            g_i_cycle_loss = img_cycle_loss(cycle_img, img)
           # g_c_cycle_loss = cycle_loss(cycle_classes, classes)

            
            g_i_loss = g_i_gan_loss + (g_i_cycle_loss * LAMBDA_CYCLE)
            g_c_loss = g_c_gan_loss + (g_c_cycle_loss * LAMBDA_CYCLE)

        # Identity loss doesnt make sense since inputs have different amounts of channels
        scaler.scale(d_i_loss).backward(retain_graph=True)
        scaler.scale(d_c_loss).backward(retain_graph=True)

        scaler.scale(g_i_loss).backward(retain_graph=True)
        scaler.scale(g_c_loss).backward()

        scaler.step(i_d_opt)
        scaler.step(i_g_opt)
        scaler.step(c_d_opt)
        scaler.step(c_g_opt)

        scaler.update()

        total_d_i_loss += d_i_loss.item()
        total_d_c_loss = d_c_loss.item()

        total_g_i_gan_loss = g_i_gan_loss.item()
        total_g_c_gan_loss = g_c_gan_loss.item()

        total_cycle_c_loss = g_c_cycle_loss.item()
        total_cycle_i_loss = g_i_cycle_loss.item()

        total += 1

    d_i_loss = total_d_i_loss / total
    d_c_loss = total_d_c_loss / total

    g_i_gan_loss = total_g_i_gan_loss / total
    g_c_gan_loss = total_g_c_gan_loss / total

    cycle_i_loss = total_cycle_i_loss / total
    cycle_c_loss = total_cycle_c_loss / total
             
    return d_i_loss, d_c_loss, g_i_gan_loss, g_c_gan_loss, cycle_i_loss, cycle_c_loss


def main():

    img_generator = Generator(14, 3).to(device)
    classes_generator = Generator(3, 14).to(device)

    img_discriminator = Discriminator(3).to(device)
    classes_discriminator = Discriminator(14).to(device)

    betas = (0.5, 0.999)
    # Betas ?????
    i_d_opt = torch.optim.Adam(img_discriminator.parameters(), lr=LEARNING_RATE, betas=betas)
    i_g_opt = torch.optim.Adam(img_generator.parameters(), lr=LEARNING_RATE, betas=betas)

    c_d_opt = torch.optim.Adam(classes_discriminator.parameters(), lr=LEARNING_RATE, betas=betas)
    c_g_opt = torch.optim.Adam(classes_generator.parameters(), lr=LEARNING_RATE, betas=betas)

    # Include the replay buffer????

    d = SatelliteDataset("../../data/train", 100) 
    loader = DataLoader(d, BATCH_SIZE, shuffle=True, num_workers=2)

    val_d = SatelliteDataset("../../data/val")
    val_loader = DataLoader(val_d, 1)

    total_epochs = 3
    losses = {}
    losses["d_i_loss"] = []
    losses["d_c_loss"] = []
    losses["g_i_gan_loss"] = []
    losses["g_c_gan_loss"] = []
    losses["cycle_i_loss"] = []
    losses["cycle_c_loss"] = []
    for epoch in range(total_epochs):
        print("\nepoch", epoch)
        d_i_loss, d_c_loss, g_i_gan_loss, g_c_gan_loss, cycle_i_loss, cycle_c_loss = train(img_generator, classes_generator, img_discriminator, classes_discriminator, i_d_opt, i_g_opt, c_d_opt, c_g_opt, loader)
        losses["d_i_loss"].append(d_i_loss)
        losses["d_c_loss"].append(d_c_loss)
        losses["g_i_gan_loss"].append(g_i_gan_loss)
        losses["g_c_gan_loss"].append(g_c_gan_loss)
        losses["cycle_i_loss"].append(cycle_i_loss)
        losses["cycle_c_loss"].append(cycle_c_loss)
        save_example(classes_generator, img_generator, val_loader, epoch, "eval", device)
        plot_losses(losses, "losses")
    save_models(classes_generator, img_generator, classes_discriminator, img_discriminator, epoch)


if __name__ == "__main__":
    main()