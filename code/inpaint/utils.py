import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from datautils import unprocess, create_img_from_classes

def save_example(generator, discriminator, folder, epoch, loader, device, num_examples = 5):
    # In pix2pix they talk about using the dropout as the random noise
    generator.eval()
    for m in generator.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    discriminator.eval()
    for m in discriminator.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    if not os.path.exists(folder):
        os.mkdir(folder)

    example = 0
    
    for rgb_a, lc_a, rgb_a_masked, masked_areas in loader:
        rgb_a, lc_a, rgb_a_masked = rgb_a.to(device), lc_a.to(device), rgb_a_masked.to(device)

        with torch.cuda.amp.autocast():
            fake_img = generator(rgb_a_masked, lc_a)
            gen_lc, _ = discriminator(rgb_a)
            fake_gen_lc, _ = discriminator(fake_img)
            if torch.isnan(fake_img).any():
                print("fake rgb has nan") 

            fake_img = fake_img.cpu()
            gen_lc = gen_lc.cpu()
            fake_gen_lc = fake_gen_lc.cpu()

            fake_img = unprocess(fake_img.detach())
            fake_img = np.array(fake_img, dtype=np.float32)

            rgb_a = unprocess(rgb_a.cpu())
            lc_a = unprocess(lc_a.cpu(), )
            lc_a = create_img_from_classes(lc_a)
            rgb_a_masked = unprocess(rgb_a_masked.cpu().detach())

            gen_lc = unprocess(gen_lc.detach())
            gen_lc = create_img_from_classes(gen_lc)

            fake_gen_lc = unprocess(fake_gen_lc.detach())
            fake_gen_lc = create_img_from_classes(fake_gen_lc)

            fig, ax = plt.subplots(3, 2, figsize=(10,15))
            
            fig.tight_layout()
            plt.grid(False)

            ax[0,0].imshow(lc_a)
            ax[0,0].set_title("lc_a (input)")
            ax[0,0].grid(False)

            ax[0,1].imshow(rgb_a_masked)
            ax[0,1].set_title("rgb_a_masked (input)")
            ax[0,1].grid(False)

            ax[1,0].imshow(rgb_a)
            ax[1,0].set_title("rgb_a (target)")
            ax[1,0].grid(False)

            ax[1,1].imshow(fake_img)
            ax[1,1].set_title("generate image")
            ax[1,1].grid(False)

            ax[2,0].imshow(gen_lc)
            ax[2,0].set_title("gen lc of rgb_a")
            ax[2,0].grid(False)

            ax[2,1].imshow(fake_gen_lc)
            ax[2,1].set_title("gen lc of gen img")
            ax[2,1].grid(False)


            plt.savefig(f"{folder}/epoch_{epoch}_{example}.png")
            plt.close()

            example += 1
            if example == num_examples:
                break


    generator.train() 
    discriminator.train()



def IoU(lc_a, lc_b, cla):
    union = lc_a[(lc_a == cla) | (lc_b == cla)].shape[0]
    if union == 0:
        return None
    return len(((lc_a == cla) | (lc_b == cla))[(lc_a == cla) & (lc_b == cla)]) / union

def calc_single_IoUs(lc_a, lc_b):
    """
    Calculates the mean IoU,
    the weights of each class depend on their total ratio in lc_a
    """
    c_ratio = []
    ious = []
    for c in range(14):
        iou = IoU(lc_a, lc_b, c)
        n = lc_a.shape[0] * lc_a.shape[1]
        if iou != None:
            ious.append(iou)
            c_ratio.append(lc_a[lc_a==c].shape[0] / n)
    return np.sum(np.array(ious) * np.array(c_ratio))


def calc_all_IoUs(lc_a, lc_b):
    lc_a = torch.argmax(lc_a, dim=1)
    lc_b = torch.argmax(lc_b, dim=1)
    return np.mean([calc_single_IoUs(lc_a[i], lc_b[i]) for i in range(lc_a.shape[0])])

