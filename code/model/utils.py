import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from datautils import unprocess, create_img_from_classes
import config

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
    
    for rgb_a, rgb_b, lc_a, lc_b, binary_mask, lc_ab, masked_areas in loader:
        rgb_a, rgb_b, lc_a, lc_b, binary_mask = rgb_a.to(device), rgb_b.to(device), lc_a.to(device), lc_b.to(device), binary_mask.to(device)

        with torch.cuda.amp.autocast():
            fake_img = generator(rgb_a, lc_a, binary_mask)
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
            lc_ab = unprocess(lc_ab.cpu(), )
            lc_ab = create_img_from_classes(lc_ab)
            binary_mask = np.array(binary_mask.cpu())[0][0]

            gen_lc = unprocess(gen_lc.detach())
            gen_lc = create_img_from_classes(gen_lc)

            fake_gen_lc = unprocess(fake_gen_lc.detach())
            fake_gen_lc = create_img_from_classes(fake_gen_lc)

            fig, ax = plt.subplots(3, 2, figsize=(10,15))
            
            fig.tight_layout()
            plt.grid(False)

            ax[0,0].imshow(lc_ab)
            ax[0,0].set_title("lc_ab (input)")
            ax[0,0].grid(False)

            ax[0,1].imshow(binary_mask, cmap="gray")
            ax[0,1].set_title("binary_mask (input)")
            ax[0,1].grid(False)

            ax[1,0].imshow(rgb_a)
            ax[1,0].set_title("rgb_a (input)")
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
num_classes = 14
def calc_single_IoUs(lc_a, lc_b):
    """
    Calculates the mean IoU,
    the weights of each class depend on their total ratio in lc_a
    """
    c_ratio = []
    ious = []
    for c in range(num_classes):
        iou = IoU(lc_a, lc_b, c)
        n = lc_a.shape[0] * lc_a.shape[1]
        if iou != None:
            ious.append(iou)
            c_ratio.append(lc_a[lc_a==c].shape[0] / n)
    return np.sum(np.array(ious) * np.array(c_ratio))


def calc_all_IoUs(lc_a, lc_b):
    lc_a = torch.argmax(lc_a, dim=1)
    lc_b = torch.argmax(lc_b, dim=1)
    # lc_a.shape[0] are the batches
    return np.mean([calc_single_IoUs(lc_a[i], lc_b[i]) for i in range(lc_a.shape[0])])

class StyleLoss(nn.Module):
    
    def __init__(self, relu3_3, device):
        super(StyleLoss, self).__init__()
        self.relu3_3 = relu3_3 
        
    def forward(self, img1, img2):
        if len(img1.shape) == 3:
            img1 = img1.reshape((1, img1.shape[0], img1.shape[1], img1.shape[2]))
            img2 = img2.reshape((1, img2.shape[0], img2.shape[1], img2.shape[2]))
        phi1 = self.relu3_3(img1)
        phi2 = self.relu3_3(img2)

        batch_size, c, h, w = phi1.shape
        psi1 = phi1.reshape((batch_size, c, w*h))
        psi2 = phi2.reshape((batch_size, c, w*h))

        gram1 = torch.matmul(psi1, torch.transpose(psi1, 1, 2)) / (c*h*w)
        gram2 = torch.matmul(psi2, torch.transpose(psi2, 1, 2)) / (c*h*w)
        # as described in johnson et al.
        style_loss = torch.sum(torch.norm(gram1 - gram2, p = "fro", dim=(1,2))) / batch_size

        # Why so many infs???
        # not always inf but sometimes
        # the generator is not apart of this computation graph so I dont think this works.
        if style_loss.isinf().any() or style_loss.isnan().any():
            return torch.tensor(1., requires_grad=True).to(config.device)
        return style_loss


def calc_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    target = torch.argmax(target, dim=1)
    pred = pred.flatten()
    target = target.flatten() 
    return torch.tensor(target[target == pred].shape[0] / target.shape[0], requires_grad=True)