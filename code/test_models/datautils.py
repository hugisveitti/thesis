import numpy as np
import ast
import os
import torch

num_classes = 9

lc_pixels = {
    0: str([255, 255, 255, 255]),
    1: str([210, 0, 0, 255]),
    2: str([253, 211, 39, 255]),
    3: str([35, 152, 0, 255]),
    4: str([8, 98, 0, 255]),
    5: str([249, 150, 39, 255]),
    6: str([141, 139, 0, 255]),
    7: str([149, 107, 196, 255]),
    8: str([77, 37, 106, 255]),
}

lc_labels_classes = {
    0: 'Other',
    1: 'Artificial surfaces and constructions',
    2: 'Cultivated areas',
    3: 'Broadleaf tree cover',
    4: 'Coniferous tree cover',
    5: 'Herbaceous vegetation',
    6: 'Moors and Heathland',
    7: 'Marshes',
    8: 'Peatbogs',
}

# to revert to an image
def create_img_from_classes(img_classes):
    img = np.zeros((img_classes.shape[0], img_classes.shape[1], 4), dtype=np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.equal(img_classes[i,j], np.zeros(9)).all():
                img[i,j, :] = [0, 0, 0, 255]
            else:
                img[i,j] = ast.literal_eval(lc_pixels[np.argmax(img_classes[i,j])])
    return img

def unprocess(ma):
    ma = np.array(ma.cpu(), dtype=np.float32)[0]
    ma = np.moveaxis(ma, 0, -1)
    return ma


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
