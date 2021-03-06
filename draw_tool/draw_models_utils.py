import torch
import ast
import numpy as np
num_classes = 9

def lc_to_sieve(lc):
    lc_sieve = np.zeros((256, 256, 3))
    for i in range(lc.shape[0]):
        for j in range(lc.shape[1]):
            num = np.argmax(lc[i,j])
            pix = lc_pixels[num]
            l = ast.literal_eval(pix)
            lc_sieve[i,j,:] = l[:3]

    return lc_sieve


if num_classes == 9:
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
    possible_lc_classes = {
        str([255, 255, 255]): 0,
        str([210, 0, 0]): 1,
        str([253, 211, 39]): 2,
        str([35, 152, 0, ]): 3,
        str([8, 98, 0, ]): 4,
        str([249, 150, 39, ]): 5,
        str([141, 139, 0, ]): 6,
        str([149, 107, 196, ]): 7,
        str([77, 37, 106, ]): 8,
        str([20, 69, 249,]): 0,
        str([106, 255, 255,]): 0,
        str([154, 154, 154,]): 0,
        str([95, 53, 6]): 0,
        str([176, 91, 16]): 0,
    }
else:
    possible_lc_classes = {
        str([255, 255, 255]): 0,
        str([210, 0, 0]): 1,
        str([253, 211, 39]): 2,
        str([176, 91, 16]): 3,
        str([35, 152, 0]): 4,
        str([8, 98, 0]): 5,
        str([249, 150, 39]): 6,
        str([141, 139, 0]): 7,
        str([95, 53, 6]): 8,
        str([149, 107, 196,]): 9,
        str([77, 37, 106,]): 10,
        str([154, 154, 154,]): 11,
        str([106, 255, 255,]): 12,
        str([20, 69, 249,]): 13,
    }

tensor_type = torch.FloatTensor


def create_img_from_classes(img_classes):
    img = np.zeros((img_classes.shape[0], img_classes.shape[1], 4), dtype=np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.equal(img_classes[i,j], np.zeros(num_classes)).all():
                img[i,j, :] = [0, 0, 0, 255]
            else:
                img[i,j] = ast.literal_eval(lc_pixels[np.argmax(img_classes[i,j])])
    return img

def unprocess(ma):
    ma = np.array(ma, dtype=np.float32)[0]
    ma = np.moveaxis(ma, 0, -1)
    return ma