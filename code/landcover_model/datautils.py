import numpy as np
import ast
import os
import config


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
            if np.equal(img_classes[i,j], np.zeros(config.num_classes)).all():
                img[i,j, :] = [0, 0, 0, 255]
            else:
                img[i,j] = ast.literal_eval(lc_pixels[np.argmax(img_classes[i,j])])
    return img

def unprocess(ma):
    ma = np.array(ma, dtype=np.float32)[0]
    ma = np.moveaxis(ma, 0, -1)
    return ma