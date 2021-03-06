import numpy as np
import ast
import os


lc_pixels_9 = {
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

lc_labels_classes_9 = {
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
def create_img_from_classes_9(img_classes):
    img = np.zeros((img_classes.shape[0], img_classes.shape[1], 4), dtype=np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.equal(img_classes[i,j], np.zeros(9)).all():
                img[i,j, :] = [0, 0, 0, 255]
            else:
                img[i,j] = ast.literal_eval(lc_pixels_9[np.argmax(img_classes[i,j])])
    return img

def unprocess(ma):
    ma = np.array(ma, dtype=np.float32)[0]
    ma = np.moveaxis(ma, 0, -1)
    return ma


lc_pixels_14 = {
    0: str([255, 255, 255, 255]),
    1: str([210, 0, 0, 255]),
    2: str([253, 211, 39, 255]),
    3: str([176, 91, 16, 255]),
    4: str([35, 152, 0, 255]),
    5: str([8, 98, 0, 255]),
    6: str([249, 150, 39, 255]),
    7: str([141, 139, 0, 255]),
    8: str([95, 53, 6, 255]),
    9: str([149, 107, 196, 255]),
    10: str([77, 37, 106, 255]),
    11: str([154, 154, 154, 255]),
    12: str([106, 255, 255, 255]),
    13: str([20, 69, 249, 255]),
}

lc_labels_classes_14 = {
    0:"Clouds",
    1:"Artificial surfaces and constructions",
    2:"Cultivated areas",
    3:"Vineyards",
    4:"Broadleaf tree cover",
    5:"Coniferous tree cover",
    6:"Herbaceous vegetation",
    7:"Moors and Heathland",
    8:"Sclerophyllous vegetation",
    9:"Marshes",
    10:"Peatbogs",
    11:"Natural material surfaces",
    12:"Permanent snow covered surfaces",
    13:"Water bodies",
   # 14:"No data", not used same as clouds
}

# to revert to an image
def create_img_from_classes_14(img_classes):
    img = np.zeros((img_classes.shape[0], img_classes.shape[1], 4), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ast.literal_eval(lc_pixels_14[np.argmax(img_classes[i,j])])
    return img
