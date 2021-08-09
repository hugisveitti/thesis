# One script that combines several notebooks that I used to create data from 
# the 'satellite' folder and denmark_landcover.tif

# this creates folders rgb, lc, scl, lc_classes, lc_sieve, reduced_classes
# We used lc_sieve and rgb for training.

import rasterio as rio
from rasterio import warp
from rasterio.features import sieve
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import box
import os
import matplotlib.patches as mpatches
import ast
from tqdm import tqdm
import gc
from PIL import Image

# align classfication data to rgb data because of different spatial resolution
def align(src, dst, src_arr, dst_arr):
    src_new = np.repeat(np.zeros_like(dst_arr[[0]]), src.count, 0)
    warp.reproject(
        src_arr, src_new, 
        src_crs=src.crs, dst_crs=dst.crs,
        src_transform=src.transform, dst_transform=dst.transform,
        dst_resolution=dst.res,
        resampling=warp.Resampling.nearest
    )
    return src_new

# SCL labels
scl_labels = ["no data", "saturated or defective","dark_area_pixels", "cloud_shadows", "vegetation","not vegetated","water", "unclassified","cloud medium probability","cloud medium probability","thin cirrus", "snow"]
scl_labels = [f"{i} {scl_labels[i]}" for i in range(len(scl_labels))]

lc_labels_14 = {
    str([255, 255, 255, 255]):"Clouds",
    str([210,0,0,255]):"Artificial surfaces and constructions",
    str([253,211,39,255]):"Cultivated areas",
    str([176,91,16,255]):"Vineyards",
    str([35,152,0,255]):"Broadleaf tree cover",
    str([8,98,0,255]):"Coniferous tree cover",
    str([249,150,39,255]):"Herbaceous vegetation",
    str([141,139,0,255]):"Moors and Heathland",
    str([95,53,6,255]):"Sclerophyllous vegetation",
    str([149,107,196,255]):"Marshes",
    str([77,37,106,255]):"Peatbogs",
    str([154,154,154,255]):"Natural material surfaces",
    str([106,255,255,255]):"Permanent snow covered surfaces",
    str([20,69,249,255]):"Water bodies",
    str([255,255,255,255]):"No data",
}

water = 6
cloud = [3,8,9]
invalid = [0, 1, 2, 7, 10, 11]
def is_lc_valid(scl):
    classes = scl.flatten()
    n = scl.shape[0] * scl.shape[1]
    
    if n * .4 < classes[classes == water].size:
        # too much water
        return False
    # if more than 10% clouds or no data
    sum_invalid = classes[classes == water].size
    for c in cloud:
        sum_invalid += classes[classes == c].size
    
    for c in invalid:
        sum_invalid += classes[classes == c].size
        
    return sum_invalid < n * 0.1

# Save scl, rgb and lc all in one image
def print_all(scl_256, rgb_256, lc_256, root_dir):
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1,3, figsize=(12,16))
    fig.tight_layout()
    ax[0].imshow(np.moveaxis(lc_256, 0, -1))
    ax[0].set_title("lc")
    ax[1].imshow(np.moveaxis(rgb_256, 0, -1))
    ax[1].set_title("rgb")
    ax[2].imshow(scl_256/11 , cmap=cmap, vmin=0, vmax=1)
    ax[2].set_title("scl")

    patches = [mpatches.Patch(color=np.array(ast.literal_eval(i))/255, label=lc_labels_14[i]) for i in lc_labels_14]
    l1 = plt.legend(handles=patches, loc="best",bbox_to_anchor=(-.3,-.3,1,.1), ncol=3, title="lc labels")

    patches2 = [mpatches.Patch(color=cmap(i/11), label=scl_labels[i]) for i in range(len(scl_labels))]
    l2 = plt.legend(handles=patches2, loc="best",bbox_to_anchor=(0,1.5,1,.1), ncol=3, title="scl labels")
    plt.gca().add_artist(l1)

    fn = rgb_files[idx].split("_")
    fn = f"{root_dir}/all/{fn[0][3:6]}_{w}_{h}.png"
    
    plt.savefig(fn)
    plt.close()

# Save scl, rgb and lc individually
def save_imgs(scl_256, rgb_256, lc_256, root_dir):
    if not is_lc_valid(scl_256):
        return False
    fn = rgb_files[idx].split("_")
    fn = fn[0][3:6]
    pathname = f"{fn}_{w}_{h}.png"
    scl_fn = f"{root_dir}/scl/{pathname}"
    lc_fn = f"{root_dir}/lc/{pathname}"
    rgb_fn = f"{root_dir}/rgb/{pathname}"

    plt.imsave(scl_fn, scl_256)
    lc_256 = np.ascontiguousarray(np.moveaxis(lc_256, 0, -1))
    rgb_256 = np.ascontiguousarray(np.moveaxis(rgb_256, 0, -1))

    plt.imsave(lc_fn, lc_256)
    plt.imsave(rgb_fn, rgb_256)
    plt.close()
    return True

def get_scl_rgb_lc(idx):
    rgb = rio.open(rgb_path + rgb_files[idx])
    scl = rio.open(scl_path + scl_files[idx])
    
    rgb_r = rgb.read()
    scl_a = align(scl, rgb, scl.read(), rgb_r)
    lc_r = lc.read()
    lc_a = align(lc, rgb, lc_r, rgb_r)
    del lc_r
    scl.close()
    rgb.close()
    return scl_a, rgb_r, lc_a

rgb_path = "satellite/rgb/"
scl_path = "satellite/classifications/"
rgb_files = os.listdir(rgb_path)
scl_files = os.listdir(scl_path)

# Rest is train tiles
val_tile = "UNG"
test_tile = "UPG"
# there are 13 total tiles but only 9 unique
# 
# The unique tiles
tiles = []
for f in rgb_files:
    tile = f[3:6]
    if tile not in tiles:
        tiles.append(tile)


# create dirs
grid_dir = "grid_dir"
if not os.path.exists(grid_dir):
    os.mkdir(grid_dir)
possible_dirs = [f"{grid_dir}/train", f"{grid_dir}/test", f"{grid_dir}/val"]
possible_subdirs = ["rgb", "all", "lc", "scl"]
for d in possible_dirs:
    if not os.path.exists(d):
        os.mkdir(d)
    for sd in possible_subdirs:
        full_sd = f"{d}/{sd}"
        if not os.path.exists(full_sd):
            os.mkdir(full_sd)

lc = rio.open("denmark_landcover.tif")
# Here we create the all the images as rgb's
# But we want the lc's to be a .npz one hot encoded into its class
# save the images as a grid
# have a window which slides
# if train, move it 128 pixels (overlap) else 256
print("""
# # # # # # # # # # # # # # #
# Creating rgb, lc, and scl #
# # # # # # # # # # # # # # #
""")
for idx in range(0,len(rgb_files)):
    j=0
    # save 1000 samples
    tile_name = rgb_files[idx].split("_")[0][3:6]
    
    window_slide_size = 128
    if tile_name == test_tile:
        root_dir = f"{grid_dir}/test"
        window_slide_size = 256
    elif tile_name == val_tile:
        root_dir = f"{grid_dir}/val"
        window_slide_size = 256
    else:
        root_dir = f"{grid_dir}/train"
    
    scl_a, rgb_r, lc_a = get_scl_rgb_lc(idx)
    loop = tqdm(range(0, rgb_r.shape[2] - 256, window_slide_size))
    loop.set_description(f"{idx} / {len(rgb_files)-1}, {root_dir}")
    for w in loop:
        for h in range(0, rgb_r.shape[1] - 256, 128):

            lc_256 = lc_a[:,h:h+256,w:w+256]
            rgb_256 = rgb_r[:,h:h+256,w:w+256]
            scl_256 = scl_a[0][h:h+256,w:w+256]

            if save_imgs(scl_256, rgb_256, lc_256, root_dir):
                print_all(scl_256, rgb_256, lc_256, root_dir)
                
######################################
# Create a one hot encoding for lc's #
######################################
print("""
# # # # # # # # # # # # # # #
# Saving lc_classes as .npz #
# # # # # # # # # # # # # # #
""")

lc_classes_14 = {
    str([255, 255, 255, 255]): 0,
    str([210, 0, 0, 255]): 1,
    str([253, 211, 39, 255]): 2,
    str([176, 91, 16, 255]): 3,
    str([35, 152, 0, 255]): 4,
    str([8, 98, 0, 255]): 5,
    str([249, 150, 39, 255]): 6,
    str([141, 139, 0, 255]): 7,
    str([95, 53, 6, 255]): 8,
    str([149, 107, 196, 255]): 9,
    str([77, 37, 106, 255]): 10,
    str([154, 154, 154, 255]): 11,
    str([106, 255, 255, 255]): 12,
    str([20, 69, 249, 255]): 13,
}

lc_pixels_9 = {
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

def get_onehot_14(num):
    arr = np.zeros(14)
    arr[num] = 1
    return arr

def create_classes(idx, root_dir, files):
    img = plt.imread(os.path.join(root_dir, class_folder, files[idx]))*255
    img = np.array(img, dtype=np.uint8)

    img_classes = np.zeros((img.shape[0],img.shape[1], 14))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = str(list(img[i,j]))
            img_classes[i,j] = get_onehot_14(lc_classes_14[a])
    fn = files[idx].split(".")[0]
    
    np.savez_compressed(f"{root_dir}/{new_class_folder}/{fn}", img_classes)
    return img_classes


class_folder = "lc"
new_class_folder = "lc_classes"
grid_dir = "grid_dir"
dirs = [f"{grid_dir}/train/", f"{grid_dir}/val/", f"{grid_dir}/test/"]
for d in dirs:
    if not os.path.exists(os.path.join(d, new_class_folder)):
        os.mkdir(os.path.join(d, new_class_folder))


for d in dirs:
    files = os.listdir(os.path.join(d,class_folder))
    loop = tqdm(range(len(files)))
    loop.set_description(d)
    for idx in loop:
        create_classes(idx, d, files)
        

# Here we want to reduce the classes from 14 to 9
# There are 8 abundant classes, so we combine the rest into one "Other" class
print("""
# # # # # # # # # # # # # # #
#      Reducing classes     #
# # # # # # # # # # # # # # #
""")

class_map = {
    0:0, # clouds -> other
    1:1, # artificial -> artificial
    2:2, # Cultivated -> cultivated
    3:0, # vineyards -> other
    4:3, # broadleaf tree cover -> broadleaf tree cover
    5:4, # Coniferous tree cover -> Coniferous tree cover
    6:5, # Herbaceous vegetation -> Herbaceous vegetation
    7:6, # Moors and Heathland -> Moors and Heathland
    8:0, # Sclerophyllous vegetation -> other
    9:7, # Marches -> Marches
    10:8, # Peatbogs -> Peatbogs
    11:0, # Natural material -> other
    12:0, # snow -> other
    13:0, # water -> other
}

def create_class_from_num(num):
    c = np.zeros(9)
    c[class_map[num]] = 1
    return c


for dataset in ["train", "val", "test"]:
    root_dir = f"{grid_dir}/{dataset}/lc_classes"
    new_dir = f"{grid_dir}/{dataset}/reduced_classes"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    files = os.listdir(root_dir)
    loop = tqdm(range(len(files)))
    loop.set_description(dataset)
    for i in loop:
        fn = files[i] 
        with np.load(os.path.join(root_dir, fn)) as f:
            lc = f["arr_0"]
        lc = np.argmax(lc, axis=2)
        reduced_lc = [create_class_from_num(i) for i in lc.flatten()]
        reduced_lc = np.array(reduced_lc).reshape(256,256,9)
        np.savez_compressed(os.path.join(new_dir, fn), reduced_lc)


lc_labels_9 = {
    str([255, 255, 255, 255]):"Other",
    str([210,0,0,255]):"Artificial surfaces and constructions",
    str([253,211,39,255]):"Cultivated areas",
    str([35,152,0,255]):"Broadleaf tree cover",
    str([8,98,0,255]):"Coniferous tree cover",
    str([249,150,39,255]):"Herbaceous vegetation",
    str([141,139,0,255]):"Moors and Heathland",
    str([149,107,196,255]):"Marshes",
    str([77,37,106,255]):"Peatbogs",
}

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

# to revert to an image
def create_img_pixels(img_classes):
    img = np.zeros((img_classes.shape[0], img_classes.shape[1], 4), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ast.literal_eval(lc_pixels_9[np.argmax(img_classes[i,j])])
    return img



print("""
# # # # # # # # # # # # # # #
#        Doing sieve        #
# # # # # # # # # # # # # # #
""")

sieve_size = 5

def get_one_hot_9(num):
    c = np.zeros(9)
    c[num] = 1
    return c


def do_sieve(root_dir, file):

    with np.load(os.path.join(root_dir, file)) as n_file:
        lc = n_file["arr_0"]
        lc_c = np.argmax(lc, 2)
        lc_s = sieve(np.array(lc_c, dtype=rio.uint8), size=sieve_size)
        
    return np.array([get_one_hot_9(i) for i in lc_s.flatten()]).reshape((256, 256, 9))


for dataset in [ "val", "test", "train"]:
    root_dir = f"{grid_dir}/{dataset}/reduced_classes"
    new_dir = f"{grid_dir}/{dataset}/lc_sieve"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    files = os.listdir(root_dir)
    loop =  tqdm(range(len(files)))
    loop.set_description("dataset: " + dataset)
    for idx in loop:
        file = files[idx]
        fn = file.split(".")[0]
        
        new_classes = do_sieve(root_dir, file)
        np.savez_compressed(os.path.join(new_dir, fn + ".npz"), new_classes)




# Now there should be a grid_dir/train/rgb and grid_dir/train/lc_sieve which contain pairs