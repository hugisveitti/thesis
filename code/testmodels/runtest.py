import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

toTensor = T.ToTensor()
normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

from ..mixlc.generator import Generator as MixlcGenerator
from ..datautils import unprocess, create_img_from_classes, lc_labels_classes

def test(model_path, data_dir, start = 96, size = 64, idx=None):
    print("Hello from test")
    gen_type = "mix_lc"
  
    mixlc_generator = MixlcGenerator()
    mixlc_generator.load_state_dict(torch.load(model_path))
    d = "val"
    rgb_dir = os.path.join(data_dir, d, "rgb")
    lc_dir = os.path.join(data_dir, d, "lc_classes")
    files = os.listdir(rgb_dir)
    if not idx:
        idx = np.random.randint(len(files))
    print(f"index of rgb {idx}")
    rgb_fn = files[idx]
    fn = rgb_fn.split(".")[0]
    print(f"file name is {fn}")
    with Image.open(os.path.join(rgb_dir, rgb_fn)) as rgb:
        rgb = toTensor(rgb)[:3,:,:]
        rgb = normalize(rgb)
        rgb = rgb.type(torch.FloatTensor)
        rgb = rgb.reshape((1, 3, 256, 256))

    with np.load(os.path.join(lc_dir, fn + ".npz")) as classes:
        classes = classes["arr_0"]
        classes = torch.from_numpy(classes)
        classes = torch.movedim(classes, -1, 0)

    
 
    modify = True
    
    classes = classes.type(torch.FloatTensor)
    classes = classes.reshape((1, 14, 256, 256))
    possible_classes = np.arange(1, 14, dtype=np.uint8)
    
    all_pairs = []
    for c in possible_classes:
        new_classes = classes.clone()

        for i in range(start, start + size):
            for j in range(start, start + size):
                new_classes[:,:, i, j] = torch.zeros(14)
                new_classes[:, c, i, j] = torch.tensor(1).type(torch.FloatTensor)
    
        new_gen_rgb = mixlc_generator(rgb, new_classes)

        new_classes = unprocess(new_classes.detach(), False)
        new_classes = create_img_from_classes(new_classes)
        new_gen_rgb = unprocess(new_gen_rgb.detach())

        pair = [new_classes, new_gen_rgb]
        all_pairs.append(pair)


    gen_rgb = mixlc_generator(rgb, classes)
    gen_rgb = unprocess(gen_rgb.detach())


    classes = unprocess(classes.detach(), False)
    classes = create_img_from_classes(classes)

    rgb = unprocess(rgb.detach())

    all_pairs.append([classes, rgb])
    all_pairs.append([classes, gen_rgb])


    img_dir = "test_images"
    if not os.path.exists(img_dir):
            os.mkdir(img_dir)

    curr_img_dir = os.path.join(img_dir, fn + "_" + gen_type)
    if not os.path.exists(curr_img_dir):
        os.mkdir(curr_img_dir)

    for i, (rgb, classes) in enumerate(all_pairs):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        if i == len(all_pairs) - 2:
            ax[0].set_title("unchanged lc")
            ax[1].set_title("original rgb")
            name = "original"
        elif i == len(all_pairs) -1:
            ax[0].set_title("unchanged lc")
            ax[1].set_title("gen rgb")
            name = "gen_rgb"
        else:
            ax[0].set_title(f"class {lc_labels_classes[possible_classes[i]]}")
            name = f"class_{i}"
        ax[0].imshow(rgb)
        ax[0].axis("off")
        ax[1].imshow(classes)
        ax[1].axis("off")

       
        # plt.show()

        


        plt.savefig(f"{curr_img_dir}/{name}.png")


        