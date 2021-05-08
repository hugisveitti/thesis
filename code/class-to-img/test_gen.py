import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from generator import Generator

from utils import create_img_from_classes


files_path = "../../data/val/"
files = os.listdir(os.path.join(files_path, "lc_classes"))


idx = 60
filename = files[idx]
file = os.path.join(files_path, "lc_classes", filename)

classes = np.load(file)["arr_0"]


classes_img = create_img_from_classes(classes)


fn = filename.split(".")[0] + ".png"
target_img = plt.imread(os.path.join(files_path, "rgb", fn))



generator = Generator(14,3)
generator.load_state_dict(torch.load("colab/models/generator29.pt",map_location=torch.device('cpu')))

classes_t = torch.tensor(classes)
classes_t = torch.movedim(classes_t, -1, 0).float()
classes_t = classes_t.reshape((1, 14, 256, 256))

gen_img = generator(classes_t)

gen_img = gen_img.detach().numpy()[0]
gen_img = np.moveaxis(gen_img, 0, -1)
gen_img = np.array((gen_img *0.5 + 0.5) * 255, dtype = np.uint8)


fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(classes_img)
ax[0].set_title("input")

ax[1].imshow(target_img)
ax[1].set_title("target")

ax[2].imshow(gen_img)
ax[2].set_title("generated")

plt.show()