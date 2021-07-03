import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import os

sns.set_theme()

def plot_losses(dir_name, losses_file):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    with open(losses_file, "r") as f:
        losses = json.load(f)


    for key in losses.keys():
        if len(losses[key]) < 2:
            return
        plt.figure(figsize=(6,6))
        plt.plot(losses[key])
        plt.title(key)
        plt.savefig(f"{dir_name}/{key}.png")
        plt.close()

    def normalize(a):
        if np.min(a) == np.max(a):
            return np.min(a)
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
        return a


    plt.figure(figsize=(10,10))
    for key in losses.keys():
        s = "--" if key[0] == "d" else ""
        plt.plot(normalize(losses[key]), s, label=key)

    fr = plt.gca()
    fr.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.title("all losses normalized to be 0-1")
    plt.savefig(f"{dir_name}/losses_norm.png")
    plt.close() 

    # Not normalized

    plt.figure(figsize=(10,10))
    for key in losses.keys():
        s = "--" if key[0] == "d" else ""
        plt.plot(losses[key], s, label=key)

    plt.legend()
    plt.title("all losses")
    plt.savefig(f"{dir_name}/losses.png")
    plt.close() 


    plt.figure(figsize=(10,10))
    plt.plot(losses["g_loss"], label="generator loss")
    plt.plot(losses["d_loss"], label="discriminator loss")


    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Model Losses")
    plt.legend()
    plt.savefig(f"{dir_name}/model_losses.png")
    plt.close()