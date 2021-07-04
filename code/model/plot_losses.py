import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import os



def plot_losses(dir_name, train_losses, val_losses):
    sns.set_theme()
    plt.grid(True)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)



    for key in train_losses.keys():
        if len(train_losses[key]) < 2:
            return
        plt.figure(figsize=(6,6))
        plt.plot(train_losses[key], label="train")
        plt.plot(val_losses[key], label="val")
        plt.title(key)
        plt.legend()
        plt.savefig(f"{dir_name}/{key}.png")
        plt.close()


    plt.figure(figsize=(10,10))
    for key in val_losses.keys():
        s = "--" if key[0] == "d" else ""
        plt.plot(val_losses[key], s, label=key)

    plt.legend()
    plt.title("all val losses")
    plt.savefig(f"{dir_name}/val_losses.png")
    plt.close() 

    plt.figure(figsize=(10,10))
    for key in train_losses.keys():
        s = "--" if key[0] == "d" else ""
        plt.plot(train_losses[key], s, label=key)

    plt.legend()
    plt.title("all train losses")
    plt.savefig(f"{dir_name}/train_losses.png")
    plt.close() 


    plt.figure(figsize=(10,10))
    plt.plot(val_losses["g_loss"], label="generator loss")
    plt.plot(val_losses["d_loss"], label="discriminator loss")


    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Val Model Losses")
    plt.legend()
    plt.savefig(f"{dir_name}/val_model_losses.png")
    plt.close()
    plt.axis("off")

    plt.figure(figsize=(10,10))
    plt.plot(train_losses["g_loss"], label="generator loss")
    plt.plot(train_losses["d_loss"], label="discriminator loss")


    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Train Model Losses")
    plt.legend()
    plt.savefig(f"{dir_name}/train_model_losses.png")
    plt.close()
    plt.grid(False)


def plot_ious(ious, dir_name):
    """
    ious is a dictionary where each value is a list
    """

    plt.figure(figsize=(10,10))
    for key in ious.keys():
        plt.plot(ious[key], label=key)
    
    plt.title("IoU")
    plt.legend()
    plt.savefig(os.path.join(dir_name,"ious.png"))
    plt.close()
    plt.grid(False)

