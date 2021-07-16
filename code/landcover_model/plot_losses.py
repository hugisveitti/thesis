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


   
