import os
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")


topdir = "data"
reward = "r3"
cost   = "c2"
date   = "2020-02-04"

palettes = [sns.color_palette("Blues", 10), sns.color_palette("Reds", 10)]


fig, axes = plt.subplots(1, 3, figsize=(10,3))

for i, dim in enumerate([5, 10, 20]):
    directory = f"{topdir}/{reward}{cost}-{dim}x{dim}-{date[5:]}"
    cindex = [0, 0]
    for subdir in os.listdir(directory):
        j = 0 if subdir.startswith("L") else 1
        ratios = np.load(
            directory + "/" + subdir + "/ratios.npy"
        )
        axes[i].plot(ratios, color=palettes[j][cindex[j]])
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("Ratio")
        axes[i].set_title(f"{dim}x{dim}")
        cindex[j] = (cindex[j] + 1) % 10
        sns.despine(ax=axes[i])

fig.suptitle("Experiment 1")

fig.savefig("plots/exp1.eps", bbox_inches="tight")
    
