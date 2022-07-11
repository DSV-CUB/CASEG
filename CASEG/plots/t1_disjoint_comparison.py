import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, equivalence_limit=None):
    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    min_val = np.inf
    max_val = 0
    # run plot
    for i in range(len(data["models_name"])):
        x = []
        y = []

        for j in range(len(data["expectations"][i])):
            delta, _ = tools.tool_plot.masks2delta2rgba(data["expectations"][i][j], data["predictions"][i][j])
            indeces_exp = np.argwhere(delta==1)
            indeces_pred = np.argwhere(delta==-1)

            if len(indeces_exp) == 0:
                mean_exp = 0
            else:
                mean_exp = np.mean(data["data"][j][indeces_exp[:,0], indeces_exp[:,1]])

            if len(indeces_pred) == 0:
                mean_pred = 0
            else:
                mean_pred = np.mean(data["data"][j][indeces_pred[:,0], indeces_pred[:,1]])

            x.append(mean_exp)
            y.append(mean_pred)

        min_val = np.min([min_val, np.min(x), np.min(y)])
        max_val = np.max([max_val, np.max(x), np.max(y)])

        x = np.array(x)
        y = np.array(y)

        if equivalence_limit is None:
            axes[i].scatter(x,y,1)
        else:
            diff = np.abs(np.array(data["predictions_values"][i])-np.array(data["expectations_values"][i]))
            indeces = np.argwhere(diff<=equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],1,c="C0", zorder=10)
            indeces = np.argwhere(diff>equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],1,c="r", zorder=10)

        axes[i].set_xlabel("mean T1 of segmented pixels\nsolely by expert in ms", fontsize=14)
        axes[i].set_ylabel("mean T1 of segmented pixels\nsolely by model in ms", fontsize=14)

        axes[i].set(adjustable='box', aspect='equal')
        #axes[i].xaxis.set_minor_locator(MultipleLocator(100))
        #axes[i].yaxis.set_minor_locator(MultipleLocator(100))
        #axes[i].grid(True, axis="both", which="both", ls=":")
        #if i == 0:
        if axes is None:
            axes[i].title.set_text(data["models_name"][i])

    lim_min = np.floor(min_val / 100) * 100
    lim_max = np.ceil(max_val / 100) * 100

    for i in range(len(data["models_name"])):
        axes[i].set_xlim(lim_min, lim_max)
        axes[i].set_ylim(lim_min, lim_max)
        axes[i].set(adjustable='box', aspect='equal')
        axes[i].xaxis.set_minor_locator(MultipleLocator(100))
        axes[i].yaxis.set_minor_locator(MultipleLocator(100))
        axes[i].grid(True, axis="both", which="both", ls=":")

    # show plot if axes is None
    if axes is None:
        #plt.tight_layout()
        plt.show()

    return

if __name__ == "__main__":
    pass

