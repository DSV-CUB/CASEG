import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, bounds=[None,None]):
    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    min_val = np.inf
    max_val = 0
    y_max = 0
    # run plot
    for i in range(len(data["models_name"])):
        x = []
        y = []
        data_exp = np.array([])
        data_pred = np.array([])

        for j in range(len(data["expectations"][i])):
            delta, _ = tools.tool_plot.masks2delta2rgba(data["expectations"][i][j], data["predictions"][i][j])
            indeces_exp = np.argwhere(delta==1)
            indeces_pred = np.argwhere(delta==-1)

            if len(indeces_exp) > 0:
                data_exp = np.concatenate((data_exp, data["data"][j][indeces_exp[:,0], indeces_exp[:,1]].flatten()), axis=0).flatten()

            if len(indeces_pred) > 0:
                data_pred = np.concatenate((data_pred, data["data"][j][indeces_pred[:,0], indeces_pred[:,1]].flatten()), axis=0).flatten()

        n_exp, _, _, = axes[i].hist(data_exp, bins=50, facecolor='b', alpha=0.5, label="expert", range=(data_exp.min() if bounds[0] is None else bounds[0], data_exp.max() if bounds[1] is None else bounds[1]))
        n_pred, _, _, = axes[i].hist(data_pred, bins=50, facecolor='r', alpha=0.5, label="model", range=(data_pred.min() if bounds[0] is None else bounds[0], data_pred.max() if bounds[1] is None else bounds[1]))

        min_val = np.min([min_val, np.min(data_exp), np.min(data_pred)]) if bounds[0] is None else bounds[0]
        max_val = np.max([max_val, np.max(data_exp), np.max(data_pred)]) if bounds[1] is None else bounds[1]
        y_max = np.max([y_max, np.max(n_exp), np.max(n_pred)])

        axes[i].set_xlabel("T1 time in ms of disjoint pixels", fontsize=14)
        axes[i].set_ylabel("absoulte frequency", fontsize=14)

        axes[i].legend(prop={'size': 12}, loc=1)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

        if axes is None:
            axes[i].title.set_text(data["models_name"][i])

    lim_min = np.floor(min_val / 100) * 100
    lim_max = np.ceil(max_val / 100) * 100
    y_max = np.ceil(y_max / 100) * 100

    for i in range(len(data["models_name"])):
        axes[i].set_xlim(lim_min, lim_max)
        axes[i].set_ylim(0, y_max)
        #axes[i].set(adjustable='box', aspect='equal')
        xvals,yvals = axes[i].axes.get_xlim(),axes[i].axes.get_ylim()
        xrange = xvals[1]-xvals[0]
        yrange = yvals[1]-yvals[0]
        axes[i].set_aspect(1*(xrange/yrange), adjustable='box')
    # show plot if axes is None
    if axes is None:
        #plt.tight_layout()
        plt.show()

    return

if __name__ == "__main__":
    pass

