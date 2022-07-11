import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats, special

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, equivalence_limit=None, **kwargs):

    do_abs = kwargs.get("abs", False)


    # data preparation
    data_x = []
    for i in range(len(data["models_name"])):
        omdscs = []
        for j in range(len(data["expectations"][i])):
            omdscs.append(1 - tools.tool_general.get_metric_from_masks(data["expectations"][i][j], data["predictions"][i][j]) / 100)
        data_x.append(copy.copy(omdscs))

    data_y = []
    max_y = 0
    for i in range(len(data["models_name"])):
        if do_abs:
            error = np.abs(np.array(data["predictions_values"][i]) - np.array(data["expectations_values"][i]))
        else:
            error = np.array(data["predictions_values"][i]) - np.array(data["expectations_values"][i])

        max_y = np.ceil(np.max([max_y, np.max(np.abs(error))]) / 10) * 10
        data_y.append(copy.copy(error))

    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    # do plots
    for i in range(len(data["models_name"])):

        if equivalence_limit is None:
            axes[i].scatter(data_x[i], data_y[i],2, zorder=10)
        else:
            x = np.array(data_x[i])
            y = np.abs(np.array(data_y[i]))

            indeces = np.argwhere(y<=equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],2,c="C0", zorder=10)
            indeces = np.argwhere(y>equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],2,c="r", zorder=10)

            axes[i].plot(np.array([0, 1]), np.array([equivalence_limit, equivalence_limit]), 'C1', lw=1, linestyle=":", label="equivalence margin")
            axes[i].plot(np.array([0, 1]), np.array([-equivalence_limit, -equivalence_limit]), 'C1', lw=1, linestyle=":")
            axes[i].fill_between(np.array([0, 1]), np.array([equivalence_limit, equivalence_limit]), np.array([-equivalence_limit, -equivalence_limit]), color="#fff1c780")

        kc, kp = stats.kendalltau(data_x[i], data_y[i])
        pc, pp = stats.pearsonr(data_x[i], data_y[i])

        lr = stats.linregress(data_x[i], data_y[i])
        axes[i].plot(np.array([0, 1]), lr[1] + lr[0] * np.array([0, 1]), 'm', lw=1, label="Regression y=" + '{:.2f}'.format(lr[0]) + "\u22C5x+" + '{:.2f}'.format(lr[1]) + "\nPearson r=" + '{:.2f}'.format(pc) + " | p=" + '{:.3f}'.format(pp) + "\nKendall \u03C4=" + '{:.2f}'.format(kc) + " | p=" + '{:.3f}'.format(kp))

        if do_abs:
            axes[i].set_ylim(0, max_y)
            axes[i].set_ylabel("absolute T1 error in ms", fontsize=14)
        else:
            axes[i].set_ylim(-max_y, max_y)
            axes[i].set_ylabel("T1 error in ms", fontsize=14)

        axes[i].yaxis.set_minor_locator(MultipleLocator(10))

        axes[i].set_xlim(0, 1)
        axes[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        axes[i].set_xlabel("1 - DSC", fontsize=14)

        axes[i].legend(prop={'size': 12}, loc=1)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

        #axes[i].legend()
        axes[i].grid(True, axis="x", which="both", ls=":")
        axes[i].grid(True, axis="y", which="both", ls=":")
        xvals,yvals = axes[i].axes.get_xlim(),axes[i].axes.get_ylim()
        xrange = xvals[1]-xvals[0]
        yrange = yvals[1]-yvals[0]
        axes[i].set_aspect(1*(xrange/yrange), adjustable='box')



    # show figure if no axes given
    if axes is None:
        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    pass
