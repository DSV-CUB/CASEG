import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy import special

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, **kwargs):

    tolerance = kwargs.get("tolerance", 0.95)
    alpha = kwargs.get("alpha", 0.05)
    bonferoni = kwargs.get("bonferoni", True)
    nobias = kwargs.get("nobias", True)
    limits = kwargs.get("limits", None)

    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    if limits is None:
        reference_std_factor = np.sqrt(2) * special.erfinv(tolerance)
        reference_mean = np.mean(data["equivalence"])
        reference_std = np.std(data["equivalence"]) * reference_std_factor
    else:
        reference_mean = 0
        reference_std = limits

    if bonferoni:
        alpha_c = alpha / len(data)# bonferoni corrected alpha
    else:
        alpha_c = copy.copy(alpha)

    CI = (1 - 2*alpha_c)
    test_std_factor = np.sqrt(2) * special.erfinv(CI)

    max_value = reference_std
    for i in range(len(data["models_name"])):
        diff = np.array(data["predictions_values"][i]).flatten() - np.array(data["expectations_values"][i]).flatten()
        test_mean = (np.mean(diff) - reference_mean)
        test_std = (test_std_factor * np.std(diff))
        max_value = int(np.max(np.array([max_value, abs(test_mean+test_std), abs(test_mean-test_std)])))
    max_value = (np.ceil(max_value/10)+1)*10

    for i in range(len(data["models_name"])):
        diff = np.array(data["predictions_values"][i]).flatten() - np.array(data["expectations_values"][i]).flatten()
        test_mean = (np.mean(diff) - reference_mean)
        test_std = (test_std_factor * np.std(diff))

        axes[i].barh(-reference_std, 2, 2*reference_std, align="edge", color="#fff1c780")
        axes[i].axhline(0, 0, 2, color="C1", linestyle=":")
        axes[i].axhline(reference_std, 0, 2, color="C1", linestyle="--")
        axes[i].axhline(-reference_std, 0, 2, color="C1", linestyle="--")

        if nobias:
            axes[i].errorbar(1, 0, yerr=test_std, capsize=2, color="C0", zorder=10)
            axes[i].scatter(1, 0, color="C0", zorder=10)
        else:
            axes[i].errorbar(1, test_mean, yerr=test_std, capsize=2, color="C0", zorder=10)
            axes[i].scatter(1, test_mean, color="C0", zorder=10)

        print(data["models_name"][i] + " bias corrected residuals: +/-" + str(test_std))

        axes[i].set_xlim(0, 2)
        axes[i].set_xticks([0, 1, 2])

        axes[i].set_ylim(-max_value, max_value)
        xvals,yvals = axes[i].axes.get_xlim(),axes[i].axes.get_ylim()
        xrange = xvals[1]-xvals[0]
        yrange = yvals[1]-yvals[0]
        axes[i].set_aspect(1*(xrange/yrange), adjustable='box')
        #axes[i].set_xticklabels(["", data["models_name"][i], ""])
        axes[i].set_xticklabels(["", "", ""])
        axes[i].yaxis.set_minor_locator(MultipleLocator(10))
        axes[i].grid(True, axis="y", which="both", ls=":")
        #axes[i].setp(axes[i].get_xticklabels(), rotation=90, fontsize='x-small')

        #if i == 0:
        axes[i].set_ylabel("error in ms", fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

    if axes is None:
        plt.tight_layout()
        plt.show()

    return