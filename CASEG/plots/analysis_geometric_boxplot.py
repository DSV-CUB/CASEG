import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib.ticker import MultipleLocator

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import metric_boxplot

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"
path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\geometric_boxplot_DSC_HD.jpg"


do_tight = True

models = []
models_names = []
model_bb = None
metrics = ["DSC", "HD"]

for root, _, files in os.walk(path_weights):
    for file in files:
        if file.endswith(".pickle"):
            model = mmsmodels.unet.Setup()
            try:
                model.load(os.path.join(root, file))
            except:
                continue

            if model.configuration.model_settings.model_predictor.upper() == "BB" or model.configuration.model_settings.model_predictor.upper() == "BOUNDINGBOX":
                model_bb = copy.copy(model)
            else:
                models.append(copy.copy(model))
                models_names.append(root.replace(path_weights, "")[1:root.replace(path_weights, "")[1:].find("\\")+1][1:])

data = mmsgenerators.unet.Setup(path_data, None, "TEST",  model_bb=model_bb, ws_dir=path_data, mask_mode="RASTERIZE")

predictions = []
expectations = []
predictions_values = []
expectations_values = []
for i in range(len(models)):
    data.configuration.set(**models[i].configuration.__dict__)
    prediction, expectation = models[i].predict_generator(data)
    predictions.append(copy.copy(prediction))
    expectations.append(copy.copy(expectation))

    prediction_value = []
    expectation_value = []
    for j in range(len(data.x)):
        prediction_value.append(np.mean(data.x[j][np.where(prediction[j])]))
        expectation_value.append(np.mean(data.x[j][np.where(expectation[j])]))
    predictions_values.append(copy.copy(prediction_value))
    expectations_values.append(copy.copy(expectation_value))



information = {"models_name": models_names,
               "data": data.x,
               "predictions": predictions,
               "expectations": expectations,
               "predictions_values": predictions_values,
               "expectations_values": expectations_values}


# RUN PLOT

cols = int(len(information["models_name"]))
rows = 2
fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols}, figsize=(cols*5, rows*5))

if rows == 1:
    axes = np.array([axes])

for i in range(2):
    metric_boxplot.plot(information, axes[i,:].flatten(), metric=metrics[i], voxel_sizes=data.pixel_spacing)

for i in range(len(information["models_name"])):
    axes[0,i].set_title(information["models_name"][i], fontdict=dict(fontsize=20), pad=20)

    axes[0, i].set_ylabel("DSC in %", fontsize=12)
    axes[0, i].set_ylim(0, 100)
    axes[0, i].set(adjustable='box', aspect=0.005)#'equal')
    axes[0, i].yaxis.set_minor_locator(MultipleLocator(10))
    axes[0, i].grid(True, axis="y", which="both", ls=":")

    axes[1, i].set_ylabel("HD in mm", fontsize=12)
    axes[1, i].set_ylim(0, 12.5)
    axes[1, i].set(adjustable='box', aspect=0.04)#'equal')
    axes[1, i].yaxis.set_minor_locator(MultipleLocator(1))
    axes[1, i].grid(True, axis="y", which="both", ls=":")

#axes[0,0].annotate("native", xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad -5, 0), xycoords=axes[0,0].yaxis.label, textcoords='offset points', fontsize=20, ha='right', va='center', rotation="vertical")
#axes[1,0].annotate("pca", xy=(0, 0.5), xytext=(-axes[1,0].yaxis.labelpad -5, 0), xycoords=axes[1,0].yaxis.label, textcoords='offset points', fontsize=20, ha='right', va='center', rotation="vertical")

if do_tight:
    plt.tight_layout()
if not path_out:
    plt.show()
else:
    plt.gcf().set_dpi(300)
    plt.savefig(path_out)