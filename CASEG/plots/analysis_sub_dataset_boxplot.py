import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import dsc_boxplot, t1_bland_altman, t1_correlation, t1_equivalence

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST"

load = False
load_equivalence = False
do_plots = ["dsc_boxplot", "t1_equivalence"]

do_tight = False
do_title = True

# TEST DATA LOAD
models = []
models_names = []
model_bb = None

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

data_generators = []
sub_dirs = os.listdir(path_data)
for i in range(len(sub_dirs)):
    data_generators.append(mmsgenerators.unet.Setup(os.path.join(path_data, sub_dirs[i]), None, "TEST",  model_bb=model_bb, ws_dir=os.path.join(path_data, sub_dirs[i]), mask_mode="RASTERIZE"))

predictions = []
expectations = []
for i in range(len(models)):
    predictions_model = []
    expectations_model = []

    for j in range(len(data_generators)):
        data_generators[j].configuration.set(**models[i].configuration.__dict__)
        prediction, expectation = models[i].predict_generator(data_generators[j])

        predictions_model.append(copy.copy(prediction))
        expectations_model.append(copy.copy(expectation))

    predictions.append(copy.copy(predictions_model))
    expectations.append(copy.copy(expectations_model))





# RUN PLOT

rows = int(len(models_names))
cols = int(len(sub_dirs))
fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

if rows == 1:
    axes = np.array([axes])

for i in range(len(models_names)):
    information = {"models_name": sub_dirs,
                   "predictions": predictions[i],
                   "expectations": expectations[i]}

    dsc_boxplot.plot(information, axes[i,:].flatten())

if do_title:
    for i in range(len(sub_dirs)):
        axes[0,i].set_title(sub_dirs[i], fontdict=dict(fontsize=10), pad=20)

    for i in range(len(models_names)):
        pad = 5
        axes[i,0].annotate(models_names[i], xy=(0, 0.5), xytext=(-axes[i,0].yaxis.labelpad - pad, 0), xycoords=axes[i,0].yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

if do_tight:
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.95)
plt.show()