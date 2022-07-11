import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib.ticker import MultipleLocator

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import t1_disjoint_histogram

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\quantitative_disjoint_histogram.jpg"

equivalence_margin=24.5
do_tight = True
do_abs = True

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

indeces_pca = np.argwhere(np.array(expectations_values[0]) < 800).flatten()
indeces_native = np.argwhere(np.array(expectations_values[0]) > 800).flatten()

data_pca = np.array(data.x, dtype=object)[indeces_pca]
data_native = np.array(data.x, dtype=object)[indeces_native]
expectations_pca = []
predictions_pca = []
expectations_native = []
predictions_native = []
expectations_values_pca = []
predictions_values_pca = []
expectations_values_native = []
predictions_values_native = []

for i in range(len(expectations_values)):
    expectations_pca.append(np.array(expectations[i], dtype=object)[indeces_pca])
    predictions_pca.append(np.array(predictions[i], dtype=object)[indeces_pca])
    expectations_native.append(np.array(expectations[i], dtype=object)[indeces_native])
    predictions_native.append(np.array(predictions[i], dtype=object)[indeces_native])

    expectations_values_pca.append(np.array(expectations_values[i])[indeces_pca])
    predictions_values_pca.append(np.array(predictions_values[i])[indeces_pca])
    expectations_values_native.append(np.array(expectations_values[i])[indeces_native])
    predictions_values_native.append(np.array(predictions_values[i])[indeces_native])

information = {"models_name": models_names,
               "data": data.x,
               "predictions": predictions,
               "expectations": expectations,
               "predictions_values": predictions_values,
               "expectations_values": expectations_values}

information_pca = {"models_name": models_names,
               "data": data_pca,
               "predictions": predictions_pca,
               "expectations": expectations_pca,
               "predictions_values": predictions_values_pca,
               "expectations_values": expectations_values_pca}

information_native = {"models_name": models_names,
               "data": data_native,
               "predictions": predictions_native,
               "expectations": expectations_native,
               "predictions_values": predictions_values_native,
               "expectations_values": expectations_values_native}




# RUN PLOT
# all together
#cols = int(len(information["models_name"]))
#rows = 1
#fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

#if rows == 1:
#    axes = np.array([axes])

#for i in range(1):
#    t1_disjoint_histogram.plot(information, axes[i,:].flatten())

#for i in range(len(information["models_name"])):
#    axes[0,i].set_title(information["models_name"][i], fontdict=dict(fontsize=20), pad=20)

#if do_tight:
#    plt.tight_layout()
#plt.show()

# separate
cols = int(len(information["models_name"]))
rows = 2
fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols}, figsize=(cols*5, rows*5))

t1_disjoint_histogram.plot(information_native, axes[0,:].flatten())
t1_disjoint_histogram.plot(information_pca, axes[1,:].flatten())

axes[0,0].annotate("native", xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad -5, 0), xycoords=axes[0,0].yaxis.label, textcoords='offset points', fontsize=20, ha='right', va='center', rotation="vertical")
axes[1,0].annotate("pca", xy=(0, 0.5), xytext=(-axes[1,0].yaxis.labelpad -5, 0), xycoords=axes[1,0].yaxis.label, textcoords='offset points', fontsize=20, ha='right', va='center', rotation="vertical")

for i in range(len(information["models_name"])):
    axes[0,i].set_title(information["models_name"][i], fontdict=dict(fontsize=20), pad=20)

if do_tight:
    plt.tight_layout()
if not path_out:
    plt.show()
else:
    plt.gcf().set_dpi(300)
    plt.savefig(path_out)