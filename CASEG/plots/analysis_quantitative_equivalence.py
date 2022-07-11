import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib.ticker import MultipleLocator

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import t1_equivalence

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\quantitative_equivalence.jpg"

equivalence_limits=24.5

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

expectations_values_pca = []
predictions_values_pca = []
expectations_values_native = []
predictions_values_native = []

for i in range(len(expectations_values)):
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
               "data": data.x,
               "predictions": predictions,
               "expectations": expectations,
               "predictions_values": predictions_values_pca,
               "expectations_values": expectations_values_pca}

information_native = {"models_name": models_names,
               "data": data.x,
               "predictions": predictions,
               "expectations": expectations,
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
#    t1_equivalence.plot(information, axes[i,:].flatten(), limits=equivalence_limits)

#for i in range(len(information["models_name"])):
#    axes[0,i].set_title(information["models_name"][i], fontdict=dict(fontsize=20), pad=20)

#if do_tight:
#    plt.tight_layout()
#plt.show()

# separate
cols = 3
rows = 1.75

fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True, dpi=300)
subfigs = fig.subfigures(2, 1)#, width_ratios = [0.05, 1, 1, 0.05], height_ratios = [1, 1, 1])


axes_sf0 = subfigs[0].subplots(1,3, gridspec_kw={'width_ratios': [1] * 3, 'height_ratios': [1] * 1})
axes_sf1 = subfigs[1].subplots(1,3, gridspec_kw={'width_ratios': [1] * 3, 'height_ratios': [1] * 1})
axes = np.vstack((axes_sf0, axes_sf1))


subfigs[0].supylabel("native", fontsize=24, fontweight="bold")
subfigs[1].supylabel("post contrast agent", fontsize=24, fontweight="bold")
axes[0,0].set_title("refU", fontsize=24, fontweight="bold")
axes[0,1].set_title("cropU", fontsize=24, fontweight="bold")
axes[0,2].set_title("crinU", fontsize=24, fontweight="bold")
axes[1,0].set_title("refU", fontsize=24, fontweight="bold")
axes[1,1].set_title("cropU", fontsize=24, fontweight="bold")
axes[1,2].set_title("crinU", fontsize=24, fontweight="bold")
plt.plot([0, 1], [0.5, 0.5], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)



t1_equivalence.plot(information_native, axes[0,:].flatten(), limits=equivalence_limits)
t1_equivalence.plot(information_pca, axes[1,:].flatten(), limits=equivalence_limits)


if not path_out:
    plt.show()
else:
    plt.savefig(path_out)