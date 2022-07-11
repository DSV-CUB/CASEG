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
metrics = ["DSC", "HD"]

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
# RUN PLOT
# all together
cols = 3
rows = 3.75

fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True)
subfigs = fig.subfigures(2, 1)#, width_ratios = [0.05, 1, 1, 0.05], height_ratios = [1, 1, 1])
#subfigs[0,1].suptitle("native", fontsize=24, fontweight='bold')
#subfigs[0,2].suptitle("post contrast agent", fontsize=24, fontweight='bold')


#axes = None
#for i in range(len(information["models_name"])):
#    subfigs[i,0].supylabel(information["models_name"][i], fontsize=24, fontweight="bold")
#    subfigs[i,3].supylabel(" ", fontsize=24, fontweight="bold")
    #subfigs[i,1].supylabel("\n" + information["models_name"][i], fontsize=24, fontweight="bold", x=0.98)
#
#    axes_sf0 = subfigs[i,1].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1] * 1})
#    axes_sf1 = subfigs[i,2].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1] * 1})
#    if axes is None:
#        axes = np.hstack((axes_sf0, axes_sf1))
#    else:
#        axes = np.vstack((axes, np.hstack((axes_sf0, axes_sf1))))

#separating line


axes_sf0 = subfigs[0].subplots(2,3, gridspec_kw={'width_ratios': [1] * 3, 'height_ratios': [1] * 2})
axes_sf1 = subfigs[1].subplots(2,3, gridspec_kw={'width_ratios': [1] * 3, 'height_ratios': [1] * 2})
axes = np.vstack((axes_sf0, axes_sf1))


subfigs[0].supylabel("native", fontsize=24, fontweight="bold")
subfigs[1].supylabel("post contrast agent", fontsize=24, fontweight="bold")
axes[0,0].set_title("refU", fontsize=24, fontweight="bold")
axes[0,1].set_title("cropU", fontsize=24, fontweight="bold")
axes[0,2].set_title("crinU", fontsize=24, fontweight="bold")
axes[2,0].set_title("refU", fontsize=24, fontweight="bold")
axes[2,1].set_title("cropU", fontsize=24, fontweight="bold")
axes[2,2].set_title("crinU", fontsize=24, fontweight="bold")
plt.plot([0, 1], [0.5, 0.5], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)




for i in range(2):
    metric_boxplot.plot(information_native, axes[i,:].flatten(), metric=metrics[i], voxel_sizes=np.array(data.pixel_spacing)[indeces_native])
    metric_boxplot.plot(information_pca, axes[i+2,:].flatten(), metric=metrics[i], voxel_sizes=np.array(data.pixel_spacing)[indeces_pca])

for i in range(len(information["models_name"])):
    a = 0
    b = i
    axes[a, b].set_ylim(0, 100)
    axes[a, b].set(adjustable='box', aspect=0.005)#'equal')
    axes[a, b].yaxis.set_minor_locator(MultipleLocator(10))
    axes[a, b].grid(True, axis="y", which="both", ls=":")
    axes[a, b].set_ylabel("DSC in %", fontsize=16)
    axes[a, b].tick_params(axis='both', which='major', labelsize=14)

    a = 1
    b = i
    axes[a, b].set_ylim(0, 12.5)
    axes[a, b].set(adjustable='box', aspect=0.04)#'equal')
    axes[a, b].yaxis.set_minor_locator(MultipleLocator(1))
    axes[a, b].grid(True, axis="y", which="both", ls=":")
    axes[a, b].set_ylabel("HD in mm", fontsize=16)
    axes[a, b].tick_params(axis='both', which='major', labelsize=14)

    a = 2
    b = i
    axes[a, b].set_ylim(0, 100)
    axes[a, b].set(adjustable='box', aspect=0.005)#'equal')
    axes[a, b].yaxis.set_minor_locator(MultipleLocator(10))
    axes[a, b].grid(True, axis="y", which="both", ls=":")
    axes[a, b].set_ylabel("DSC in %", fontsize=16)
    axes[a, b].tick_params(axis='both', which='major', labelsize=14)
    #axes[a, b].yaxis.set_label_position("right")
    #axes[a, b].yaxis.tick_right()

    a = 3
    b = i
    axes[a, b].set_ylim(0, 12.5)
    axes[a, b].set(adjustable='box', aspect=0.04)#'equal')
    axes[a, b].yaxis.set_minor_locator(MultipleLocator(1))
    axes[a, b].grid(True, axis="y", which="both", ls=":")
    axes[a, b].set_ylabel("HD in mm", fontsize=16)
    axes[a, b].tick_params(axis='both', which='major', labelsize=14)
    #axes[a, b].yaxis.set_label_position("right")
    #axes[a, b].yaxis.tick_right()

    #axes[i,0].annotate(information["models_name"], xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad -5, 0), xycoords=axes[i,0].yaxis.label, textcoords='offset points', fontsize=20, ha='right', va='center', rotation="vertical")

if not path_out:
    plt.show()
else:
    #plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.savefig(path_out)