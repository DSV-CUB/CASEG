import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, metric="DSC", pixel_spacing=None, bins=20):
    # data preparation
    data_plot = []
    for i in range(len(data["models_name"])):
        metrics = []
        for j in range(len(data["expectations"][i])):
            try:
                metrics.append(tools.tool_general.get_metric_from_masks(data["expectations"][i][j], data["predictions"][i][j],metric=metric, voxel_sizes=pixel_spacing[j], values=data["data"][j]))
            except:
                metrics.append(tools.tool_general.get_metric_from_masks(data["expectations"][i][j], data["predictions"][i][j],metric=metric))
        data_plot.append(metrics)

    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    # do plots
    for i in range(len(data["models_name"])):
        axes[i].hist(data_plot[i], bins=bins, zorder=10, color="C0")
        # axes[i].set(adjustable='box', aspect=0.005)#'equal')
        #if i == 0:
        axes[i].set_ylabel("Abs freq")
        axes[i].set_xlabel(metric)

    # show figure if no axes given
    if axes is None:
        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

    path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST"

    do_tight = True
    do_title = True
    run_metrics = ["DSC", "ASD", "HD", "AE", "SE"]
    bins = 50


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

    data = mmsgenerators.unet.Setup(path_data, None, "TEST",  model_bb=model_bb, ws_dir=path_data, mask_mode="RASTERIZE")

    pixel_spacing = []
    for i in range(len(data.x)):
        pixel_spacing.append(np.array(data.pixel_spacing[i]))


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
    rows = int(len(run_metrics))
    fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})
    if rows == 1:
        axes = np.array([axes])

    for i in range(len(run_metrics)):
        plot(information,axes[i,:].flatten(), metric=run_metrics[i],pixel_spacing=pixel_spacing,bins=bins)

    if do_title:
        for i in range(len(information["models_name"])):
            axes[0,i].set_title(information["models_name"][i], fontdict=dict(fontsize=20), pad=20)

    if do_tight:
        plt.tight_layout()
    plt.show()