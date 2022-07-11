import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, metric="DSC", **kwargs):
    # data preparation
    voxel_sizes = kwargs.get("voxel_sizes", None)

    data_plot = []
    for i in range(len(data["models_name"])):
        dscs = []
        for j in range(len(data["expectations"][i])):
            dscs.append(tools.tool_general.get_metric_from_masks(data["expectations"][i][j], data["predictions"][i][j], metric, voxel_sizes=(None if voxel_sizes is None else voxel_sizes[j])))
        data_plot.append(dscs)

    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    # do plots
    for i in range(len(data["models_name"])):
        axes[i].boxplot(data_plot[i], zorder=10, flierprops=dict(markerfacecolor='C0', markeredgecolor="C0", marker='.', markersize=5), boxprops=dict(color="C0"), capprops=dict(color="C0"), whiskerprops=dict(color="C0"))
        axes[i].set_xlim(0.75, 1.25)
        axes[i].set_xticks([0.75, 1, 1.25])
        #axes[i].set_xticklabels(["", data["models_name"][i], ""])
        axes[i].set_xticklabels(["", "", ""])
        plt.setp(axes[i].get_xticklabels()) #, rotation=90, fontsize='x-small')


        #if i == 0:


    # show figure if no axes given
    if axes is None:
        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"
    path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
    path_information = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\STATISTICS\models_SAXpaper_data_TEST_SAX.pickle"
    load = False

    if load:
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
                        models_names.append(copy.copy(model.configuration.name))

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

        with open(path_information, "wb") as f:
            pickle.dump(information, f)
            f.close()
    else:
        with open(path_information, "rb") as f:
            information = pickle.load(f)
            f.close()


    information["models_name"] = [name[11:].replace("_1.5", "") for name in information["models_name"]]








