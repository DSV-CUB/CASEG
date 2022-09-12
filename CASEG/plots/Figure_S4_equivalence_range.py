import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import special

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, equivalence_margin=24.5, **kwargs):
    # data preparation
    alpha = kwargs.get("alpha", 0.05)
    bc = kwargs.get("bc", True) # bonferroni correction

    data_plot = []
    for i in range(len(data["models_name"])):
        data_model = []
        for j in range(len(data["expectations"][i])):
            data_model.append(tools.tool_general.get_metric_from_masks(data["predictions"][i][j], data["expectations"][i][j], "E", values=data["data"][j]))
        data_plot.append(data_model)

    if bc:
        alpha_c = alpha / len(data_plot)# bonferoni corrected alpha
    else:
        alpha_c = copy.copy(alpha)


    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    # do plots
    for i in range(len(data["models_name"])):
        data_mean = np.mean(data_plot[i])
        data_std = np.std(data_plot[i])
        CI = (1 - alpha_c)
        factor_std = np.sqrt(2) * special.erfinv(CI)

        limits = np.sort(np.array([data_mean - factor_std * data_std / np.sqrt(len(data_plot[i])), data_mean + factor_std * data_std / np.sqrt(len(data_plot[i]))]))
        print(limits)
        plt.plot()

        axes[i].axhline(equivalence_margin, c="C1", lw=1, linestyle=":")
        axes[i].axhline(-equivalence_margin, c="C1", lw=1, linestyle=":")
        axes[i].axhline(0, c="C1", lw=1, linestyle="-")
        axes[i].fill_between(np.array([0.75, 1.25]), [equivalence_margin, equivalence_margin], [-equivalence_margin, -equivalence_margin], color="#fff1c780")
        axes[i].plot([1, 1], limits, zorder=10, c="C0")
        axes[i].scatter([1, 1], limits, zorder=10, c="C0", marker="s")

        axes[i].set_xlim(1.5*equivalence_margin, -1.5*equivalence_margin)
        axes[i].set_xlim(0.75, 1.25)
        axes[i].set_xticks([0.75, 1, 1.25])
        axes[i].set_xticklabels(["", "", ""])
        plt.setp(axes[i].get_xticklabels())



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








