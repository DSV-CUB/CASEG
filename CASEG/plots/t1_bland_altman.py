import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def plot(data, axes=None, equivalence_limit=None):
    # create figure if no axes given
    if axes is None:
        cols = int(len(data["models_name"]))
        rows = 1
        fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

    lim_min_x = np.floor(np.min(np.mean(np.array([np.array(data["predictions_values"]), np.array(data["expectations_values"])]), axis=0))/100) * 100
    lim_max_x = np.ceil(np.max(np.mean(np.array([np.array(data["predictions_values"]), np.array(data["expectations_values"])]), axis=0))/100) * 100

    lim_min_y = np.floor(np.min((np.array([np.array(data["predictions_values"]) - np.array(data["expectations_values"])])))/10) * 10
    lim_max_y = np.ceil(np.max((np.array([np.array(data["predictions_values"]) - np.array(data["expectations_values"])])))/10) * 10

    for i in range(len(data["models_name"])):
        x = np.mean(np.array([np.array(data["predictions_values"][i]).flatten(), np.array(data["expectations_values"][i]).flatten()]), axis=0)
        y = np.array([np.array(data["predictions_values"][i]).flatten() - np.array(data["expectations_values"][i]).flatten()]).flatten()

        if equivalence_limit is None:
            axes[i].scatter(x, y, 2, zorder=10, c="C0")
        else:
            indeces = np.argwhere(np.abs(y)<=equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],2, c="C0", zorder=10)
            indeces = np.argwhere(np.abs(y)>equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],2,c="r", zorder=10)


        axes[i].barh(np.mean(y), lim_max_x-lim_min_x, 2 * 1.96 * np.std(y), lim_min_x, color="#fff1c780")
        axes[i].axhline(np.mean(y), color='C1', linestyle='-')
        axes[i].axhline(np.mean(y) + 1.96 * np.std(y), color='C1', linestyle='--')
        axes[i].axhline(np.mean(y) - 1.96 * np.std(y), color='C1', linestyle='--')

        axes[i].set_xlim(lim_min_x, lim_max_x)
        axes[i].set_ylim(lim_min_y, lim_max_y)
        axes[i].set_xlabel("Mean of prediction and expectation", fontsize=14)
        axes[i].set_ylabel("prediction - expectation", fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].xaxis.set_minor_locator(MultipleLocator(100))
        axes[i].yaxis.set_minor_locator(MultipleLocator(10))
        axes[i].grid(True, axis="both", which="both", ls=":")

        xvals,yvals = axes[i].axes.get_xlim(),axes[i].axes.get_ylim()
        xrange = xvals[1]-xvals[0]
        yrange = yvals[1]-yvals[0]
        axes[i].set_aspect(1*(xrange/yrange), adjustable='box')

        if axes is None:
            axes[i].title.set_text(information["models_name"][i])
        #axes[i].set(adjustable='box', aspect='equal')

    if axes is None:
        #plt.tight_layout()
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