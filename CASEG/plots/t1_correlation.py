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

    # plot limits
    lim_min = np.floor(np.min([data["predictions_values"], data["expectations_values"]]) / 100) * 100
    lim_max = np.ceil(np.max([data["predictions_values"], data["expectations_values"]]) / 100) * 100

    # run plot
    for i in range(len(data["models_name"])):
        x = np.array(data["expectations_values"][i])
        y = np.array(data["predictions_values"][i])

        if equivalence_limit is None:
            axes[i].scatter(x,y,2)
        else:
            diff = np.abs(y-x)
            indeces = np.argwhere(diff<=equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],2,c="C0", zorder=10)
            indeces = np.argwhere(diff>equivalence_limit).flatten()
            axes[i].scatter(x[indeces],y[indeces],2,c="r", zorder=10)
            axes[i].plot(np.array([lim_min, lim_max]), equivalence_limit + np.array([lim_min, lim_max]), 'C1', lw=1, linestyle=":", label="equivalence margin")
            axes[i].plot(np.array([lim_min, lim_max]), - equivalence_limit + np.array([lim_min, lim_max]), 'C1', lw=1, linestyle=":")
            axes[i].fill_between(np.array([lim_min, lim_max]), equivalence_limit + np.array([lim_min, lim_max]), - equivalence_limit + np.array([lim_min, lim_max]), color="#fff1c780")




        kc, kp = stats.kendalltau(x,y)
        pc, pp = stats.pearsonr(x,y)

        #ts = stats.theilslopes(y,x)
        #axes[i].plot(np.array([lim_min, lim_max]), ts[1] + ts[0] * np.array([lim_min, lim_max]), 'white', label="Kendall \u03C4=" + str(np.round(c,2)) + " | p="+ str(np.round(p,3)))
        #axes[i].plot(np.array([lim_min, lim_max]), ts[1] + ts[0] * np.array([lim_min, lim_max]), 'r', label="Theil Slope\n" + str(np.round(ts[0],2)) + "\u22C5x+" + str(np.round(ts[1],2)))
        #axes[i].plot(x, ts[1] + ts[2] * x, 'r--')
        #axes[i].plot(x, ts[1] + ts[3] * x, 'r--')

        lr = stats.linregress(x, y)
        axes[i].plot(np.array([lim_min, lim_max]), lr[1] + lr[0] * np.array([lim_min, lim_max]), 'm', lw=1, label="Regression: y=" + '{:.2f}'.format(lr[0]) + "\u22C5x+" + '{:.2f}'.format(lr[1]) + "\nPearson: r=" + '{:.2f}'.format(pc) + " | p=" + '{:.3f}'.format(pp) + "\nKendall: \u03C4=" + '{:.2f}'.format(kc) + " | p=" + '{:.3f}'.format(kp))


        axes[i].set_xlim(lim_min, lim_max)
        axes[i].set_ylim(lim_min, lim_max)
        axes[i].set_xlabel("expected T1", fontsize=14)
        axes[i].legend(prop={'size': 12}, loc=2)
        axes[i].set(adjustable='box', aspect='equal')
        axes[i].xaxis.set_minor_locator(MultipleLocator(100))
        axes[i].yaxis.set_minor_locator(MultipleLocator(100))
        axes[i].grid(True, axis="both", which="both", ls=":")
        #if i == 0:
        axes[i].set_ylabel("predicted T1", fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        if axes is None:
            axes[i].title.set_text(data["models_name"][i])

    # show plot if axes is None
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

