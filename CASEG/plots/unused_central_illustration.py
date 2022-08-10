import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import dsc_boxplot, t1_bland_altman, t1_correlation, t1_equivalence, t1_omDSC_plot

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV_Trunc"
path_information = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\STATISTICS\models_SAX_data_MV_TRUNC.pickle"

path_intra_1 = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_MRT 1_checked"
path_intra_2 = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_Intaobserver"
path_intra_pickle = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\STATISTICS\equivalence.pickle"

load = False
load_equivalence = True
do_plots = ["dsc_boxplot", "t1_equivalence", "t1_omDSC_plot"]

do_tight = True
do_order = "row" # ROW or COL
do_title = True

# TEST DATA LOAD
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

    with open(path_information, "wb") as f:
        pickle.dump(information, f)
        f.close()
else:
    with open(path_information, "rb") as f:
        information = pickle.load(f)
        f.close()


# REFERENCE EQUIVALENCE MARGIN DATA
data_reference = []
data_reference_dsc = []
if load_equivalence:
    gen_intra_1 = mmsgenerators.unet.Setup(path_intra_1, None, "TEST", ws_dir=path_intra_1, mask_mode="RASTERIZE")
    gen_intra_2 = mmsgenerators.unet.Setup(path_intra_2, None, "TEST", ws_dir=path_intra_2, mask_mode="RASTERIZE")

    info1 = np.array(gen_intra_1.information)[:, 1]
    info2 = np.array(gen_intra_2.information)[:, 1]

    matches, match_ID1, match_ID2 = np.intersect1d(info1, info2, False, True)

    for i in range(len(match_ID1)):
        d1 = gen_intra_1.x[match_ID1[i]]
        d2 = gen_intra_2.x[match_ID2[i]]
        c1 = gen_intra_1.y_contour[match_ID1[i]]
        c2 = gen_intra_2.y_contour[match_ID2[i]]
        m1 = tools.tool_general.contour2mask(c1, d1, False, "RASTERIZE")
        m2 = tools.tool_general.contour2mask(c2, d2, False, "RASTERIZE")
        d1 = d1[np.argwhere(m1)[:, 0], np.argwhere(m1)[:, 1]]
        d2 = d2[np.argwhere(m2)[:, 0], np.argwhere(m2)[:, 1]]
        data_reference.append(np.mean(d2) - np.mean(d1))
        data_reference_dsc.append( tools.tool_general.get_metric_from_masks(gen_intra_1.y[match_ID1[i]], gen_intra_2.y[match_ID2[i]]))
    data_reference = np.array(data_reference)
    data_reference_dsc = np.array(data_reference_dsc)

    with open(path_intra_pickle, "wb") as f:
        pickle.dump(data_reference, f)
        f.close()
else:
    with open(path_intra_pickle, "rb") as f:
        data_reference = pickle.load(f)
        f.close()
    data_reference_dsc = None

information["equivalence"] = data_reference
information["equivalence_dsc"] = data_reference_dsc



# RUN PLOT
if do_order.upper() == "ROW":
    cols = int(len(information["models_name"]))
    rows = int(len(do_plots))
    fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})
else:
    rows = int(len(information["models_name"]))
    cols = int(len(do_plots))
    fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

if rows == 1:
    axes = np.array([axes])

for i in range(len(do_plots)):
    if do_order.upper() == "ROW":
        eval(do_plots[i] + ".plot(information, axes[i,:].flatten())")
    else:
        eval(do_plots[i] + ".plot(information, axes[:, i].flatten())")

if do_title:
    if do_order.upper() == "ROW":
        for i in range(len(information["models_name"])):
            axes[0,i].set_title(information["models_name"][i], fontdict=dict(fontsize=20), pad=20)
    else:
        for i in range(len(do_plots)):
            axes[0,i].set_title(do_plots[i], fontdict=dict(fontsize=20), pad=20)

if do_tight:
    plt.tight_layout()
plt.show()