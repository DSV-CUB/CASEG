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

path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"

path_intra_1 = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_MRT 1_checked"
path_intra_2 = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_Intaobserver"

do_abs = False

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
predictions_values = []
expectations_values = []

for i in range(len(models)):
    predictions_model = []
    expectations_model = []
    predictions_values_model = []
    expectations_values_model = []


    for j in range(len(data_generators)):
        data_generators[j].configuration.set(**models[i].configuration.__dict__)
        prediction, expectation = models[i].predict_generator(data_generators[j])

        prediction_value = []
        expectation_value = []
        for k in range(len(data_generators[j].x)):
            prediction_value.append(np.mean(data_generators[j].x[k][np.where(prediction[k])]))
            expectation_value.append(np.mean(data_generators[j].x[k][np.where(expectation[k])]))

        predictions_model.append(copy.copy(prediction))
        expectations_model.append(copy.copy(expectation))

        predictions_values_model.append(copy.copy(prediction_value))
        expectations_values_model.append(copy.copy(expectation_value))

    predictions.append(copy.copy(predictions_model))
    expectations.append(copy.copy(expectations_model))
    predictions_values.append(copy.copy(predictions_values_model))
    expectations_values.append(copy.copy(expectations_values_model))

data_reference = []
data_reference_dsc = []

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



# RUN PLOT

rows = int(len(models_names))
cols = int(len(sub_dirs))
fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols})

if rows == 1:
    axes = np.array([axes])

for i in range(len(models_names)):
    information = {"models_name": sub_dirs,
                   "predictions": predictions[i],
                   "expectations": expectations[i],
                   "predictions_values": predictions_values[i],
                   "expectations_values": expectations_values[i],
                   "equivalence": data_reference,
                   "equivalence_dsc": data_reference_dsc}

    t1_omDSC_plot.plot(information, axes[i,:].flatten(), abs=do_abs)

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