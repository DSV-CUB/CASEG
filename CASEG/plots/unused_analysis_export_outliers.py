import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from scipy import special
from datetime import datetime
import shutil


from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import dsc_boxplot, t1_bland_altman, t1_correlation, t1_equivalence, t1_omDSC_plot

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"

path_intra_1 = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_MRT 1_checked"
path_intra_2 = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_Intaobserver"

path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\OUTLIERS"

do_abs = True
do_intra1minus2 = True

# LOAD MODELS
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

# LOAD DATA

data_generator = mmsgenerators.unet.Setup(path_data, None, "TEST",  model_bb=model_bb, ws_dir=path_data, mask_mode="RASTERIZE")

predictions = []
expectations = []
predictions_values = []
expectations_values = []
data_dsc = []
data_error = []

for i in range(len(models)):

    data_generator.configuration.set(**models[i].configuration.__dict__)
    prediction, expectation = models[i].predict_generator(data_generator)

    prediction_value = []
    expectation_value = []
    data_dsc_model_case = []
    for k in range(len(data_generator.x)):
        prediction_value.append(np.mean(data_generator.x[k][np.where(prediction[k])]))
        expectation_value.append(np.mean(data_generator.x[k][np.where(expectation[k])]))
        data_dsc_model_case.append(tools.tool_general.get_metric_from_masks(prediction[k], expectation[k]) / 100)

    predictions.append(copy.copy(prediction))
    expectations.append(copy.copy(expectation))

    predictions_values.append(copy.copy(prediction_value))
    expectations_values.append(copy.copy(expectation_value))

    data_dsc.append(data_dsc_model_case)
    data_error.append(copy.copy(np.array(prediction_value)-np.array(expectation_value)))

data_dsc = np.array(data_dsc)
data_error = np.array(data_error)

if do_abs:
    data_error = np.abs(data_error)

# LOAD REFERENCE
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

    if do_intra1minus2:
        data_reference.append(np.mean(d1) - np.mean(d2))
    else:
        data_reference.append(np.mean(d2) - np.mean(d1))
    data_reference_dsc.append(tools.tool_general.get_metric_from_masks(gen_intra_1.y[match_ID1[i]], gen_intra_2.y[match_ID2[i]]) / 100)

data_reference = np.array(data_reference)
data_reference_dsc = np.array(data_reference_dsc)

if do_abs:
    data_reference = np.abs(data_reference)

tolerance = 0.95
reference_std_factor = np.sqrt(2) * special.erfinv(tolerance)
reference_mean = np.mean(data_reference)
reference_std = np.std(data_reference) * reference_std_factor
reference_value_margin = np.array([reference_mean-reference_std, reference_mean+reference_std])
reference_mean_dsc = np.mean(data_reference_dsc)
reference_std_dsc = np.std(data_reference_dsc) * reference_std_factor
reference_dsc_margin = np.array([reference_mean_dsc-reference_std_dsc, reference_mean_dsc+reference_std_dsc])

# EVALUATE OUTLIERS
error_outliers = ((data_error < reference_value_margin[0]) | (data_error > reference_value_margin[1]))
dsc_outliers = ((data_dsc < reference_dsc_margin[0]) | (data_dsc > reference_dsc_margin[1]))
both_outliers = (error_outliers & dsc_outliers)
any_outliers = (error_outliers | dsc_outliers)

any_error_outlier = error_outliers.any(axis=0)
any_dsc_outlier = dsc_outliers.any(axis=0)
any_both_outlier = both_outliers.any(axis=0)
any_any_outlier = any_outliers.any(axis=0)

path_out = os.path.join(path_out, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(os.path.join(path_out, "DICOMS"))
os.makedirs(os.path.join(path_out, "outlier_DSC"))
os.makedirs(os.path.join(path_out, "outlier_VALUE"))
os.makedirs(os.path.join(path_out, "outlier_both"))


for i in np.argwhere(any_outliers.any(axis=0)).flatten():
    os.makedirs(os.path.dirname(data_generator.information[i][0].replace(path_data, os.path.join(path_out, "DICOMS"))), exist_ok=True)
    shutil.copyfile(data_generator.information[i][0], data_generator.information[i][0].replace(path_data, os.path.join(path_out, "DICOMS")))
    shutil.copyfile(data_generator.information[i][0].replace(".dcm", ".pickle"), data_generator.information[i][0].replace(path_data, os.path.join(path_out, "DICOMS")).replace(".dcm", ".pickle"))

    masks = []
    _, drgba = tools.tool_plot.masks2delta2rgba(expectations[0][i], np.zeros(expectations[0][i].shape))
    masks.append(copy.copy(drgba))
    for k in range(len(models)):
        _, drgba = tools.tool_plot.masks2delta2rgba(expectations[k][i], predictions[k][i])
        masks.append(copy.copy(drgba))

    fig, axes = plt.subplots(1, (len(masks)+1), gridspec_kw={'width_ratios': [1] * (len(masks)+1)}, dpi=250, figsize=(10*(len(masks)+1),10))
    axes[0].imshow(data_generator.x[i], cmap="gray")
    axes[0].axis("off")
    axes[0].set_title(os.path.dirname(data_generator.information[i][0]).replace(path_data, "")[1:])
    tools.tool_plot.plot_masks(data_generator.x[i],masks,["reference"] + models_names,None,axes=axes[1:])

    for k in range(len(models)):
        text = "DSC = " + str(round(data_dsc[k][i],2))
        text = text + "\n\u0394T1 = " + str(round(data_error[k][i],2))

        if dsc_outliers[k][i]:
            text = text + "\nDSC out of range"
        else:
            text = text + "\n"

        if error_outliers[k][i]:
            text = text + "\nT1 error out of range"
        else:
            text = text + "\n"

        axes[2+k].text(0.05, 0.95, text, transform=axes[2+k].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()

    if any_dsc_outlier[i]:
        plt.savefig(os.path.join(path_out, "outlier_DSC", data_generator.information[i][1] + ".jpg"))

    if any_error_outlier[i]:
        plt.savefig(os.path.join(path_out, "outlier_VALUE", data_generator.information[i][1] + ".jpg"))

    if any_both_outlier[i]:
        plt.savefig(os.path.join(path_out, "outlier_both", data_generator.information[i][1] + ".jpg"))

a = 1

