import os
import pickle
import copy
import numpy as np
import skimage.measure
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

def image_entropy(image):
    flatimage = image.flatten()
    unique_val, unique_occur = np.unique(flatimage, return_counts=True)
    pg = unique_occur / len(flatimage)
    calc = pg * np.log(1/pg) / np.log(2)
    return np.sum(calc)


def get_relevant_portion(mask):
    return np.sum(mask) / len(mask.flatten())

def get_data_info(data, mode="pred"):
    entropy_raw = []
    entropy_crop = []
    entropy_crin = []
    relevant_raw = []
    relevant_crop = []
    relevant_crin = []

    for i in range(len(data.x)):
        if mode == "pred":
            indeces = np.argwhere(np.squeeze(data.x_bb[i]))
            index_x_min = np.min(indeces[:,0])
            index_x_max = np.max(indeces[:,0])
            index_y_min = np.min(indeces[:,1])
            index_y_max = np.max(indeces[:,1])

            x_diff = index_x_max - index_x_min
            y_diff = index_y_max - index_y_min

            extend_factor = (float(1.5) - 1) / 2

            index_x_min = max(0, index_x_min - int(x_diff * extend_factor))
            index_x_max = min(np.shape(data.x_bb[i])[0], index_x_max + int(x_diff * extend_factor))
            index_y_min = max(0, index_y_min - int(y_diff * extend_factor))
            index_y_max = min(np.shape(data.x_bb[i])[1], index_y_max + int(y_diff * extend_factor))

        else:
            indeces = np.argwhere(np.squeeze(data.y[i]))
            index_x_min = np.min(indeces[:,0])
            index_x_max = np.max(indeces[:,0])
            index_y_min = np.min(indeces[:,1])
            index_y_max = np.max(indeces[:,1])

        entropy_raw.append(skimage.measure.shannon_entropy(data.x[i]))
        entropy_crop.append(skimage.measure.shannon_entropy((data.x[i][index_x_min:index_x_max, index_y_min:index_y_max])))
        entropy_crin.append(skimage.measure.shannon_entropy((np.dstack((data.x[i][index_x_min:index_x_max, index_y_min:index_y_max], data.x_bb[i][index_x_min:index_x_max, index_y_min:index_y_max])))))

        relevant_raw.append(get_relevant_portion(data.y[i]))
        relevant_crop.append(get_relevant_portion(data.y[i][index_x_min:index_x_max, index_y_min:index_y_max]))
        relevant_crin.append(get_relevant_portion(data.y[i][index_x_min:index_x_max, index_y_min:index_y_max]))

    entropy_raw = np.array(entropy_raw)
    entropy_crop = np.array(entropy_crop)
    entropy_crin = np.array(entropy_crin)
    relevant_raw = np.array(relevant_raw) * 100
    relevant_crop = np.array(relevant_crop) * 100
    relevant_crin = np.array(relevant_crin) * 100

    result = {"entropy_raw": entropy_raw,
              "entropy_crop": entropy_crop,
              "entropy_crin": entropy_crin,
              "relevant_raw": relevant_raw,
              "relevant_crop": relevant_crop,
              "relevant_crin": relevant_crin,
              }

    return result

def boxplot(ax, data):
    ax.boxplot(data, zorder=10, flierprops=dict(markerfacecolor='C0', markeredgecolor="C0", marker='.', markersize=5), boxprops=dict(color="C0"), capprops=dict(color="C0"), whiskerprops=dict(color="C0"))
    ax.set_xlim(0.75, 1.25)
    ax.set_xticks([0.75, 1, 1.25])
    #axes[i].set_xticklabels(["", data["models_name"][i], ""])
    ax.set_xticklabels(["", "", ""])
    plt.setp(ax.get_xticklabels()) #, rotation=90, fontsize='x-small')
    ax.set_ylim(0, 32)
    ax.set(adjustable='box', aspect=0.015625)#'equal')
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, axis="y", which="both", ls=":")
    ax.set_ylabel("relevant pixel ratio in %", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

def get_indeces_native_pca(data, limit=800):
    mean_value = []
    for i in range(len(data.x)):
        mean_value.append(np.sum(data.x[i] * data.y[i]) / (np.sum(data.y[i])))
    mean_value = np.array(mean_value)

    indeces_pca = np.argwhere(mean_value < limit).flatten()
    indeces_native = np.argwhere(mean_value >= limit).flatten()
    return indeces_native, indeces_pca

def split_native_pca(data, limit=800):
    indeces_native, indeces_pca = get_indeces_native_pca(data, limit)
    data_native = copy.copy(data)
    data_pca = copy.copy(data)

    data_native.x = [data_native.x[i] for i in indeces_native]
    data_native.x_bb = [data_native.x_bb[i] for i in indeces_native]
    data_native.y = [data_native.y[i] for i in indeces_native]
    data_native.y_contour = [data_native.y_contour[i] for i in indeces_native]
    data_native.information = [data_native.information[i] for i in indeces_native]
    data_native.pixel_spacing = [data_native.pixel_spacing[i] for i in indeces_native]

    data_pca.x = [data_pca.x[i] for i in indeces_pca]
    data_pca.x_bb = [data_pca.x_bb[i] for i in indeces_pca]
    data_pca.y = [data_pca.y[i] for i in indeces_pca]
    data_pca.y_contour = [data_pca.y_contour[i] for i in indeces_pca]
    data_pca.information = [data_pca.information[i] for i in indeces_pca]
    data_pca.pixel_spacing = [data_pca.pixel_spacing[i] for i in indeces_pca]

    return data_native, data_pca




if __name__ == "__main__":
    mode = "true" #"pred"
    path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

    path_data_train = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TRAIN"
    path_data_val = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\VALIDATION"
    path_data_test = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
    path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\bb_ratio_relevant_pixels.jpg"

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
                    break
        if not model_bb is None:
            break

    data_train = mmsgenerators.unet.Setup(path_data_train, None, "TEST",  model_bb=model_bb, ws_dir=path_data_train, mask_mode="RASTERIZE")
    data_val = mmsgenerators.unet.Setup(path_data_val, None, "TEST",  model_bb=model_bb, ws_dir=path_data_val, mask_mode="RASTERIZE")
    data_test = mmsgenerators.unet.Setup(path_data_test, None, "TEST",  model_bb=model_bb, ws_dir=path_data_test, mask_mode="RASTERIZE")


    data_train_native, data_train_pca = split_native_pca(data_train)
    data_val_native, data_val_pca = split_native_pca(data_val)
    data_test_native, data_test_pca = split_native_pca(data_test)

    # INFO PRINT
    info_train = get_data_info(data_train, mode)
    info_val = get_data_info(data_val, mode)
    info_test = get_data_info(data_test, mode)

    print("Train data: original: " + str(np.mean(info_train["relevant_raw"])) + " +/- " + str(np.std(info_train["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_train["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_train["relevant_crop"])) + " +/- " + str(np.std(info_train["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_train["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_train["relevant_raw"], info_train["relevant_crop"])[-1]))
    print("Validation data: original: " + str(np.mean(info_val["relevant_raw"])) + " +/- " + str(np.std(info_val["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_val["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_val["relevant_crop"])) + " +/- " + str(np.std(info_val["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_val["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_val["relevant_raw"], info_val["relevant_crop"])[-1]))
    print("Test data: original: " + str(np.mean(info_test["relevant_raw"])) + " +/- " + str(np.std(info_test["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_test["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_test["relevant_crop"])) + " +/- " + str(np.std(info_test["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_test["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_test["relevant_raw"], info_test["relevant_crop"])[-1]))



    #rows = 1
    #cols = 1
    #fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols}, figsize=(20, 10))

    #axes.boxplot([info_train["relevant_raw"], info_val["relevant_raw"], info_test["relevant_raw"], info_train["relevant_crop"], info_val["relevant_crop"], info_test["relevant_crop"]], zorder=10, flierprops=dict(markerfacecolor='C0', markeredgecolor="C0", marker='.', markersize=5), boxprops=dict(color="C0"), capprops=dict(color="C0"), whiskerprops=dict(color="C0"))
    #axes.axvline(3.5, ls = "--", c="lightgray")

    #axes.set_xlim(0.5, 6.5)
    #axes.set_ylim(0, 32)
    #axes.yaxis.set_minor_locator(MultipleLocator(1))
    #axes.yaxis.set_major_locator(MultipleLocator(5))
    #axes.set_ylabel("relevant pixel ratio in %", fontsize=12)
    #axes.grid(True, axis="y", which="both", ls=":")

    #axes.set_xticks([0.5, 1, 2, 3, 4, 5, 6, 6.5])
    #axes.set_xticklabels(["", "train\ndata", "validation\ndata", "test\ndata", "train\ndata", "validation\ndata", "test\ndata", ""])

    #axes.annotate("original image", xy=(0.005, 0.97), xycoords="axes fraction", fontsize=20, ha='left', va='center', rotation="horizontal")
    #axes.annotate("image section", xy=(0.505, 0.97), xycoords="axes fraction", fontsize=20, ha='left', va='center', rotation="horizontal")

    #axes.set_xlabel("image")

    cols = 3
    rows = 3.75

    fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True, dpi=300)
    subfigs = fig.subfigures(2, 1)#, width_ratios = [0.05, 1, 1, 0.05], height_ratios = [1, 1, 1])

    subsubfigs0 = subfigs[0].subfigures(2,1)
    subsubfigs1 = subfigs[1].subfigures(2,1)

    axes_ssf00 = subsubfigs0[0].subplots(1,3)
    axes_ssf01 = subsubfigs0[1].subplots(1,3)
    axes_ssf10 = subsubfigs1[0].subplots(1,3)
    axes_ssf11 = subsubfigs1[1].subplots(1,3)

    axes = np.vstack((axes_ssf00, axes_ssf01, axes_ssf10, axes_ssf11))


    subfigs[0].supylabel("native", fontsize=24, fontweight="bold")
    subsubfigs0[0].supylabel("original image", fontsize=24, fontweight="bold")
    subsubfigs0[1].supylabel("cropped image", fontsize=24, fontweight="bold")
    subfigs[1].supylabel("post contrast agent", fontsize=24, fontweight="bold")
    subsubfigs1[0].supylabel("original image", fontsize=24, fontweight="bold")
    subsubfigs1[1].supylabel("cropped image", fontsize=24, fontweight="bold")

    axes[0,0].set_title("train data", fontsize=24, fontweight="bold")
    axes[0,1].set_title("validation data", fontsize=24, fontweight="bold")
    axes[0,2].set_title("test data", fontsize=24, fontweight="bold")

    axes[2,0].set_title("train data", fontsize=24, fontweight="bold")
    axes[2,1].set_title("validation data", fontsize=24, fontweight="bold")
    axes[2,2].set_title("test data", fontsize=24, fontweight="bold")

    plt.plot([0, 1], [0.5, 0.5], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)


    info_train = get_data_info(data_train_native, mode)
    info_val = get_data_info(data_val_native, mode)
    info_test = get_data_info(data_test_native, mode)

    print("Train data native: original: " + str(np.mean(info_train["relevant_raw"])) + " +/- " + str(np.std(info_train["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_train["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_train["relevant_crop"])) + " +/- " + str(np.std(info_train["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_train["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_train["relevant_raw"], info_train["relevant_crop"])[-1]))
    print("Validation data native: original: " + str(np.mean(info_val["relevant_raw"])) + " +/- " + str(np.std(info_val["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_val["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_val["relevant_crop"])) + " +/- " + str(np.std(info_val["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_val["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_val["relevant_raw"], info_val["relevant_crop"])[-1]))
    print("Test data native: original: " + str(np.mean(info_test["relevant_raw"])) + " +/- " + str(np.std(info_test["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_test["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_test["relevant_crop"])) + " +/- " + str(np.std(info_test["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_test["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_test["relevant_raw"], info_test["relevant_crop"])[-1]))

    boxplot(axes[0, 0], info_train["relevant_raw"])
    boxplot(axes[0, 1], info_val["relevant_raw"])
    boxplot(axes[0, 2], info_test["relevant_raw"])

    boxplot(axes[1, 0], info_train["relevant_crop"])
    boxplot(axes[1, 1], info_val["relevant_crop"])
    boxplot(axes[1, 2], info_test["relevant_crop"])

    info_train = get_data_info(data_train_pca, mode)
    info_val = get_data_info(data_val_pca, mode)
    info_test = get_data_info(data_test_pca, mode)

    print("Train data pca: original: " + str(np.mean(info_train["relevant_raw"])) + " +/- " + str(np.std(info_train["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_train["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_train["relevant_crop"])) + " +/- " + str(np.std(info_train["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_train["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_train["relevant_raw"], info_train["relevant_crop"])[-1]))
    print("Validation data pca: original: " + str(np.mean(info_val["relevant_raw"])) + " +/- " + str(np.std(info_val["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_val["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_val["relevant_crop"])) + " +/- " + str(np.std(info_val["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_val["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_val["relevant_raw"], info_val["relevant_crop"])[-1]))
    print("Test data pca: original: " + str(np.mean(info_test["relevant_raw"])) + " +/- " + str(np.std(info_test["relevant_raw"])) + " shapiro = " + str(stats.shapiro(info_test["relevant_raw"])[-1]) + " / crop: " + str(np.mean(info_test["relevant_crop"])) + " +/- " + str(np.std(info_test["relevant_crop"])) + " shapiro = " + str(stats.shapiro(info_test["relevant_crop"])[-1]) + " / Wilcoxon = " + str(stats.wilcoxon(info_test["relevant_raw"], info_test["relevant_crop"])[-1]))

    boxplot(axes[2, 0], info_train["relevant_raw"])
    boxplot(axes[2, 1], info_val["relevant_raw"])
    boxplot(axes[2, 2], info_test["relevant_raw"])

    boxplot(axes[3, 0], info_train["relevant_crop"])
    boxplot(axes[3, 1], info_val["relevant_crop"])
    boxplot(axes[3, 2], info_test["relevant_crop"])


    if not path_out:
        plt.show()
    else:
        plt.savefig(path_out)
