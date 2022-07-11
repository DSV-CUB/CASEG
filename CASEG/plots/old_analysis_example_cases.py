import os
import pickle
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib.ticker import MultipleLocator

from marissa.toolbox import tools
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from CASEG.plots import t1_disjoint_comparison

def plot_example_cases(expectations, predictions, expectations_values, predictions_values, data, indeces, path_out=False, do_tight=False, contrast_increase=False):
    set_alpha = 0.66
    dscs = []
    hds = []
    t1e = []
    for j in range(len(models_names)):
        dscs_model = []
        hds_model = []
        t1e_model = []
        for i in range(len(expectations[j])):
            dscs_model.append(tools.tool_general.get_metric_from_masks(expectations[j][i], predictions[2][i], "DSC"))
            hds_model.append(tools.tool_general.get_metric_from_masks(expectations[j][i], predictions[2][i], "HD", voxel_sizes=data.pixel_spacing[indeces[i]]))
            t1e_model.append(predictions_values[2][i] - expectations_values[j][i])
        dscs.append(dscs_model)
        hds.append(hds_model)
        t1e.append(t1e_model)

    cols = 4
    rows = 4
    fig, axes = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [1] * cols}, figsize=(cols*5, rows*5))

    axes[0,0].set_title("original\nimage", fontsize=14)
    axes[0,1].set_title("refU vs. expert\nsegmentation", fontsize=14)
    axes[0,2].set_title("cropU vs. expert\nsegmentation", fontsize=14)
    axes[0,3].set_title("crinU vs. expert\nsegmentation", fontsize=14)

    axes[0,0].set_ylabel("Good\nExample", fontsize=14)
    axes[1,0].set_ylabel("Improvement\nExample", fontsize=14)
    axes[2,0].set_ylabel("Bad Geometric\nExample", fontsize=14)
    axes[3,0].set_ylabel("Bad Quantitative\nExample", fontsize=14)

    # 0 GOOD CASE
    index = np.argmax(dscs[2])
    image = data.x[indeces[index]]
    if contrast_increase:
        image = image / np.max(image)
        image = 10 * image
        image = image - 0.5
        image = np.clip(image,0, 1)

    indeces2 = np.argwhere(expectations[2][index])
    x_min = np.min(indeces2[:,0])
    x_max = np.max(indeces2[:,0])
    y_min = np.min(indeces2[:,1])
    y_max = np.max(indeces2[:,1])

    dx1 = int((x_max-x_min)/2 + x_min) - 30
    dx2 = int((x_max-x_min)/2 + x_min) + 30
    dy1 = int((y_max-y_min)/2 + y_min) - 30
    dy2 = int((y_max-y_min)/2 + y_min) + 30

    #orig
    axes[0,0].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    axes[0,0].xaxis.set_visible(False)
    axes[0,0].yaxis.set_ticklabels([])
    axes[0,0].yaxis.set_ticks([])

    #refU
    axes[0,1].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[0][index], predictions[0][index], alpha=set_alpha)
    axes[0,1].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[0,1].xaxis.set_visible(False)
    axes[0,1].yaxis.set_ticklabels([])
    axes[0,1].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[0][index] - predictions_values[0][index],2)) + " ms"
    axes[0,1].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #cropU
    axes[0,2].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[1][index], predictions[1][index], alpha=set_alpha)
    axes[0,2].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[0,2].xaxis.set_visible(False)
    axes[0,2].yaxis.set_ticklabels([])
    axes[0,2].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[1][index] - predictions_values[1][index],2)) + " ms"
    axes[0,2].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #crinU
    axes[0,3].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[2][index], predictions[2][index], alpha=set_alpha)
    axes[0,3].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[0,3].xaxis.set_visible(False)
    axes[0,3].yaxis.set_ticklabels([])
    axes[0,3].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[2][index] - predictions_values[2][index],2)) + " ms"
    axes[0,3].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    # 1 IMPROVEMENT CASE
    index = np.argmax(np.abs(np.array(dscs[2])-np.array(dscs[0])))
    image = data.x[indeces[index]]
    if contrast_increase:
        image = image / np.max(image)
        image = 10 * image
        image = image - 0.5
        image = np.clip(image,0, 1)

    indeces2 = np.argwhere(expectations[2][index])
    x_min = np.min(indeces2[:,0])
    x_max = np.max(indeces2[:,0])
    y_min = np.min(indeces2[:,1])
    y_max = np.max(indeces2[:,1])

    diff = np.max([x_max-x_min, y_max-y_min])

    dx1 = int((x_max-x_min)/2 + x_min) - 30
    dx2 = int((x_max-x_min)/2 + x_min) + 30
    dy1 = int((y_max-y_min)/2 + y_min) - 30
    dy2 = int((y_max-y_min)/2 + y_min) + 30


    #orig
    axes[1,0].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    axes[1,0].xaxis.set_visible(False)
    axes[1,0].yaxis.set_ticklabels([])
    axes[1,0].yaxis.set_ticks([])

    #refU
    axes[1,1].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[0][index], predictions[0][index], alpha=set_alpha)
    axes[1,1].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[1,1].xaxis.set_visible(False)
    axes[1,1].yaxis.set_ticklabels([])
    axes[1,1].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[0][index] - predictions_values[0][index],2)) + " ms"
    axes[1,1].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #cropU
    axes[1,2].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[1][index], predictions[1][index], alpha=set_alpha)
    axes[1,2].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[1,2].xaxis.set_visible(False)
    axes[1,2].yaxis.set_ticklabels([])
    axes[1,2].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[1][index] - predictions_values[1][index],2)) + " ms"
    axes[1,2].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #crinU
    axes[1,3].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[2][index], predictions[2][index], alpha=set_alpha)
    axes[1,3].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[1,3].xaxis.set_visible(False)
    axes[1,3].yaxis.set_ticklabels([])
    axes[1,3].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[2][index] - predictions_values[2][index],2)) + " ms"
    axes[1,3].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    # 2 BAD GEOMETRIC CASE
    index = np.argsort(dscs[0])[1]
    image = data.x[indeces[index]]
    if contrast_increase:
        image = image / np.max(image)
        image = 10 * image
        image = image - 0.5
        image = np.clip(image,0, 1)

    indeces2 = np.argwhere(expectations[2][index])
    x_min = np.min(indeces2[:,0])
    x_max = np.max(indeces2[:,0])
    y_min = np.min(indeces2[:,1])
    y_max = np.max(indeces2[:,1])

    diff = np.max([x_max-x_min, y_max-y_min])

    dx1 = int((x_max-x_min)/2 + x_min) - 30
    dx2 = int((x_max-x_min)/2 + x_min) + 30
    dy1 = int((y_max-y_min)/2 + y_min) - 30
    dy2 = int((y_max-y_min)/2 + y_min) + 30

    #orig
    axes[2,0].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    axes[2,0].xaxis.set_visible(False)
    axes[2,0].yaxis.set_ticklabels([])
    axes[2,0].yaxis.set_ticks([])

    #refU
    axes[2,1].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[0][index], predictions[0][index], alpha=set_alpha)
    axes[2,1].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[2,1].xaxis.set_visible(False)
    axes[2,1].yaxis.set_ticklabels([])
    axes[2,1].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[0][index] - predictions_values[0][index],2)) + " ms"
    axes[2,1].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #cropU
    axes[2,2].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[1][index], predictions[1][index], alpha=set_alpha)
    axes[2,2].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[2,2].xaxis.set_visible(False)
    axes[2,2].yaxis.set_ticklabels([])
    axes[2,2].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[1][index] - predictions_values[1][index],2)) + " ms"
    axes[2,2].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #crinU
    axes[2,3].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[2][index], predictions[2][index], alpha=set_alpha)
    axes[2,3].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[2,3].xaxis.set_visible(False)
    axes[2,3].yaxis.set_ticklabels([])
    axes[2,3].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[2][index] - predictions_values[2][index],2)) + " ms"
    axes[2,3].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    # 3 BAD QUANTITATIVE CASE
    index = np.argsort(t1e[0])[-1]
    image = data.x[indeces[index]]
    if contrast_increase:
        image = image / np.max(image)
        image = 10 * image
        image = image - 0.5
        image = np.clip(image,0, 1)

    indeces2 = np.argwhere(expectations[2][index])
    x_min = np.min(indeces2[:,0])
    x_max = np.max(indeces2[:,0])
    y_min = np.min(indeces2[:,1])
    y_max = np.max(indeces2[:,1])

    diff = np.max([x_max-x_min, y_max-y_min])

    dx1 = int((x_max-x_min)/2 + x_min) - 30
    dx2 = int((x_max-x_min)/2 + x_min) + 30
    dy1 = int((y_max-y_min)/2 + y_min) - 30
    dy2 = int((y_max-y_min)/2 + y_min) + 30


    #orig
    axes[3,0].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    axes[3,0].xaxis.set_visible(False)
    axes[3,0].yaxis.set_ticklabels([])
    axes[3,0].yaxis.set_ticks([])

    #refU
    axes[3,1].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[0][index], predictions[0][index], alpha=set_alpha)
    axes[3,1].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[3,1].xaxis.set_visible(False)
    axes[3,1].yaxis.set_ticklabels([])
    axes[3,1].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[0][index], predictions[0][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[0][index] - predictions_values[0][index],2)) + " ms"
    axes[3,1].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #cropU
    axes[3,2].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[1][index], predictions[1][index], alpha=set_alpha)
    axes[3,2].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[3,2].xaxis.set_visible(False)
    axes[3,2].yaxis.set_ticklabels([])
    axes[3,2].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[1][index], predictions[1][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[1][index] - predictions_values[1][index],2)) + " ms"
    axes[3,2].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    #crinU
    axes[3,3].imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[2][index], predictions[2][index], alpha=set_alpha)
    axes[3,3].imshow(rgba_mask[dx1:dx2, dy1:dy2])
    axes[3,3].xaxis.set_visible(False)
    axes[3,3].yaxis.set_ticklabels([])
    axes[3,3].yaxis.set_ticks([])
    info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[2][index], predictions[2][index], "HD", voxel_sizes=data.pixel_spacing[indeces[index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[2][index] - predictions_values[2][index],2)) + " ms"
    axes[3,3].text(30, 58, info_text, horizontalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    if do_tight:
        plt.tight_layout()
    if not path_out:
        plt.show()
    else:
        plt.gcf().set_dpi(300)
        plt.savefig(path_out)
    return



if __name__ == "__main__":
    path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"
    path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
    path_out_native = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\cases_example_native.jpg"
    path_out_pca = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\cases_example_pca.jpg"

    do_tight = True

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


    for i in range(len(models_names)):
        print(models_names[i] + " global RMSE = " + str(np.sqrt(np.mean((np.array(predictions_values[i])-np.array(expectations_values[i]))**2))))
        print(models_names[i] + " native RMSE = " + str(np.sqrt(np.mean((np.array(predictions_values_native[i])-np.array(expectations_values_native[i]))**2))))
        print(models_names[i] + " pca RMSE = " + str(np.sqrt(np.mean((np.array(predictions_values_pca[i])-np.array(expectations_values_pca[i]))**2))))
        print("")

    plot_example_cases(expectations_native, predictions_native, expectations_values_native, predictions_values_native, data, indeces_native, path_out=path_out_native, do_tight=do_tight)
    plot_example_cases(expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, data, indeces_pca, path_out=path_out_pca, do_tight=do_tight, contrast_increase=True)