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

def subplot_example_case(data, expectations, predictions, indeces, indece_index, ax, set_alpha, is_native=True):

    image = data.x[indeces[indece_index]]
    if not is_native:
        image = image / np.max(image)
        image = 10 * image
        image = image - 0.5
        image = np.clip(image,0, 1)

    indeces_bb = np.argwhere(expectations[0][indece_index])
    x_min = np.min(indeces_bb[:,0])
    x_max = np.max(indeces_bb[:,0])
    y_min = np.min(indeces_bb[:,1])
    y_max = np.max(indeces_bb[:,1])

    dx1 = int((x_max-x_min)/2 + x_min) - 35
    dx2 = int((x_max-x_min)/2 + x_min) + 35
    dy1 = int((y_max-y_min)/2 + y_min) - 35
    dy2 = int((y_max-y_min)/2 + y_min) + 35

    ax.imshow(image[dx1:dx2, dy1:dy2], cmap="gray")
    _, rgba_mask = tools.tool_plot.masks2delta2rgba(predictions[0][indece_index], expectations[0][indece_index], alpha=set_alpha)
    ax.imshow(rgba_mask[dx1:dx2, dy1:dy2])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])

    #info_text = "DSC = " + str(tools.tool_general.get_metric_from_masks(expectations[0][indece_index], predictions[0][indece_index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[0][indece_index], predictions[0][indece_index], "HD", voxel_sizes=data.pixel_spacing[indeces[indece_index]])) + " mm"
    #height, width, box_height = dy2-dy1, dx2-dx1, 7
    #ax.add_patch(plt.Rectangle((-0.5, height-box_height-0.5), width+1, box_height,color="#fff1c7"))
    #ax.text(int(0.5*width), height-4, info_text, horizontalalignment='center', verticalalignment='center', fontsize=14, color='black', backgroundcolor="#fff1c7")
    info_text = "DSC = " + str(tools.tool_general.get_metric_from_masks(expectations[0][indece_index], predictions[0][indece_index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[0][indece_index], predictions[0][indece_index], "HD", voxel_sizes=data.pixel_spacing[indeces[indece_index]])) + " mm"
    height, width, box_height = dy2-dy1, dx2-dx1, 5
    ax.add_patch(plt.Rectangle((-0.5, height-box_height-0.5), width+1, box_height,color="#fff1c7"))
    ax.text(int(0.5*width), height-3, info_text, horizontalalignment='center', verticalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")

    return


def plot_example_cases(expectations, predictions, expectations_pca, predictions_pca, data, indeces, indeces_pca, path_out=False, do_tight=False):
    set_alpha = 0.33
    dscs = []
    hds = []
    dscs_pca = []
    hds_pca = []
    for i in range(len(expectations[0])):
        dscs.append(tools.tool_general.get_metric_from_masks(expectations[0][i], predictions[0][i], "DSC"))
        hds.append(tools.tool_general.get_metric_from_masks(expectations[0][i], predictions[0][i], "HD", voxel_sizes=data.pixel_spacing[indeces[i]]))
    for i in range(len(expectations_pca[0])):
        dscs_pca.append(tools.tool_general.get_metric_from_masks(expectations_pca[0][i], predictions_pca[0][i], "DSC"))
        hds_pca.append(tools.tool_general.get_metric_from_masks(expectations_pca[0][i], predictions_pca[0][i], "HD", voxel_sizes=data.pixel_spacing[indeces_pca[i]]))

    index_best_dsc = np.argmax(dscs)
    index_best_hds = np.argmin(hds)
    index_bad_dsc = np.argmin(dscs)
    index_bad_hds = np.argmax(hds)
    print(index_best_dsc == index_best_hds)

    index_best_dsc_pca = np.argmax(dscs_pca)
    index_best_hds_pca = np.argmin(hds_pca)
    index_bad_dsc_pca = np.argmin(dscs_pca)
    index_bad_hds_pca = np.argmax(hds_pca)
    print(index_best_dsc_pca == index_best_hds_pca)


    # RUN PLOT
    # all together
    cols = 2
    rows = 4

    fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True)
    subfigs = fig.subfigures(2, 1) #, width_ratios = [1] * 2, height_ratios = [1] * 2)
    #separating line


    axes_sf0 = subfigs[0].subplots(2,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1] * 2})
    axes_sf1 = subfigs[1].subplots(2,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1] * 2})
    axes = np.hstack((axes_sf0, axes_sf1))

    plt.plot([0, 1], [0.5, 0.5], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)

    subfigs[0].supylabel("native", fontsize=24, fontweight='bold')
    subfigs[1].supylabel("contrast enhanced", fontsize=24, fontweight='bold')

    axes[0,0].set_title("best DSC case", fontsize=20, fontweight='bold')
    axes[0,1].set_title("worst DSC case", fontsize=20, fontweight='bold')
    axes[0,2].set_title("best DSC case", fontsize=20, fontweight='bold')
    axes[0,3].set_title("worst DSC case", fontsize=20, fontweight='bold')

    axes[1,0].set_title("best HD case", fontsize=20, fontweight='bold')
    axes[1,1].set_title("worst HD case", fontsize=20, fontweight='bold')
    axes[1,2].set_title("best HD case", fontsize=20, fontweight='bold')
    axes[1,3].set_title("worst HD case", fontsize=20, fontweight='bold')

    #axes[0,0].set_ylabel("DSC", fontsize=20)
    #axes[1,0].set_ylabel("HD", fontsize=20)
    #axes[0,3].yaxis.set_label_position("right")
    #axes[0,3].set_ylabel("DSC", fontsize=20)
    #axes[1,3].yaxis.set_label_position("right")
    #axes[1,3].set_ylabel("HD", fontsize=20)


    #native DSC best case
    subplot_example_case(data, expectations, predictions, indeces, index_best_dsc, axes[0,0], set_alpha)

    #native HD best case
    subplot_example_case(data, expectations, predictions, indeces, index_best_hds, axes[1,0], set_alpha)

    #native DSC worst case
    subplot_example_case(data, expectations, predictions, indeces, index_bad_dsc, axes[0,1], set_alpha)

    #native HD worst case
    subplot_example_case(data, expectations, predictions, indeces, index_bad_hds, axes[1,1], set_alpha)


    #pca DSC best case
    subplot_example_case(data, expectations_pca, predictions_pca, indeces_pca, index_best_dsc_pca, axes[0,2], set_alpha, False)

    #pca HD best case
    subplot_example_case(data, expectations_pca, predictions_pca, indeces_pca, index_best_hds_pca, axes[1,2], set_alpha, False)

    #pca DSC worst case
    subplot_example_case(data, expectations_pca, predictions_pca, indeces_pca, index_bad_dsc_pca, axes[0,3], set_alpha, False)

    #pca HD worst case
    subplot_example_case(data, expectations_pca, predictions_pca, indeces_pca, index_bad_hds_pca, axes[1,3], set_alpha, False)



    if do_tight:
        plt.tight_layout()

    if not path_out:
        plt.show()
    else:
        plt.gcf().set_dpi(300)
        plt.savefig(path_out)
    return

if __name__ == "__main__":
    path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX_Paper"
    path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX\TEST_BASMV"
    path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\Paper\Figure 2 - Example results of the object detection algorithm showing bounding boxes for the left ventricular myocardium.jpg"

    do_tight = False

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

    data = mmsgenerators.unet.Setup(path_data, None, "TEST",  model_bb=model_bb, ws_dir=path_data, mask_mode="RASTERIZE")

    predictions = []
    expectations = []
    predictions_values = []
    expectations_values = []

    data.configuration.set(**model_bb.configuration.__dict__)
    prediction, expectation = model_bb.predict_generator(data)
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
    indeces_all = np.argwhere(np.array(expectations_values[0]) > -1).flatten()

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

    information = {"models_name": [""],
                   "data": data.x,
                   "predictions": predictions,
                   "expectations": expectations,
                   "predictions_values": predictions_values,
                   "expectations_values": expectations_values}

    information_pca = {"models_name": [""],
                   "data": data_pca,
                   "predictions": predictions_pca,
                   "expectations": expectations_pca,
                   "predictions_values": predictions_values_pca,
                   "expectations_values": expectations_values_pca}

    information_native = {"models_name": [""],
                   "data": data_native,
                   "predictions": predictions_native,
                   "expectations": expectations_native,
                   "predictions_values": predictions_values_native,
                   "expectations_values": expectations_values_native}

    plot_example_cases(expectations_native, predictions_native, expectations_pca, predictions_pca, data, indeces_native, indeces_pca, path_out=path_out, do_tight=do_tight)

