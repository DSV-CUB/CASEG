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

def subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, indece_index, ax, set_alpha, is_native=True, model_num=-1):
    image = data.x[indeces[indece_index]]
    if not is_native:
        image = image / np.max(image)
        image = 10 * image
        image = image - 0.5
        image = np.clip(image,0, 1)

    indeces2 = np.argwhere(expectations[2][indece_index])
    x_min = np.min(indeces2[:,0])
    x_max = np.max(indeces2[:,0])
    y_min = np.min(indeces2[:,1])
    y_max = np.max(indeces2[:,1])

    dx1 = int((x_max-x_min)/2 + x_min) - 30
    dx2 = int((x_max-x_min)/2 + x_min) + 30
    dy1 = int((y_max-y_min)/2 + y_min) - 30
    dy2 = int((y_max-y_min)/2 + y_min) + 30

    ax.imshow(image[dx1:dx2, dy1:dy2], cmap="gray")

    if model_num > -1:
        _, rgba_mask = tools.tool_plot.masks2delta2rgba(expectations[model_num][indece_index], predictions[model_num][indece_index], alpha=set_alpha)
        ax.imshow(rgba_mask[dx1:dx2, dy1:dy2])

        info_text = "DSC = " +  str(tools.tool_general.get_metric_from_masks(expectations[model_num][indece_index], predictions[model_num][indece_index], "DSC")) + " % | HD = " + str(tools.tool_general.get_metric_from_masks(expectations[model_num][indece_index], predictions[model_num][indece_index], "HD", voxel_sizes=data.pixel_spacing[indeces[indece_index]])) + " mm | \u0394T1 = "  + str(round(expectations_values[model_num][indece_index] - predictions_values[model_num][indece_index],2)) + " ms"
        height, width, box_height = dy2-dy1, dx2-dx1, 5
        ax.add_patch(plt.Rectangle((-0.5, height-box_height-0.5), width+1, box_height,color="#fff1c7"))
        ax.text(int(0.5*width), height-3, info_text, horizontalalignment='center', verticalalignment='center', fontsize=11, color='black', backgroundcolor="#fff1c7")


    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    return

def plot_example_cases(expectations, predictions, indeces, expectations_values, predictions_values, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, data, path_out=False, do_tight=False):
    set_alpha = 0.66

    # RUN PLOT
    # all together
    cols = 4
    rows = 6

    fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True, dpi=300)
    subfigs = fig.subfigures(2, 1) #, width_ratios = [1] * 2, height_ratios = [1] * 2)

    axes_sf0 = subfigs[0].subplots(3,4) #, gridspec_kw={'width_ratios': [1] * 4, 'height_ratios': [1] * 2})
    axes_sf1 = subfigs[1].subplots(3,4) #, gridspec_kw={'width_ratios': [1] * 4, 'height_ratios': [1] * 2})
    axes = np.hstack((axes_sf0, axes_sf1))

    plt.plot([0, 1], [0.5, 0.5], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)

    subfigs[0].supylabel("native", fontsize=24, fontweight='bold')
    subfigs[1].supylabel("post contrast agent", fontsize=24, fontweight='bold')





    #cols = 8
    #rows = 4

    #fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True)
    #subfigs = fig.subfigures(1, 2) #, width_ratios = [1] * 2, height_ratios = [1] * 2)
    #separating line


    #axes_sf0 = subfigs[0].subplots(4,4, gridspec_kw={'width_ratios': [1] * 4, 'height_ratios': [1] * 4})
    #axes_sf1 = subfigs[1].subplots(4,4, gridspec_kw={'width_ratios': [1] * 4, 'height_ratios': [1] * 4})
    #axes = np.hstack((axes_sf0, axes_sf1))

    #plt.plot([0.5, 0.5], [0, 1], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)

    subfigs[0].suptitle(" ", fontsize=4, fontweight='bold')
    subfigs[1].suptitle(" ", fontsize=4, fontweight='bold')

    axes[0,0].set_ylabel("good case", fontsize=20)
    axes[1,0].set_ylabel("improvement case", fontsize=20)
    axes[2,0].set_ylabel("bad case", fontsize=20)
    #axes[3,0].set_ylabel("crinU vs. expert\nsegmentation", fontsize=20)

    axes[0,0].set_title("original image", fontsize=20, fontweight='bold')
    axes[0,1].set_title("refU", fontsize=20, fontweight='bold')
    axes[0,2].set_title("cropU", fontsize=20, fontweight='bold')
    axes[0,3].set_title("crinU", fontsize=20, fontweight='bold')

    axes[0,4].set_title("original image", fontsize=20, fontweight='bold')
    axes[0,5].set_title("refU", fontsize=20, fontweight='bold')
    axes[0,6].set_title("cropU", fontsize=20, fontweight='bold')
    axes[0,7].set_title("crinU", fontsize=20, fontweight='bold')

    #axes[0,7].yaxis.set_label_position("right")
    #axes[1,7].yaxis.set_label_position("right")
    #axes[2,7].yaxis.set_label_position("right")
    #axes[3,7].yaxis.set_label_position("right")

    #axes[0,7].set_ylabel("original\nimage", fontsize=20)
    #axes[1,7].set_ylabel("refU vs. expert\nsegmentation", fontsize=20)
    #axes[2,7].set_ylabel("cropU vs. expert\nsegmentation", fontsize=20)
    #axes[3,7].set_ylabel("crinU vs. expert\nsegmentation", fontsize=20)

    axes[0,4].set_ylabel("good case", fontsize=20)
    axes[1,4].set_ylabel("improvement case", fontsize=20)
    axes[2,4].set_ylabel("bad case", fontsize=20)
    #axes[3,4].set_ylabel("crinU vs. expert\nsegmentation", fontsize=20)




    # native
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

    # native good
    index = np.argmax(dscs[2])
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[0,0], set_alpha, is_native=True, model_num=-1)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[0,1], set_alpha, is_native=True, model_num=0)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[0,2], set_alpha, is_native=True, model_num=1)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[0,3], set_alpha, is_native=True, model_num=2)

    # native improvement
    index = np.argmax(np.abs(np.array(dscs[2])-np.array(dscs[0])))
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[1,0], set_alpha, is_native=True, model_num=-1)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[1,1], set_alpha, is_native=True, model_num=0)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[1,2], set_alpha, is_native=True, model_num=1)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[1,3], set_alpha, is_native=True, model_num=2)

    # native bad geometric
    #index = np.argsort(dscs[0])[1]
    #subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[0,2], set_alpha, is_native=True, model_num=-1)
    #subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[1,2], set_alpha, is_native=True, model_num=0)
    #subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[2,2], set_alpha, is_native=True, model_num=1)
    #subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[3,2], set_alpha, is_native=True, model_num=2)

    # native bad quantitative
    index = np.argsort(t1e[0])[-1]
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[2,0], set_alpha, is_native=True, model_num=-1)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[2,1], set_alpha, is_native=True, model_num=0)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[2,2], set_alpha, is_native=True, model_num=1)
    subplot_example_case(data, expectations, predictions, expectations_values, predictions_values, indeces, index, axes[2,3], set_alpha, is_native=True, model_num=2)

    # pca
    dscs_pca = []
    hds_pca = []
    t1e_pca = []
    for j in range(len(models_names)):
        dscs_pca_model = []
        hds_pca_model = []
        t1e_pca_model = []
        for i in range(len(expectations_pca[j])):
            dscs_pca_model.append(tools.tool_general.get_metric_from_masks(expectations_pca[j][i], predictions_pca[2][i], "DSC"))
            hds_pca_model.append(tools.tool_general.get_metric_from_masks(expectations_pca[j][i], predictions_pca[2][i], "HD", voxel_sizes=data.pixel_spacing[indeces_pca[i]]))
            t1e_pca_model.append(predictions_values_pca[2][i] - expectations_values_pca[j][i])
        dscs_pca.append(dscs_pca_model)
        hds_pca.append(hds_pca_model)
        t1e_pca.append(t1e_pca_model)

    # pca good
    index = np.argmax(dscs_pca[2])
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[0,4], set_alpha, is_native=False, model_num=-1)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[0,5], set_alpha, is_native=False, model_num=0)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[0,6], set_alpha, is_native=False, model_num=1)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[0,7], set_alpha, is_native=False, model_num=2)

    # pca improvement
    index = np.argmax(np.abs(np.array(dscs_pca[2])-np.array(dscs_pca[0])))
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[1,4], set_alpha, is_native=False, model_num=-1)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[1,5], set_alpha, is_native=False, model_num=0)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[1,6], set_alpha, is_native=False, model_num=1)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[1,7], set_alpha, is_native=False, model_num=2)

    # pca bad geometric
    index = np.argsort(dscs_pca[0])[1]
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[2,4], set_alpha, is_native=False, model_num=-1)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[2,5], set_alpha, is_native=False, model_num=0)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[2,6], set_alpha, is_native=False, model_num=1)
    subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[2,7], set_alpha, is_native=False, model_num=2)

    # pca bad quantitative
    #index = np.argsort(t1e_pca[0])[-1]
    #subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[0,7], set_alpha, is_native=False, model_num=-1)
    #subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[1,7], set_alpha, is_native=False, model_num=0)
    #subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[2,7], set_alpha, is_native=False, model_num=1)
    #subplot_example_case(data, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, index, axes[3,7], set_alpha, is_native=False, model_num=2)

    if not path_out:
        plt.show()
    else:
        plt.savefig(path_out)
    return



if __name__ == "__main__":
    path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"
    path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
    path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\PLOTS\MV Data\cases_examples.jpg"

    do_tight = False

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

    plot_example_cases(expectations_native, predictions_native, indeces_native, expectations_values_native, predictions_values_native, expectations_pca, predictions_pca, expectations_values_pca, predictions_values_pca, indeces_pca, data, path_out=path_out, do_tight=do_tight)