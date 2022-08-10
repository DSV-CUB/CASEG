import copy
import os
import numpy as np
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from marissa.toolbox.tools import tool_plot

path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"

#path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\1 - Project Documents\Illustration Pipeline"

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

#bb_ratio = []
#for i in range(len(data.x_bb)):
#    indeces = np.argwhere(data.x_bb[i])
#    bb_ratio.append((np.max(indeces[:,0]) / np.min(indeces[:,0])) / (np.max(indeces[:,1]) / np.min(indeces[:,1])))
#bb_ratio = np.array(bb_ratio)
#example_case = np.argmin(np.abs(bb_ratio-1))

predictions = []
expectations = []
predictions_values = []
expectations_values = []
for i in range(len(models)):
    data.configuration.set(**models[i].configuration.__dict__)
    prediction, expectation = models[i].predict_generator(data)
    predictions.append(copy.copy(prediction))
    expectations.append(copy.copy(expectation))

edges = np.argwhere(data.x_bb[0])
index_x_min = int(np.min(edges[:,0]))
index_x_max = int(np.max(edges[:,0]))
index_y_min = int(np.min(edges[:,1]))
index_y_max = int(np.max(edges[:,1]))

x_diff = index_x_max - index_x_min
y_diff = index_y_max - index_y_min

extend_factor = (float(1.5 - 1)) / 2

index_x_min = max(0, index_x_min - int(x_diff * extend_factor))
index_x_max = min(np.shape(data.x_bb[0])[0], index_x_max + int(x_diff * extend_factor))
index_y_min = max(0, index_y_min - int(y_diff * extend_factor))
index_y_max = min(np.shape(data.x_bb[0])[1], index_y_max + int(y_diff * extend_factor))

data_bb_extend = np.zeros(np.shape(data.x_bb[0]))
data_bb_extend[index_x_min:index_x_max, index_y_min:index_y_max] = 1

shape = np.shape(data.x[0])
size = int(np.min(shape))
start = int((np.max(shape) - size) / 2)

if int(np.argmin(shape).flatten()) == 0:
    image = data.x[0][:,start:start+size]
    bb = data.x_bb[0][:,start:start+size]
    bb_extend = data_bb_extend[:,start:start+size]
    crop_mask = bb[index_x_min:index_x_max, index_y_min-start:index_y_max-start]
    crop_image = image[index_x_min:index_x_max, index_y_min-start:index_y_max-start]

    for i in range(len(predictions)):
        predictions[i] = predictions[i][:,start:start+size]
        expectations[i] = expectations[i][:,start:start+size]


else:
    image = data.x[0][start:start+size,:]
    bb = data.x_bb[0][start:start+size,:]
    bb_extend = data_bb_extend[start:start+size,:]
    crop_mask = bb[index_x_min-start:index_x_max-start, index_y_min:index_y_max]
    crop_image = image[index_x_min-start:index_x_max-start, index_y_min:index_y_max]

    for i in range(len(predictions)):
        predictions[i][0] = predictions[i][0][start:start+size,:]
        expectations[i][0] = expectations[i][0][start:start+size,:]

tool_plot.plot_image(image, path_data)
tool_plot.plot_image(bb, path_data)
tool_plot.plot_image(bb_extend, path_data)
tool_plot.plot_image(crop_mask, path_data)
tool_plot.plot_image(crop_image, path_data)

_, rgba_mask = tool_plot.masks2delta2rgba(crop_mask, crop_mask)
tool_plot.plot_masks(crop_image, [rgba_mask], None, None, path_data)

for i in range(len(predictions)):
    _, rgba_mask = tool_plot.masks2delta2rgba(predictions[i][0], predictions[i][0])
    tool_plot.plot_masks(image, [rgba_mask], None, None, path_data)

    _, rgba_mask = tool_plot.masks2delta2rgba(expectations[i][0], predictions[i][0])
    tool_plot.plot_masks(image, [rgba_mask], None, None, path_data)