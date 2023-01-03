import os
import copy
from datetime import datetime
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators
from marissa.toolbox import tools

########################################################################################################################
# DEFINE PATHS FOR DATA AND OUTPUT #####################################################################################
########################################################################################################################

path_data = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX\TEST_BASMV"
path_weights = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX_Paper"
path_out = r"C:\Users\CMRT\Desktop"

########################################################################################################################
########################################################################################################################
########################################################################################################################

# load models
models = []
model_bb = None
models_names = []

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
                models_names.append(model.configuration.name)


data = mmsgenerators.unet.Setup(path_data, None, "TEST",  model_bb=model_bb, ws_dir=False, mask_mode="RASTERIZE")

dir_out = "CASEG_prediction_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(path_out, dir_out), exist_ok=True)
for i in range(len(models)):
    data.configuration = models[i].configuration
    prediction, _ = models[i].predict_generator(data)

    os.makedirs(os.path.join(path_out, dir_out, models_names[i]), exist_ok=True)

    for j in range(len(prediction)):
        _, mask = tools.tool_plot.masks2delta2rgba(prediction[j], prediction[j])
        tools.tool_plot.plot_masks(data.x[j], [mask], [""], None, export=os.path.join(path_out, dir_out, models_names[i]))