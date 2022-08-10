import os
import copy
import numpy as np
from marissa.modules.segmentation import test
from marissa.modules.segmentation import models as mmsmodels, generators as mmsgenerators

########################################################################################################################
# DEFINE SETTINGS AND PATH #############################################################################################
########################################################################################################################

# DEFINE WHICH TESTING TO DO
do_test = False # TEST THE MODELS IN GENERAL
do_test_native_pca = True # TEST THE MODELS BUT SEPARATED FOR NATIVE AND PCA

# DEFINE PATHS
path_weights = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX_Paper_CASEG_ONLY"
path_data = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX\TEST_BASMV"
path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\RESULTS\Paper"

########################################################################################################################
########################################################################################################################
########################################################################################################################

# TEST MODELS
if do_test:
    test.test_bb_unet(path_data, path_weights, path_out, path_data, name="BB_SAX_Traindata_BASMV_Testdata_UNET6")
    test.test_model_unet(path_data, path_weights, path_out, ws_dir=path_data, mask_mode="RASTERIZE", reference=0, name="Model_SAX_Traindata_BASMV_Testdata_UNET6")
if do_test_native_pca:

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

    #import tensorflow as tf
    #tf.keras.utils.plot_model(models[-1].model, to_file=r"C:\Users\Omen\Desktop\model2.png", show_shapes=True, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, layer_range=None, show_layer_activations=False)

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
            try:
                prediction_value.append(np.mean(data.x[j][np.where(prediction[j])]))
            except:
                prediction_value.append(-1)
            expectation_value.append(np.mean(data.x[j][np.where(expectation[j])]))
        predictions_values.append(copy.copy(prediction_value))
        expectations_values.append(copy.copy(expectation_value))

    indeces_pca = np.argwhere(np.array(expectations_values[0]) < 800).flatten()
    indeces_native = np.argwhere(np.array(expectations_values[0]) > 800).flatten()

    data_native = copy.copy(data)
    data_pca = copy.copy(data)

    data_native.x = [data_native.x[index] for index in indeces_native]
    data_native.x_bb = [data_native.x_bb[index] for index in indeces_native]
    data_native.y = [data_native.y[index] for index in indeces_native]
    data_native.y_contour = [data_native.y_contour[index] for index in indeces_native]
    data_native.information = [data_native.information[index] for index in indeces_native]
    data_native.pixel_spacing = [data_native.pixel_spacing[index] for index in indeces_native]
    data_native.indeces = np.arange(len(data_native.x))
    data_native.gen = data_native.generator()

    data_pca.x = [data_pca.x[index] for index in indeces_pca]
    data_pca.x_bb = [data_pca.x_bb[index] for index in indeces_pca]
    data_pca.y = [data_pca.y[index] for index in indeces_pca]
    data_pca.y_contour = [data_pca.y_contour[index] for index in indeces_pca]
    data_pca.information = [data_pca.information[index] for index in indeces_pca]
    data_pca.pixel_spacing = [data_pca.pixel_spacing[index] for index in indeces_pca]
    data_pca.indeces = np.arange(len(data_pca.x))
    data_pca.gen = data_pca.generator()

    test.test_model_unet(data_native, models, path_out, call=False, mask_mode="RASTERIZE", reference=0, name="Model_SAX_Traindata_BASMV_Testdata_UNET6_native")
    test.test_model_unet(data_pca, models, path_out, call=False, mask_mode="RASTERIZE", reference=0, name="Model_SAX_Traindata_BASMV_Testdata_UNET6_pca")
    test.test_bb_unet(data_native, model_bb, path_out, call=False, name="BB_SAX_Traindata_BASMV_Testdata_UNET6_native")
    test.test_bb_unet(data_pca, model_bb, path_out, call=False, name="BB_SAX_Traindata_BASMV_Testdata_UNET6_pca")