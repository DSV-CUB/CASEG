from marissa.modules.segmentation import configurations, generators, models
import os

########################################################################################################################
# DEFINE PATHS FOR DATA AND OUTPUT #####################################################################################
########################################################################################################################

# for each dicom a pickle file with the ground truth contours must exist. both need the SOPInstanceUID as name
path_train = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TRAIN"
path_val = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\VALIDATION"
path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper"
path_bb_weights =r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\6 - Analysis\WEIGHTS\UNET6_SAX - Paper\UNET_BB"

########################################################################################################################
########################################################################################################################
########################################################################################################################

# RUN TRAINING THE MODELS

path_plot = path_out + r"\plot"
mask_mode = "RASTERIZE"

# Model BB
config_BB = {"name": "MMS_UNET_BB",
             "path_rw": path_bb_weights,
             "tissue": "MYOCARDIUM",
             "view": "SAX",
             "measure": "T1MAP",
             "examination": "HUMAN",
              "fit_epochs": 1000,
              "fit_batch_size": 10,
              "fit_monitor": "val_metric_dice",
              "fit_mode": "max",
              "fit_early_stopping": True,
              "fit_early_stopping_patience": 50,
              "fit_early_stopping_min_delta": 0,
              "model_predictor": "BB",
              "model_loss": "log_cosh_dice",
              "model_metrics": ["accuracy", "dice"],
              "model_lr_method": "rop",
              "model_lr": 1e-3,
              "model_input_size": [256, 256, 1],
              "model_crop": False,
              "model_depth": 6,
              "data_plot": False #path_plot + r"\BB"
            }


model_BB_config = configurations.Setup("UNET", **config_BB)
model_BB = models.unet.Setup(model_BB_config)
if os.path.isfile(path_bb_weights + r"\HUMAN_T1MAP_MYOCARDIUM_SAX\MMS_UNET_BB.pickle"):
    model_BB.load()
else:
    model_BB_gen_train = generators.unet.Setup(path_train, model_BB_config, "TRAINING", ws_dir=path_train, mask_mode=mask_mode)
    model_BB_gen_val = generators.unet.Setup(path_val, model_BB_config, "VALIDATION", ws_dir=path_val, mask_mode=mask_mode)
    model_BB.build()
    model_BB.fit(model_BB_gen_train, None, model_BB_gen_val)

model_gen_train = generators.unet.Setup(path_train, None, "TRAINING", model_bb=model_BB, ws_dir=path_train, mask_mode=mask_mode)
model_gen_val = generators.unet.Setup(path_val, None, "VALIDATION", model_bb=model_BB, ws_dir=path_val, mask_mode=mask_mode)

for model_set in [["refU", 1, False], ["bbU", 2, False], ["cropU", 1, 1.5], ["crinU", 2, 1.5]]:
    config = {"name": "MMS_UNET_" + "".join([character for character in str(model_set) if character.isalnum()]),
              "path_rw": path_out + "\\" + model_set[0],
              "tissue": "MYOCARDIUM",
              "view": "SAX",
              "measure": "T1MAP",
              "examination": "HUMAN",
              "fit_epochs": 1000,
              "fit_batch_size": 10,
              "fit_monitor": "val_metric_dice",
              "fit_mode": "max",
              "fit_early_stopping": True,
              "fit_early_stopping_patience": 50,
              "fit_early_stopping_min_delta": 0,
              "model_loss": "log_cosh_dice",
              "model_metrics": ["accuracy", "dice"],
              "model_lr_method": "rop",
              "model_lr": 1e-3,
              "model_input_size": [256, 256, model_set[1]],
              "model_crop": model_set[2],
              "model_depth": 6,
              "model_convdepth_max": 512,
              "data_plot": False#path_plot + r"\refU"
              }

    model_config = configurations.Setup("UNET", **config)
    model_gen_train.configuration.set(**config)
    model_gen_val.configuration.set(**config)
    model = models.unet.Setup(model_config)
    model.build()
    model.fit(model_gen_train, None, model_gen_val)