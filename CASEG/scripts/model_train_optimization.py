from marissa.modules.segmentation import configurations, generators, models
import os

########################################################################################################################
# DEFINE PATHS FOR DATA  ###############################################################################################
########################################################################################################################
paths = [r"", r""]
# just needed if training was interrupted, please consider that the last one might not be finished so set jump_over to the number of done training minus 1
jump_over = 0
magnification_factor = 1.5
########################################################################################################################
########################################################################################################################
########################################################################################################################

current_it = 0

for opt in ["adam", "rmsprop", "sgd"]:
    for path in paths:
        path_train = path + r"\\TRAIN"
        path_val = path + r"\\VALIDATION"
        path_out = path + r"_WEIGHTS_OPT"
        path_plot = path_out + r"\plot"


        # Model BB
        config_BB = {"name": "MMS_UNET_BB",
                     "path_rw": path_out + r"\BB",
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
                      "model_loss": "bce",
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
        if os.path.isfile(path_out + r"\BB\HUMAN_T1MAP_MYOCARDIUM_SAX\MMS_UNET_BB.pickle"):
            model_BB.load()
        else:
            model_BB_gen_train = generators.unet.Setup(path_train, model_BB_config, "TRAINING", ws_dir=path_train)
            model_BB_gen_val = generators.unet.Setup(path_val, model_BB_config, "VALIDATION", ws_dir=path_val)
            model_BB.build()
            model_BB.fit(model_BB_gen_train, None, model_BB_gen_val)

        model_gen_train = generators.unet.Setup(path_train, None, "TRAINING", model_bb=model_BB, ws_dir=path_train)
        model_gen_val = generators.unet.Setup(path_val, None, "VALIDATION", model_bb=model_BB, ws_dir=path_val)

        for loss in ["bce", "log_cosh_dice", "focal_tversky"]:
            for lr_method in ["rop", "linear", "constant", "triangle", "exponential"]:
                for lr in [1e-2, 1e-3]:
                    for model_set in [["refU", 1, False], ["cropU", 1, magnification_factor], ["crinU", 2, magnification_factor]]:
                        config = {"name": "MMS_UNET_" + model_set[0],
                                  "path_rw": path_out + "\\" + "".join([character for character in str(model_set) if character.isalnum()]) + "_" + opt + "_" + loss + "_" + lr_method + "_" + str(int(1000 * lr)) + "e-3",
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
                                  "model_loss": loss,
                                  "model_optimizer": opt,
                                  "model_metrics": ["accuracy", "dice"],
                                  "model_lr_method": lr_method,
                                  "model_lr": lr,
                                  "model_input_size": [256, 256, model_set[1]],
                                  "model_crop": model_set[2],
                                  "model_depth": 6,
                                  "model_convdepth_max": 512,
                                  "data_plot": False #path_plot + r"\crobbU"
                                  }

                        if jump_over <= current_it:
                            model_config = configurations.Setup("UNET", **config)
                            model_gen_train.configuration.set(**config)
                            model_gen_val.configuration.set(**config)
                            model = models.unet.Setup(model_config)
                            model.build()
                            model.fit(model_gen_train, None, model_gen_val)

                        current_it = current_it + 1