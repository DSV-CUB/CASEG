import os
from marissa.modules.segmentation.configurations import unet
from marissa.toolbox import creators


class Setup(creators.creator_configuration.Inheritance):
    def __init__(self, model_type=None, **kwargs):
        super().__init__()

        #General
        self.name = "MMS"
        self.path = os.path.dirname(os.path.realpath(__file__))

        #Data
        self.data_rescale = True
        self.data_representation = False
        self.data_segmentation_level = "CONTOUR" # or mask
        self.data_plot = False # or True (show inline) or PATH_TO_SAVE (save images in path)

        #Model
        self.model_weights = None

        #Fit
        self.fit_epochs = 1000
        self.fit_epoch_length = None
        self.fit_batch_size = 10
        self.fit_save_last = False
        self.fit_save_weights_only = True
        self.fit_save_best_only=True
        self.fit_monitor = "loss"
        self.fit_mode = "min"
        self.fit_early_stopping = True
        self.fit_early_stopping_min_delta = 0
        self.fit_early_stopping_patience = 50
        self.fit_early_stopping_stopped_epoch = 0
        self.fit_log = {}

        self.model_settings = None

        self.set(**kwargs)
        if not model_type is None:
            self.set_model(model_type, **kwargs)
        return

    def set_model(self, type, **kwargs):
        if type.upper() == "UNET":
            self.model_settings = unet.Setup(**kwargs)
        return

    def set(self, **kwargs):
        super().set(**kwargs)
        if not self.model_settings is None:
            self.model_settings.set(**kwargs)
        return
