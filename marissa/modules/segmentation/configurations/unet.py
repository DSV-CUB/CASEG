from marissa.toolbox import creators

class Setup(creators.creator_configuration.Inheritance):
    def __init__(self, **kwargs):
        super().__init__()

        self.model_type = "UNET"
        self.model_predictor = "SEGM" # BB
        self.model_name = "MMS_unet"

        self.model_input_size = [256, 256, 1]
        self.model_output_size = 1
        self.model_optimizer = "adam"
        self.model_loss = "loss_binary_crossentropy"
        self.model_lr = 1e-3
        self.model_lr_method = "constant"
        self.model_metrics = ["accuracy"]

        self.model_depth = 6
        self.model_dropout_rate = 0.2
        self.model_convdepth_initial = 32
        self.model_convdepth_max = 2048

        self.model_crop = False

        self.set(**kwargs)
        return