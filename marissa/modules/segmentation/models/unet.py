import warnings
import numpy as np

from tensorflow.keras.models import Model as TFModel

from marissa.toolbox.tools import tool_general, tool_hadler, tool_tensorflow
from marissa.modules.segmentation import configurations
from marissa.toolbox.creators import creator_model


class Setup(creator_model.Inheritance):
    def __init__(self, configuration=None, **kwargs):
        super().__init__()

        if configuration is None or configuration.model_settings.model_type.upper() != "UNET":
            self.configuration = configurations.Setup("UNET", **kwargs)
        else:
            self.configuration = configuration
            self.configuration.set(**kwargs)

        self.model = None

        self.load_from_configuration()
        return

    def build(self, **kwargs):
        self.configuration.set(**kwargs)

        if self.configuration.model_settings is None:
            raise ValueError("The model_settings is None, but this is not allowed, please define a suitable model")

        input_size = (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1], self.configuration.model_settings.model_input_size[2])
        output_size = self.configuration.model_settings.model_output_size
        config = {"depth": self.configuration.model_settings.model_depth,
                    "droput_rate": self.configuration.model_settings.model_dropout_rate,
                    "convdepth_initial": self.configuration.model_settings.model_convdepth_initial,
                    "convdepth_max": self.configuration.model_settings.model_convdepth_max
                }

        inputs, outputs, _ = tool_tensorflow.model_unet(input_size, output_size, self.configuration.model_settings.model_name, **config)

        model = TFModel(inputs=[inputs], outputs=[outputs], name=self.configuration.model_settings.model_name)
        model_optimizer = tool_tensorflow.get_optimizer(self.configuration.model_settings.model_optimizer, self.configuration.model_settings.model_lr)
        model_loss = tool_tensorflow.get_loss(self.configuration.model_settings.model_loss)
        model_metrics = []
        for metric in self.configuration.model_settings.model_metrics:
            model_metrics.append(tool_tensorflow.get_metric(metric))

        model.compile(optimizer=model_optimizer, loss=model_loss, metrics=model_metrics)
        self.model = model
        return True

    def predict(self, x, show_raw=False, resize=True, lcc=False, contour_level=True):
        if self.model is not None:
            if len(np.shape(x)) > 2:
                myinput = tool_general.array_resize(x[:,:,0], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                myinput = myinput / np.max(myinput)

                for i in range(np.shape(x)[2]):
                    if i > 0:
                        new_xi = np.squeeze(x[:,:,i])
                        try:
                            new_xi = tool_general.mask2polygonalchain(new_xi)
                            resize_x = self.configuration.model_settings.model_input_size[0] / np.shape(x)[0]
                            resize_y = self.configuration.model_settings.model_input_size[1] / np.shape(x)[1]

                            for j in range(len(new_xi)):
                                new_xi[j] = np.apply_along_axis(lambda lmbd: lmbd * np.array([resize_x, resize_y]), 1, new_xi[j])

                            new_xi = tool_general.contour2mask(new_xi, np.zeros((self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1])))
                        except:
                            new_xi = np.zeros((self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]))

                        myinput = np.dstack((myinput, new_xi))

            else:
                myinput = tool_general.array_resize(x, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                if np.max(myinput) != 0:
                    myinput = myinput / np.max(myinput)

            y = self.model.predict(np.expand_dims(myinput.astype("float32"), axis=0))[0, :, :, 0]

            if self.configuration.model_settings.model_predictor.upper() == "BB" or self.configuration.model_settings.model_predictor.upper() == "BOUNDINGBOX":
                if contour_level and np.max(y) >= 0.5 and resize:
                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    y = tool_hadler.getLargestCC(y)

                    indeces = np.argwhere(y)
                    try:
                        index_x_min = np.min(indeces[:,0])
                        index_x_max = np.max(indeces[:,0])
                        index_y_min = np.min(indeces[:,1])
                        index_y_max = np.max(indeces[:,1])

                        resize_x = np.shape(x)[0] / self.configuration.model_settings.model_input_size[0]
                        resize_y = np.shape(x)[1] / self.configuration.model_settings.model_input_size[1]

                        index_x_min = int(round(resize_x * index_x_min))
                        index_x_max = int(round(resize_x * index_x_max))
                        index_y_min = int(round(resize_y * index_y_min))
                        index_y_max = int(round(resize_y * index_y_max))

                        y = np.zeros(np.shape(x))
                        y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                    except:
                        y = np.zeros(np.shape(x))

                if not show_raw:
                    y[y >= 0.5] = 1
                    y[y < 1] = 0

                    if lcc:
                        y = tool_hadler.getLargestCC(y)

                    indeces = np.argwhere(y)
                    try:
                        index_x_min = np.min(indeces[:,0])
                        index_x_max = np.max(indeces[:,0])
                        index_y_min = np.min(indeces[:,1])
                        index_y_max = np.max(indeces[:,1])
                        y = np.zeros(np.shape(y))
                        y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                    except:
                        y = np.zeros(np.shape(y))

            elif self.configuration.model_settings.model_predictor.upper() == "SEGM" or self.configuration.model_settings.model_predictor.upper() == "SEGMENTATION":
                if contour_level and np.max(y) >= 0.5 and resize:
                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    y = tool_general.mask2polygonalchain(y)

                    resize_x = np.shape(x)[0] / self.configuration.model_settings.model_input_size[0]
                    resize_y = np.shape(x)[1] / self.configuration.model_settings.model_input_size[1]

                    for j in range(len(y)):
                        y[j] = np.apply_along_axis(lambda lmbd: lmbd * np.array([resize_x, resize_y]), 1, y[j])

                    y = tool_general.contour2mask(y, np.zeros(np.shape(x)[:2]))

                elif resize:
                    y = tool_general.array_resize(y, np.shape(x)[:2], anti_aliasing=True, normalize=False)

                if not show_raw:
                    y[y >= 0.5] = 1
                    y[y < 1] = 0

                    if lcc:
                        y = tool_hadler.getLargestCC(y)

            else:
                y = None

        else:
            warnings.warn("The model is not build or loaded yet")
            y = None

        return y

    def predict_generator(self, x, show_raw=False, resize=True, lcc=False, contour_level=True):
        '''
        Run prediction of data given as generator
        :param x: generator of data
        :param show_raw: show raw output of prediction (rezise should be set to False if True)
        :param resize: resize output to original image size
        :param lcc: run largest connected component on result
        :param contour_level: do resize on contour instead of pixelmap level (default True (preferred) for loss-less resizing)
        :return: tuple of lists (prediction and expectation)
        '''
        if self.model is not None:
            prediction = []
            expectation = []

            for i in range(len(x.x)):
                gen_x, gen_y = x.get_data(i)
                pred_y = np.squeeze(self.model.predict(np.expand_dims(gen_x, axis=0))[0, :, :, 0])

                #from marissa.toolbox.tools import tool_plot
                #_, masks = tool_plot.masks2delta2rgba(gen_y, pred_y)
                #tool_plot.plot_masks(x.x[i], [masks], ["TEST"], export=True)

                # BOUNDING BOX PREDICTION
                if self.configuration.model_settings.model_predictor.upper() == "BB" or self.configuration.model_settings.model_predictor.upper() == "BOUNDINGBOX":
                    if contour_level and np.max(pred_y) >= 0.5 and resize:
                        pred_y[pred_y >= 0.5] = 1
                        pred_y[pred_y < 1] = 0
                        pred_y = tool_hadler.getLargestCC(pred_y)

                        indeces = np.argwhere(pred_y)
                        try:
                            index_x_min = np.min(indeces[:,0])
                            index_x_max = np.max(indeces[:,0])
                            index_y_min = np.min(indeces[:,1])
                            index_y_max = np.max(indeces[:,1])

                            resize_x = np.shape(x.x[i])[0] / self.configuration.model_settings.model_input_size[0]
                            resize_y = np.shape(x.x[i])[1] / self.configuration.model_settings.model_input_size[1]

                            index_x_min = int(round(resize_x * index_x_min))
                            index_x_max = int(round(resize_x * index_x_max))
                            index_y_min = int(round(resize_y * index_y_min))
                            index_y_max = int(round(resize_y * index_y_max))

                            pred_y = np.zeros(np.shape(x.x[i]))
                            pred_y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                        except:
                            pred_y = np.zeros(np.shape(gen_y))

                        indeces = np.argwhere(x.y[i])
                        try:
                            index_x_min = int(np.min(indeces[:,0]))
                            index_x_max = int(np.max(indeces[:,0]))
                            index_y_min = int(np.min(indeces[:,1]))
                            index_y_max = int(np.max(indeces[:,1]))

                            gen_y = np.zeros(np.shape(x.x[i]))
                            gen_y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                        except:
                            gen_y = np.zeros(np.shape(gen_y))

                    if not show_raw:
                        pred_y[pred_y >= 0.5] = 1
                        pred_y[pred_y < 1] = 0

                        if lcc:
                            pred_y = tool_hadler.getLargestCC(pred_y)

                        indeces = np.argwhere(pred_y)
                        try:
                            index_x_min = int(np.min(indeces[:,0]))
                            index_x_max = int(np.max(indeces[:,0]))
                            index_y_min = int(np.min(indeces[:,1]))
                            index_y_max = int(np.max(indeces[:,1]))
                            pred_y = np.zeros(np.shape(pred_y))
                            pred_y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                        except:
                            pred_y = np.zeros(np.shape(pred_y))

                # SEGMENTATION PREDICTION
                elif self.configuration.model_settings.model_predictor.upper() == "SEGM" or self.configuration.model_settings.model_predictor.upper() == "SEGMENTATION":
                    if self.configuration.model_settings.model_crop:
                        indeces = np.argwhere(np.squeeze(x.x_bb[i]) > 0.5)
                        if len(indeces) > 0:
                            indeces = np.array([[np.min(indeces[:,0]), np.min(indeces[:,1])], [np.min(indeces[:,0]), np.max(indeces[:,1])], [np.max(indeces[:,0]), np.max(indeces[:,1])], [np.max(indeces[:,0]), np.min(indeces[:,1])]])
                            edges = np.array(indeces).astype("float32")
                            index_x_min = int(np.min(edges[:,0]))
                            index_x_max = int(np.max(edges[:,0]))
                            index_y_min = int(np.min(edges[:,1]))
                            index_y_max = int(np.max(edges[:,1]))

                            x_diff = index_x_max - index_x_min
                            y_diff = index_y_max - index_y_min

                            extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                            index_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                            index_x_max = min(np.shape(x.x[i])[0], index_x_max + int(x_diff * extend_factor))
                            index_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                            index_y_max = min(np.shape(x.x[i])[1], index_y_max + int(y_diff * extend_factor))
                        else:
                            index_x_min = 0
                            index_x_max = np.shape(x.x[i])[0]
                            index_y_min = 0
                            index_y_max = np.shape(x.x[i])[1]
                    else:
                        index_x_min = 0
                        index_x_max = np.shape(x.x[i])[0]
                        index_y_min = 0
                        index_y_max = np.shape(x.x[i])[1]

                    if contour_level and np.max(pred_y) >= 0.5 and resize:
                        pred_y[pred_y >= 0.5] = 1
                        pred_y[pred_y < 1] = 0
                        pred_y = tool_general.mask2polygonalchain(pred_y)

                        resize_x = (index_x_max-index_x_min) / self.configuration.model_settings.model_input_size[0]
                        resize_y = (index_y_max-index_y_min) / self.configuration.model_settings.model_input_size[1]

                        for j in range(len(pred_y)):
                            pred_y[j] = np.apply_along_axis(lambda lmbd: lmbd * np.array([resize_x, resize_y]) + np.array([index_x_min, index_y_min]), 1, pred_y[j])

                        pred_y = tool_general.contour2mask(pred_y, x.x[i])
                        gen_y = x.y[i]

                    elif resize:
                        pred_y = tool_general.array_resize(pred_y, (index_x_max-index_x_min, index_y_max-index_y_min), anti_aliasing=True, normalize=False)
                        mask = np.zeros(np.shape(x.x[i]))
                        mask[index_x_min:index_x_max, index_y_min:index_y_max] = pred_y
                        pred_y = mask
                        gen_y = x.y[i]

                    if not show_raw:
                        pred_y[pred_y >= 0.5] = 1
                        pred_y[pred_y < 1] = 0

                        if lcc:
                            pred_y = tool_hadler.getLargestCC(pred_y)
                else:
                    gen_y = None
                    pred_y = None

                #from marissa.toolbox.tools import tool_plot
                #_, masks = tool_plot.masks2delta2rgba(gen_y, pred_y)
                #tool_plot.plot_masks(x.x[i], [masks], ["TEST"], export=True)

                expectation.append(gen_y)
                prediction.append(pred_y)
        else:
            warnings.warn("The model is not build or loaded yet")
            expectation = None
            prediction = None
        return prediction, expectation
