from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization

#CALLBACKS
class MARISSA_Callback(Callback):
    def __init__(self, config):
        super().__init__()
        self.configuration = config

        self.best = None
        self.mode = None
        self.wait = 0
        return

    def on_train_begin(self, logs=None):
        if str(self.configuration.fit_mode).upper() == "MAX" or (isinstance(self.configuration.fit_mode, bool) and self.configuration.fit_mode == True):
            self.best = float('-inf')
            self.mode = np.greater
        else:
            self.best = float('inf')
            self.mode = np.less

        self.wait = 0
        self.configuration.fit_early_stopping_stopped_epoch = 0
        K.set_value(self.model.optimizer.lr, K.get_value(self.scheduler(0)))
        return

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.configuration.fit_monitor]

        if self.configuration.fit_early_stopping:
            if self.mode(metric_value, self.best) and abs(metric_value - self.best) > self.configuration.fit_early_stopping_min_delta:
                self.wait = 0
                self.best = metric_value
                self.configuration.fit_log["best_monitor"] = metric_value
                self.configuration.fit_log["loss"] = logs["loss"]
                for m in self.configuration.model_settings.model_metrics:
                    self.configuration.fit_log[m] = logs["metric_" + m]
                self.configuration.model_weights = self.model.get_weights()
                self.configuration.save()
            else:
                self.wait = self.wait + 1

                if self.wait > self.configuration.fit_early_stopping_patience:
                    self.configuration.fit_early_stopping_stopped_epoch = epoch
                    self.configuration.save()
                    self.model.stop_training = True
                    self.model.set_weights(self.configuration.model_weights)
                    return
        else:
            self.wait = self.wait + 1
            if self.mode(metric_value, self.best):
                self.wait = 0
                self.best = metric_value
                self.configuration.fit_log["best_monitor"] = metric_value
                self.configuration.fit_log["loss"] = logs["loss"]
                for m in self.configuration.model_settings.model_metrics:
                    self.configuration.fit_log[m] = logs["metric_" + m]
                self.configuration.model_weights = self.model.get_weights()
                self.configuration.save()

        K.set_value(self.model.optimizer.lr, K.get_value(self.scheduler(epoch)))

        return

    def on_train_end(self, logs=None):
        if self.configuration.fit_early_stopping_stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.configuration.fit_early_stopping_stopped_epoch + 1))
            self.configuration.fit_log["best_weights_epoch"] = self.configuration.fit_early_stopping_stopped_epoch - self.configuration.fit_early_stopping_patience
        else:
            self.configuration.fit_log["best_weights_epoch"] = self.configuration.fit_epochs
        return

    def scheduler(self, epoch):
        lr = self.model.optimizer.lr

        if self.configuration.model_settings.model_lr_method == "rop": #reduce on plateau
            if (self.configuration.fit_early_stopping and self.wait == int(self.configuration.fit_early_stopping_patience/2)):
                lr = 0.5 * lr
            elif (not self.configuration.fit_early_stopping and self.wait == 20):
                lr = 0.5 * lr
                self.wait = 0
        elif self.configuration.model_settings.model_lr_method == "linear":
            lr = (1 - (epoch / (self.configuration.fit_epochs + 1))) * lr
        elif self.configuration.model_settings.model_lr_method == "triangular":
            # https://arxiv.org/pdf/1506.01186.pdf
            stepsize = 5
            lr_max = self.configuration.model_settings.model_lr #self.configuration.fit_lr_max
            lr_min = self.configuration.model_settings.model_lr * 1e-4 #self.configuration.fit_lr_min
            lr_diff = lr_max - lr_min

            cycle = np.floor(1 + (epoch / (2*stepsize)))
            x = np.abs((epoch/stepsize) - (2*cycle) + 1)
            lr = lr_min + (lr_diff) * np.max(0, (1-x))
        elif self.configuration.model_settings.model_lr_method == "triangular2":
            stepsize = 5
            lr_max = self.configuration.model_settings.model_lr #self.configuration.fit_lr_max
            lr_min = self.configuration.model_settings.model_lr * 1e-4 #self.configuration.fit_lr_min
            lr_diff = lr_max - lr_min

            cycle_epoch = np.floor(epoch / (2*stepsize))
            lr_diff = lr_diff / (1.5 ** cycle_epoch)

            cycle = np.floor(1 + (epoch / (2*stepsize)))
            x = np.abs((epoch/stepsize) - (2*cycle) + 1)
            lr = lr_min + lr_diff * np.max((0, (1-x)))
        elif self.configuration.model_settings.model_lr_method == "exponential":
            decay = 0.05
            lr = self.configuration.model_settings.model_lr * np.exp(-epoch * decay)

        return lr


# LOSSES
def loss_dice(y_true, y_pred):
    y_true_cast = tf.cast(y_true, tf.float32)
    y_pred_cast = tf.cast(y_pred, tf.float32)
    num = 2 * tf.reduce_sum(y_true_cast * y_pred_cast) + 1
    denom = tf.reduce_sum(y_true_cast + y_pred_cast) + 1
    return 1 - num / denom
    #return 1 - (metric_dice(y_true, y_pred))


def loss_log_cosh_dice(y_true, y_pred):
    return tf.math.log(tf.math.cosh(loss_dice(y_true, y_pred)))

def loss_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1) / (union + 1)
    return 1 - iou

def loss_tversky(y_true, y_pred, alpha=0.5, beta=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1-y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1-y_pred))
    tversky = (tp + 1) / (tp + alpha*fp + beta*fn + 1)
    return 1 - tversky

def loss_focal_tversky(y_true, y_pred, alpha=0.3, beta=0.7, gamma=0.75):
    ltv = loss_tversky(y_true, y_pred, alpha, beta)
    return tf.math.pow(ltv, gamma)

def combined_loss(y_true, y_predict, loss_list = []):
    loss = 0

    losses = []
    for loss_item in loss_list:
        losses.append(eval(loss_item + "(y_true, y_predict)"))

    if len(losses) > 0:
        loss = tf.reduce_sum(tf.convert_to_tensor(losses)) / len(losses)

    return loss



# METRICS
def metric_dice(y_true, y_pred):
    #from marissa.toolbox import tools
    #y_true_cast = tf.cast(y_true, tf.float32)
    #y_true_cast[y_true_cast >= 0.5] = 1
    #y_true_cast[y_true_cast < 1] = 0
    #y_true_cast = tf.cast(y_true_cast, tf.bool)

    #y_pred_cast = np.copy(y_pred.eval(session=tf.compat.v1.Session()))
    #y_pred_cast[y_pred_cast >= 0.5] = 1
    #y_pred_cast[y_pred_cast < 1] = 0
    #y_pred_cast = tools.tool_hadler.getLargestCC(y_pred_cast)

    #result = tools.tool_general.get_metric_from_masks(y_true_cast, y_pred_cast, "DSC")

    y_true_cast = tf.cast(tf.math.round(tf.cast(y_true, tf.float32)), tf.int32)
    y_pred_cast = tf.cast(tf.math.round(tf.cast(y_pred, tf.float32)), tf.int32)
    num = 2 * tf.cast(tf.reduce_sum(y_true_cast * y_pred_cast), tf.float32) + 1
    denom = tf.cast(tf.reduce_sum(y_true_cast + y_pred_cast), tf.float32) + 1
    result = num/ denom
    return result

def metric_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.math.round(tf.cast(y_true, tf.float32)), tf.int32)
    y_pred = tf.cast(tf.math.round(tf.cast(y_pred, tf.float32)), tf.int32)
    equal = tf.reduce_sum(y_true*y_pred + (1-y_true)*(1-y_pred))
    nr_entries = tf.reduce_prod(tf.shape(y_true))
    return equal / nr_entries

# Functions:
def get_optimizer(optimizer, lr):
    if optimizer == "sgd":
        function = SGD(learning_rate=lr, decay=0.0005, momentum=0.9)
    elif optimizer == "adam":
        function = Adam(learning_rate=lr, clipnorm=0.001)
    elif optimizer == "rmsprop":
        function = RMSprop(learning_rate=lr)
    else:
        function = "sgd"
    return function

def get_loss(loss, **kwargs):
    if loss == "binary_crossentropy" or loss == "bce":
        function = tf.keras.losses.BinaryCrossentropy()
    elif loss == "dice":
        function = lambda y_true, y_predict: loss_dice(y_true, y_predict)
    elif loss == "log_cosh_dice":
        function = lambda y_true, y_predict: loss_log_cosh_dice(y_true, y_predict)
    elif loss == "iou":
        function = lambda y_true, y_predict: loss_iou(y_true, y_predict)
    elif loss == "tversky":
        function = lambda y_true, y_predict: loss_tversky(y_true, y_predict, **kwargs)
    elif loss == "focal_tversky":
        function = lambda y_true, y_predict: loss_tversky(y_true, y_predict, **kwargs)
    else:
        function = None
    return function

def get_metric(metric):
    if metric == "accuracy":
        function = metric_accuracy
    elif metric == "dice":
        function = metric_dice
    else:
        function = None
    return function

# Models

def model_unet(input_tuple, output_size, name, **kwargs):
    depth = kwargs.get("depth", 6)
    droput_rate = kwargs.get("dropoutrate", 0.2)
    convdepth_initial = kwargs.get("convdepth_initial", 32)
    convdepth_max = kwargs.get("convdepth_max", convdepth_initial * 2 ** (depth+1))

    layers = []

    x = Input(input_tuple, name=name + "_input")
    layers.append(x)

    for i in range(depth+1):
        convdepth = min(convdepth_initial*(2**i), convdepth_max)

        if i >0:
            x = MaxPooling2D((2, 2), name=name + "_maxpool_" + str(i+ 1) + "_2") (x)
        x = Conv2D(convdepth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name=name + "_conv2d_" + str(i+ 1) + "_1") (x)
        x = BatchNormalization(name=name + "_batchnorm_" + str(i+ 1) + "_1")(x)
        x = Dropout(droput_rate, name=name + "_dropout_" + str(i+ 1) + "_1") (x)
        x = Conv2D(convdepth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name=name + "_conv2d_" + str(i+ 1) + "_2") (x)
        x = BatchNormalization(name=name + "_batchnorm_" + str(i+ 1) + "_2")(x)
        layers.append(x)

    for j in range(depth):
        convdepth = min(convdepth_initial*(2**(i-j-1)), convdepth_max)

        x = Conv2DTranspose(convdepth, (2, 2), strides=(2, 2), padding='same', name=name + "_conv2dt_" + str(i+1+j+1) + "_1") (x)
        x = concatenate([x, layers[(depth-j)]], name=name + "_concat_" + str(i+1+j+1) + "_1")
        x = Conv2D(convdepth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name=name + "_conv2d_" + str(i+1+j+1) + "_2") (x)
        x = BatchNormalization(name=name + "_batchnorm_" + str(i+1+j+1) + "_2")(x)
        x = Dropout(droput_rate, name=name + "_dropout_" + str(i+1+j+1) + "_2") (x)
        x = Conv2D(convdepth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name=name + "_conv2d_" + str(i+1+j+1) + "_3") (x)
        x = BatchNormalization(name=name + "_batchnorm_" + str(i+1+j+1) + "_3")(x)
        layers.append(x)

    x = Conv2D(output_size, (1, 1), activation='sigmoid', name=name + "_conv2d_out") (x)
    layers.append(x)

    return layers[0], layers[-1], layers