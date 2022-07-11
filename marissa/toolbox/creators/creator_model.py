#from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.utils import Sequence
from marissa.toolbox.tools import tool_tensorflow


class Inheritance:
    def __init__(self):
        self.configuration = None
        self.model = None
        return

    def build(self, **kwargs):
        self.configuration.set(**kwargs)
        return True

    def fit(self, x, y=None, validation=None, **kwargs):
        if self.model is None:
            self.build(**kwargs)
        else:
            self.configuration.set(**kwargs)

        self.save()

        #cb = [ModelCheckpoint(filepath=self.configuration.path_rw + "\\" + self.configuration.name + ".hdf5", save_weights_only=self.configuration.fit_save_weights_only, save_best_only=self.configuration.fit_save_best_only, monitor=self.configuration.fit_monitor, mode=self.configuration.fit_mode)]
        cb = [tool_tensorflow.MARISSA_Callback(self.configuration)]
        if y is None:
            if isinstance(x, Sequence):
                results = self.model.fit(x, validation_data = validation, callbacks=cb, epochs=self.configuration.fit_epochs, steps_per_epoch=self.configuration.fit_epoch_length) # epochs=epochs, callbacks=None
            else:
                results = self.model.fit(x, validation_data = validation, callbacks=cb, epochs=self.configuration.fit_epochs, batch_size=self.configuration.fit_batch_size, steps_per_epoch=self.configuration.fit_epoch_length)
        else:
            results = self.model.fit(x, y, validation_data = validation, callbacks=cb, epochs=self.configuration.fit_epochs, batch_size=self.configuration.fit_batch_size, steps_per_epoch=self.configuration.fit_epoch_length)

        if self.configuration.fit_save_last:
            self.model.save_weights(self.configuration.path_rw + "\\" + self.configuration.name + "_last_weights.hdf5")

        return results

    def predict(self, x):
        if self.model is not None:
            value = self.model.predict(x)
        else:
            raise RuntimeError("The model is not build or loaded yet")
        return value

    def load(self, path=None, **kwargs):
        self.configuration.load(path)
        self.load_from_configuration(**kwargs)
        return True

    def load_from_configuration(self, **kwargs):
        if self.configuration.model_weights is not None:
            if self.model is not None:
                try:
                    self.model.set_weights(self.configuration.model_weights)
                except Exception:
                    self.build(**kwargs)
                    self.model.set_weights(self.configuration.model_weights)
            else:
                self.build(**kwargs)
                self.model.set_weights(self.configuration.model_weights)
        return True

    def load_model_weights(self, path):
        self.model.load_weights(path, by_name=True)
        self.configuration.set(model_weights=self.model.get_weights())
        return True

    def save(self, path=None, path_extension=""):
        self.configuration.save(path=path)
        return True
