import numpy as np
import math
from tensorflow.keras.utils import Sequence


class Inheritance(Sequence):

    def __init__(self):
        self.configuration = None
        self.x = None
        self.y = None
        self.indeces = None
        self.gen = self.generator()
        self.generator_type = "TRAINING"
        return

    def __len__(self):
        return math.ceil(len(self.x) / self.configuration.fit_batch_size)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        batch = [next(self.gen) for _ in range(self.configuration.fit_batch_size)]
        batch_x = np.array([x for (x, y) in batch])
        batch_y = np.array([y for (x, y) in batch])
        return batch_x, batch_y

    def generator(self):
        while True:
            for i in self.indeces:
                yield self.get_data(i)

    def on_epoch_end(self):
        np.random.shuffle(self.indeces)
        self.gen = self.generator()
        return

    def get_data(self, i):
        x, y = self.x[i], self.y[i]
        return x, y