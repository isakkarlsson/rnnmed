import numpy as np
from .utils import IndexLookup


class Timeseries:
    def __init__(self, n_dimensions):
        """

        """
        self.__n_dimensions = n_dimensions
        self.__label_index = IndexLookup()
        self.__data = []
        self.__labels = []

    @property
    def n_dimensions(self):
        return self.__n_dimensions

    @property
    def n_timesteps(self):
        return len(self.data[0])  # assume all are of same len

    @property
    def n_labels(self):
        return len(set(self.labels))

    @property
    def data(self):
        return self.__data

    @property
    def labels(self):
        return self.__labels

    @property
    def label_index(self):
        return self.__label_index

    def add(self, timeseries, label):
        self.__data.append(timeseries)
        self.__labels.append(self.label_index.update(label))

    def __setitem__(self, key, item):
        x, y = item
        self.__data[key] = x
        self.__labels[key] = y

    def __getitem__(self, key):
        return self.__data[key], self.__labels[key]

    def __iter__(self):
        for i in range(len(self)):
            yield self.data[i], self.labels[i]

    def __len__(self):
        return len(self.data)


def timeseries_generator(timeseries):
    """Creates a generator that outputs a single time series and an output

    :param timeseries: the time series to generate over
    :yields: a numpy array of shape ``[timeseries.n_timesteps, 1, timeseries.n_dimensions]`` and an integer label
    :rtype:

    """
    for x, y in timeseries:
        x = np.array(x).reshape(timeseries.n_timesteps, 1,
                                timeseries.n_dimensions)
        yield x, y
