import numpy as np

class AlwaysIn:
    """Dummy type to simulate a collection that contains all items"""

    def __contains__(self, item):
        return True


# singleton instance of AlwaysIn
__always_in__ = AlwaysIn()


class IndexLookup:
    def __init__(self):
        self.__dictionary = {}
        self.__reverse_dictionary = {}

    def transform(self, codes):
        return np.array([self.safe_to_index(c) for c in codes])

    def reverse_transform(self, indices):
        return [self.safe_to_code(i) for i in indices]

    @property
    def to_index(self):
        """Return a dictionary of `index => code`

        :returns: a dictionary
        :rtype: dict

        """
        return self.__dictionary

    def safe_to_index(self, code):
        return self.to_index[code]

    @property
    def to_code(self):
        """Return a dictionary of `code => index`

        :returns: a dictionary
        :rtype: dict

        """
        return self.__reverse_dictionary

    def safe_to_code(self, index):
        return self.to_code[index]

    def update(self, code):
        """Update the index with the `code`

        :param code: the code to add to the index
        :returns: the index of the code
        :rtype: int

        """
        index = self.to_index.get(code)
        if index is None:
            index = len(self)
            self.__dictionary[code] = index
            self.__reverse_dictionary[index] = code
        return index

    def __len__(self):
        """Return the length of the index

        :returns: the length
        :rtype: int

        """
        return len(self.__dictionary)


class Observations:
    def __init__(self):
        self.__data = []
        self.__labels = []
        self.__data_index = IndexLookup()
        self.__label_index = IndexLookup()

    @property
    def data_index(self):
        """Return an indexer that can convert between the integer and the truu
        representation of the data items.

        :returns: an index
        :rtype:LookupIndex

        """
        return self.__data_index

    @property
    def label_index(self):
        """Return an indexer that can convert between the integer and the
        true representation of the labels

        :returns: an index
        :rtype: LookupIndex

        """
        return self.__label_index

    @property
    def data(self):
        """Return the data (each element is encoded as an integer from [0,
        self.n_features))

        :returns: the data
        :rtype: list

        """
        return self.__data

    @property
    def labels(self):
        """Returns the labels (encoded from [0, self.n_labels))

        :returns: the labels
        :rtype: list

        """
        return self.__labels

    @property
    def original_labels(self):
        """Return the original labels

        :returns: a list of labels
        :rtype: list

        """
        return list(map(self.label_index.to_code.get, self.labels))

    @property
    def original_data(self):
        """Return the data represented with its original features

        :returns: a list of observations
        :rtype: list

        """
        observations = []
        for obj in self.data:
            observation = []
            for items in obj:
                observation.append(
                    list(map(self.data_index.to_code.get, items)))
            observations.append(observation)
        return observations

    @property
    def n_features(self):
        """Return the number of features in this collection

        :returns: the number of features
        :rtype: int

        """
        return len(self.data_index)

    @property
    def n_labels(self):
        """Return the number of unique labels in this collection

        :returns: the number of labels
        :rtype: int

        """
        return len(self.label_index)

    def add(self, observation, label=None):
        """Add an observation, with an optional, label to this collection.

        :param observation: the observation
        :param label: the label (default: None)
        """
        if label is not None:
            self.__labels.append(self.label_index.update(label))

        new_observation = []
        for visit in observation:
            new_observation.append(list(map(self.data_index.update, visit)))
        self.__data.append(new_observation)

    def __len__(self):
        """Returns the number of observations

        :returns: the number of observations
        :rtype: int

        """
        return len(self.__data)

    def __iter__(self):
        """Returns an iterator of the underlying data.

        :returns: an iterator
        :rtype: int

        """
        return iter(self.__data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            data = self.data[key]
            labels = self.labels[key]
            obs = Observations()
            obs.__data = data
            obs.__labels = labels
            obs.__label_index = self.__label_index
            obs.__data_index = self.__data_index
            return obs
        else:
            return self.data[key]


def __parse_observation(pat, transform, valid_code, code_sep):
    observation = []
    for visit in (vis.strip().split(code_sep) for vis in pat):
        out = []  # set()
        for code in visit:
            code = transform(code) if transform is not None else code
            if valid_code is not None and code in valid_code:
                out.append(code)

        if out:
            observation.append(out)
    return observation


def read_observations(code_path,
                      transform=None,
                      valid_code=None,
                      visit_sep="|",
                      code_sep=" "):
    """Read data from `code_path` optionally truncating the codes to at
    max `code_len`

    Each code is separated by `code_sep` and each visit is separated by a `sep`

    :param code_path: the file path to read from
    :param transform: the transformation function
    :param valid_code: code validator
    :param visit_sep: the visit separator
    :param code_sep: the code separator

    :return: a tuple (data, dictionary, reverse_dictionary)

    """
    valid_code = valid_code or __always_in__
    with open(code_path) as code_file:
        lines = code_file.readlines()
        pats = (line.strip().split(visit_sep) for line in lines)
        observations = Observations()
        for pat in pats:
            observation = __parse_observation(pat, transform, valid_code,
                                              code_sep)
            if observation:
                observations.add(observation)

        if not observations:
            raise AttributeError("no visits")

        return observations


def read_labeled_observations(code_path,
                              visit_sep="|",
                              code_sep=" ",
                              label_sep=","):
    with open(code_path) as code_file:
        lines = code_file.readlines()
        observations = Observations()
        for line in lines:
            label, data = line.strip().split(label_sep)
            observation = __parse_observation(data.strip().split(visit_sep),
                                              None, __always_in__, code_sep)
            if observation:
                observations.add(observation, label)
        return observations

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
        return len(self.data[0]) #  assume all are of same len

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
    
def read_time_series(f, n_dimensions=1, sep="\t"):
    import re
    with f as time_series_file:
        timeseries = Timeseries(n_dimensions)
        lines = time_series_file.readlines()
        for line in lines:
            data = re.split(r"\s+", line.strip())
            x = [float(v) for v in data[1:]]
            y = int(float(data[0]))
            timeseries.add(x, y)
        return timeseries
