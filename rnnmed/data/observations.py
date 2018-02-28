import random
import numpy as np

import rnnmed.data as data


class Observations:
    def __init__(self):
        self.__data = []
        self.__labels = []
        self.__data_index = data.IndexLookup()
        self.__label_index = data.IndexLookup()

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
                new_obj = [self.data_index.to_code.get(i) for i, _ in items]
                observation.append(new_obj)
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
            new_visit = [(self.data_index.update(v), 1) for v in visit]
            new_observation.append(new_visit)
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


def one_hot_input_and_output(input_visit, output_visit, n_features):
    """ Vectorize and yield one input and one output pair at a time

    :param input_visit: list of events in visit
    :param output_visit: list of output events
    :param dictionary: dictionary
    :return: a generator

    """
    x = data.vectorize(input_visit, n_features)
    y = data.vectorize2d(output_visit, n_features)
    return np.repeat(x, y.shape[0], axis=0), y


def right_output_visit_generator(observations):
    """Generate visit pairs where the second element of the pair is always
    after (right) of the first element

    :param observations: the observations
    :return: a generator

    """
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits)):
                for j in range(i + 1, len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_a), (j, visit_b)


def left_output_visit_generator(observations):
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits)):
                for j in range(i + 1, len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_b), (j, visit_a)


def visit_generator(observations):
    """ Returns a generator of visits

    :param observations: the observations
    :return: a generator of visits
    """
    for visits in observations:
        for visit in visits:
            yield visit


def visit_pair_generator(observations):
    """
    Generate visit pairs for all possible combinations of visits per
    observation
    :param observations: the observations
    :return: a generator
    """
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits)):
                for j in range(len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_a), (j, visit_b)


def linear_input_generator(observations, n_features):
    for visit in visit_generator(observations):
        yield data.vectorize(visit, n_features)


def simple_input_output_generator(observations, max_skip=2, kind="after"):
    """
    Create a simple input output generator.

    :param observations: the observations
    :param dictionary: the dictionary (translating codes to indicies in the
           resulting one-hot vector)
    :param max_skip: maximum number of future visits to use
    :param kind: output is "before" or "after" the input
    :return: a generator which yields tuples of binary encoded training
    input and outputs
    """
    if kind == "before":
        gen = left_output_visit_generator(observations)
    elif kind == "after":
        gen = right_output_visit_generator(observations)
    elif kind == "both":
        gen = visit_pair_generator(observations)
    else:
        raise ValueError("unknown kind")

    for (i, visit_a), (j, visit_b) in gen:
        if i != j and abs(j - i) <= max_skip:
            yield data.one_hot_input_and_output(visit_a, visit_b,
                                                observations.n_features)


def random_input_output_generator(observations, window_size=2, sample=1):
    """
    Create a input output generator where each input is associated with a
    random output that occurs within at least
    `window_size` visits.

    :param observations: the observations
    :param dictionary: the dictionary
    :param window_size: the window size
    :param sample: the number of outputs to sample from each visit within
    the window
    :return: a generator

    """
    for (i, visit_a), (j, visit_b) in visit_pair_generator(observations):
        if i != j and abs(j - i) <= window_size:
            sample_b = random.sample(visit_b, min(len(visit_b), sample))
            x, y = data.one_hot_input_and_output(visit_a, sample_b,
                                                 observations.n_features)
            yield x, y


def one_hot_observation(observation, n_features, n_visits):
    """Vectorize an observation through time

    :param observation: the observation
    :param dictionary: the dictionary
    :param n_visits: the number of past visits to include
    :returns: a numpy array of shape [n_visits, 1, len(dictionary)]

    """
    arr = np.zeros([n_visits, 1, n_features])
    for i in range(min(n_visits, len(observation))):
        visit = observation[-(i + 1)]
        arr[-(i + 1), 0, :] = data.vectorize(visit, n_features)
    return arr


def time_observation_generator(observations, n_visits=2):
    """A generator that vectorizes observations

    :param observations: the observations
    :param dictionary: the dictionary
    :param n_visits: the number of visits
    :returns: a generator

    """
    for observation, y in zip(observations.data, observations.labels):
        x = one_hot_observation(observation, observations.n_features, n_visits)
        yield x, y


# if __name__ == "__main__":
#     import data.io
#     observations = data.io.read_labeled_visits("test_data/label_synthetic.seq")
#     generator = time_observation_generator(observations, n_visits=4)
#     x, y = generate_time_batch(generator, batch_size=4)
#     print(x)
#     print(y)

#     x, y = generate_input_output_batch(
#         simple_input_output_generator(observations, max_skip=1, kind="both"))
#     for i in range(x.shape[0]):
#         print(x[i, :], y[i, :])
