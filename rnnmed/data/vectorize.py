import itertools
import random

import numpy as np


def vectorize_visit(visit, n_features):
    """Vectorize a visit.

    :param visit: the visit
    :param n_features: the number of features
    :returns: array of shape [1, n_features]
    :rtype: np.array

    """
    x = np.zeros(shape=(1, n_features))
    for where in visit:
        x[0, where] = 1
    return x


def vectorize_outputs(visit, n_features):
    """Vectorize the visit as outputs. Each element of a visit is a unique
    output row.§

    :param visit:  the visit
    :param n_features: the number of features in the representation
    :returns: array of shape [len(visit), n_features]
    :rtype: np.array

    """
    x = np.zeros(shape=(len(visit), n_features))
    for row, col in enumerate(visit):
        x[row, col] = 1
    return x


def vectorize_input_and_output(input_visit, output_visit, n_features):
    """ Vectorize and yield one input and one output pair at a time

    :param input_visit: list of events in visit
    :param output_visit: list of output events
    :param dictionary: dictionary
    :return: a generator

    """
    x = vectorize_visit(input_visit, n_features)
    y = vectorize_outputs(output_visit, n_features)
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


def linear_input_generator(observations, dictionary):
    for visit in visit_generator(observations):
        yield vectorize_visit(visit, dictionary)


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
            yield vectorize_input_and_output(visit_a, visit_b,
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
            x, y = vectorize_input_and_output(
                visit_a, sample_b, observations.n_features)
            yield x, y


def vectorize_observation(observation, n_features, n_visits):
    """Vectorize an observation

    :param observation: the observation
    :param dictionary: the dictionary
    :param n_visits: the number of past visits to include
    :returns: a numpy array of shape [n_visits, 1, len(dictionary)]
    
    """
    arr = np.zeros([n_visits, 1, n_features])
    for i in range(min(n_visits, len(observation))):
        visit = observation[-(i + 1)]
        arr[-(i + 1), 0, :] = vectorize_visit(visit, n_features)
    return arr


def time_observation_generator(observations, n_visits=2):
    """A generator that vectorizes observations

    :param observations: the observations
    :param dictionary: the dictionary
    :param n_visits: the number of visits
    :returns: a generator

    """
    for observation, y in zip(observations.data, observations.labels):
        x = vectorize_observation(observation, observations.n_features,
                                  n_visits)
        yield x, y


if __name__ == "__main__":
    import data.io
    observations = data.io.read_labeled_visits("test_data/label_synthetic.seq")
    generator = time_observation_generator(observations, n_visits=4)
    x, y = generate_time_batch(generator, batch_size=4)
    print(x)
    print(y)

    x, y = generate_input_output_batch(
        simple_input_output_generator(observations, max_skip=1, kind="both"))
    for i in range(x.shape[0]):
        print(x[i, :], y[i, :])
