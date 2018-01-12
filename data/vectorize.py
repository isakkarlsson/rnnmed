import itertools
import random

import numpy as np


def _vectorize_visit(visit, dictionary):
    x = np.zeros(shape=(1, len(dictionary)))
    for code in visit:
        where = dictionary[code]
        x[0, where] = 1
    return x


def _vectorize_outputs(visit, dictionary):
    x = np.zeros(shape=(len(visit), len(dictionary)))
    for i, code in enumerate(visit):
        where = dictionary[code]
        x[i, where] = 1
    return x


def _vectorize_input_and_output(input_visit, output_visit, dictionary):
    """
    Vectorize and yield one input and one output pair at a time
    :param input_visit: list of events in visit
    :param output_visit: list of output events
    :param dictionary: dictionary
    :return: a generator
    """
    x = _vectorize_visit(input_visit, dictionary)
    y = _vectorize_outputs(output_visit, dictionary)
    x = np.repeat(x, y.shape[0], axis=0)
    for row in range(x.shape[0]):
        yield x[row, :], y[row, :]


def right_output_visit_generator(observations):
    """
    Generate visit pairs where the second element of the pair is always after
    (right) of the first element
    :param observations: the observations
    :return: a generator
    """
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits) - 1):
                for j in range(i + 1, len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_a), (j, visit_b)


# TODO: make this work...
def left_output_visit_generator(observations):
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits) - 1, 0, -1):
                for j in range(len(visits) - 1, 1, -1):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_b), (j, visit_a)


def visit_generator(observations):
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


def simple_input_output_generator(
        observations, dictionary, max_skip=2, kind="after"):
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
    else:
        raise ValueError("unknown kind")

    for (i, visit_a), (j, visit_b) in gen:
        if i != j and abs(j - i) <= max_skip:
            input_and_output = _vectorize_input_and_output(
                visit_a, visit_b, dictionary)
            for x, y in input_and_output:
                yield x, y


def random_input_output_generator(
        observations, dictionary, window_size=2, sample=1):
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
    for (i, visit_a), (j, visit_b) in visit_generator(observations):
        if i != j and abs(j - i) <= window_size:
            sample_b = random.sample(visit_b, min(len(visit_b), sample))
            input_and_output_vectorize = _vectorize_input_and_output(
                visit_a, sample_b, dictionary)
            for x, y in input_and_output_vectorize:
                yield x, y


def generate_batch(input_output_generator, batch_size=64):
    """
    Generate a batch of at most `batch_size`.

    Expects the input to be a generator which outputs input output pairs.

    :param input_output_generator:
    :param batch_size:
    :return: (training input, training output)
    """
    x, y = zip(*itertools.islice(input_output_generator, batch_size))
    return np.vstack(x), np.vstack(y)


if __name__ == "__main__":
    dictionary = {"A": 0, "B": 1, "C": 2, "D": 3}
    observations = [
        [["A", "B"], ["C", "A", "B"], ["D"]]
    ]

    x, y = generate_batch(
        simple_input_output_generator(
            observations, dictionary, max_skip=1, kind="after"),
        batch_size=10)
    print(x)
    print(y)
