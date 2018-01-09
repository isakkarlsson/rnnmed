import numpy as np
import itertools
import random


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


def _vectorize_input_and_output(visit_a, visit_b, dictionary):
    x = _vectorize_visit(visit_a, dictionary)
    y = _vectorize_outputs(visit_b, dictionary)
    x = np.repeat(x, y.shape[0], axis=0)
    for row in range(x.shape[0]):
#        print(visit_a, visit_b[row])
        yield x[row, :], y[row, :]


def non_overlapping_visit_generator(observations):
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits) - 1):
                for j in range(i + 1, len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_a), (j, visit_b)


def overlapping_visit_generator(observations):
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits)):
                for j in range(len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_a), (j, visit_b)


def simple_input_output_generator(observations, dictionary, max_skip_ahead=2):
    """
    Create a simple input output generator.

    :param observations: the observations
    :param dictionary: the dictionary (translating codes to indicies in the resulting one-hot vector)
    :param max_skip_ahead: maximum number of future visits to use
    :return: a generator which yields tuples of binary encoded training input and outputs
    """
    for (i, visit_a), (j, visit_b) in non_overlapping_visit_generator(observations):
        if abs(j - i) <= max_skip_ahead:
            for x, y in _vectorize_input_and_output(visit_a, visit_b, dictionary):
                yield x, y


def random_input_output_generator(observations, dictionary, max_skip=2, sample=1):
    """

    :param observations:
    :param dictionary:
    :param max_skip:
    :param sample:
    :return:
    """
    for (i, visit_a), (j, visit_b) in overlapping_visit_generator(observations):
        if i != j and abs(j - i) <= max_skip:
            sample_b = random.sample(visit_b, min(len(visit_b), sample))
            for x, y in _vectorize_input_and_output(visit_a, sample_b, dictionary):
                yield x, y


def generate_batch(input_ouput_generator, batch_size=64):
    """
    Generate a batch of size `batch_size`.

    Expects the input to be a generator which outputs input output pairs.

    :param input_ouput_generator:
    :param batch_size:
    :return:
    """
    x, y = zip(*itertools.islice(input_ouput_generator, batch_size))
    return np.vstack(x), np.vstack(y)


if __name__ == "__main__":
    dictionary = {"A": 0, "B": 1, "C": 2, "D": 3}
    observations = [
        [["A", "B"], ["C", "A", "B"], ["D"]]
    ]

    x, y = zip(*random_input_output_generator(observations, dictionary))
    print(np.vstack(x))
    print(np.vstack(y))

