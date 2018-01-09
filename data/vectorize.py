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


def right_consecutive_visit_generator(observations):
    """
    Generate visit pairs where the second element of the pair is always after (right) of the first element
    :param observations: the observations
    :return: a generator
    """
    for visits in observations:
        if len(visits) >= 2:
            for i in range(len(visits) - 1):
                for j in range(i + 1, len(visits)):
                    visit_a, visit_b = visits[i], visits[j]
                    yield (i, visit_a), (j, visit_b)


def visit_generator(observations):
    """
    Generate visit pairs for all possible combinations of visits per observation
    :param observations: the observations
    :return: a generator
    """
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
    for (i, visit_a), (j, visit_b) in right_consecutive_visit_generator(observations):
        if abs(j - i) <= max_skip_ahead:
            for x, y in _vectorize_input_and_output(visit_a, visit_b, dictionary):
                yield x, y


def random_input_output_generator(observations, dictionary, window_size=2, sample=1):
    """
    Create a input output generator where each input is associated with a random output that occurs within at least
    `window_size` visits.

    :param observations: the observations
    :param dictionary: the dictionary
    :param window_size: the window size
    :param sample: the number of outputs to sample from each visit within the window
    :return: a generator
    """
    for (i, visit_a), (j, visit_b) in visit_generator(observations):
        if i != j and abs(j - i) <= window_size:
            sample_b = random.sample(visit_b, min(len(visit_b), sample))
            for x, y in _vectorize_input_and_output(visit_a, sample_b, dictionary):
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

    x, y = generate_batch(simple_input_output_generator(observations, dictionary, max_skip_ahead=1), batch_size=10)
    print(x)
    print(y)
