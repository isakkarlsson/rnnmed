import itertools

import numpy as np


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


def vectorize(example, n_features):
    """One-hot encode a set of values, which are the index of the columns
    with non-zero values.

    :param visit: the input
    :param n_features: the number of features
    :returns: array of shape [1, n_features]
    :rtype: np.array

    """
    x = np.zeros(shape=(1, n_features))
    for where, value in example:
        x[0, where] = value
    return x


def vectorize2d(example, n_features):
    """One-hot encodes each index as a separate vector, i.e., with a
    single non-zero value.

    :param
    visit:  the example
    :param n_features: the number of features
    :returns: array of shape ``[len(example), n_features]``
    :rtype: np.array

    """
    x = np.zeros(shape=(len(example), n_features))
    for row, (where, value) in enumerate(example):
        x[row, where] = value
    return x


def input_output_batch(input_output_generator,
                       batch_size=64,
                       x_concat=np.vstack,
                       y_concat=np.vstack):
    """Generate a batch of at most `batch_size`.

    Expects the input to be a generator which outputs input output pairs.

    :param input_output_generator:
    :param batch_size:
    :return: (training input, training output)

    """
    a = list(itertools.islice(input_output_generator, batch_size))
    if a:
        x, y = zip(*a)
        return x_concat(x), y_concat(y)
    return None


def input_batch(input_generator, batch_size=64, concat=np.vstack):
    """Generate a batch of at most `batch_size` of a genrator generating a
    single array

    :param input_generator: a generator of numpy arrays
    :param batch_size: the batch size
    :param concat: concat the examples to ``batch_size`` (default: np.vstack)
    :yields: arrays of size ``[batch_size, None]``
    :rtype: generator

    """
    a = list(itertools.islice(input_generator, batch_size))
    if a:
        return concat(a)
    return None


def time_batch(generator, batch_size=64):
    """Generate a batch of `batch_size` input output pairs

    This method is a convenient way of calling:
      `generate_input_and_output_batch(
          generator,
          batch_size,
          x_concat=lambda x: np.concatenate(x, axis=1))`

    :param generator: the generator
    :param batch_size: the batch_size
    :returns: a [n_step, batch_size, n_features] numpy array
    """
    return input_output_batch(
        generator,
        batch_size=batch_size,
        x_concat=lambda x: np.concatenate(x, axis=1))


def collect_batch(generator, batcher=input_output_batch, batch_size=64):
    """ Collect a list of complete batches of at most `batch_size`

    :param generator: the generator to collect
    :param batcher: the function creating batches (return None when no more batches can be created)
    :param batch_size: the size of the batches
    :yields: the next batch

    """
    while True:
        data = batcher(generator, batch_size)
        if not data:
            break
        yield data


def make_ndarrays(generator, *args):
    if len(args) > 1:
        # the generator generates tuples, e.g. (x, y)
        arrays = zip(*list(generator))
        return [f(a) for f, a in zip(args, arrays)]
    else:
        return args[0](list(generator))


def make_input_output_arrays(generator, x_concat=np.vstack,
                             y_concat=np.vstack):
    return make_ndarrays(generator, x_concat, y_concat)


def make_time_input_output_arrays(generator, y_concat=np.vstack):
    return make_input_output_arrays(
        generator,
        x_concat=lambda x: np.concatenate(x, axis=1),
        y_concat=y_concat)


def chain_dict(*translators):
    """Create a translator chain.

    Example
    -------

    ::
    a = {0: "A", 1:"B"}
    b = {"A": "An A", "B": "A B"}

    translator = translator_chain(a, b)



    :param translators:
    :return:
    """
    prev_trans = translators[0]
    for translator in translators[1:]:
        if len(prev_trans) > len(translator):
            raise ValueError("illegal sizes")
        prev_trans = translator

    class F:
        def __len__(self):
            return len(translators[0])

        def __getitem__(self, code):
            r = code
            for translator in translators:
                r = translator.get(code)
                if r is None:
                    return code
                code = r
            return code

    return F()


if __name__ == "__main__":
    a = {0: "A", 1: "B"}
    b = {"A": "An A", "B": "A B"}

    translator = chain_dict(a, b)
    print(len(translator))
    print(translator[0])
