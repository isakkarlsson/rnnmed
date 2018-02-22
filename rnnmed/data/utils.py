import itertools

import numpy as np


def generate_input_output_batch(input_output_generator,
                                batch_size=64,
                                x_concat=np.vstack,
                                y_concat=np.vstack):
    """Generate a batch of at most `batch_size`.

    Expects the input to be a generator which outputs input output pairs.

    :param input_output_generator:
    :param batch_size:
    :return: (training input, training output)

    """
    x, y = zip(*itertools.islice(input_output_generator, batch_size))
    return x_concat(x), y_concat(y)


def generate_input_batch(input_generator, batch_size=64):
    """Generate a batch of at most `batch_size` of a genrator generating a
    single array

    :param input_generator: 
    :param batch_size: 
    :returns: 
    :rtype:

    """
    return np.vstack(itertools.islice(input_generator, batch_size))


def generate_time_batch(generator, batch_size=64):
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
    return generate_input_output_batch(
        generator,
        batch_size=batch_size,
        x_concat=lambda x: np.concatenate(x, axis=1))


def concatenate_generator(generators, concat=np.vstack):
    for gen in zip(*generators):
        x = [x for x, y in gen]
        yield concat(x), gen[0][1]


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
