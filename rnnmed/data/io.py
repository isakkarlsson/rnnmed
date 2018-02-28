from rnnmed.data.observations import Observations
from rnnmed.data.timeseries import Timeseries


class AlwaysIn:
    """Dummy type to simulate a collection that contains all items"""

    def __contains__(self, item):
        return True


# singleton instance of AlwaysIn
__always_in__ = AlwaysIn()


def __parse_observation(pat, transform, valid_code, code_sep):
    observation = []
    for visit in (vis.strip().split(code_sep) for vis in pat):
        out = []  # set()
        for code in visit:
            code = transform(code) if transform is not None else code
            if valid_code is not None and code in valid_code:
                out.append(code)

        if out:  # perhaps we should remove this check...
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
