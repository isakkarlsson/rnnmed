from datetime import datetime
from collections import defaultdict, OrderedDict

import numpy as np

from rnnmed.data.observations import Observations
from rnnmed.data.timeseries import Timeseries


class AlwaysIn:
    """Dummy type to simulate a collection that contains all items"""

    def __contains__(self, item):
        return True


# singleton instance of AlwaysIn
__always_in__ = AlwaysIn()


def _parse_observation(pat, transform, valid_code, code_sep):
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
            observation = _parse_observation(pat, transform, valid_code,
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
            observation = _parse_observation(data.strip().split(visit_sep),
                                             None, __always_in__, code_sep)
            if observation:
                observations.add(observation, label)
        return observations


def _day_aggregate(date):
    return (date.year, date.month, date.day)


class _Example:
    def __init__(self):
        self.data = {}

    def add(self, date, code, value):
        coll = self.data.get(date)
        if not coll:
            coll = defaultdict(list)
            self.data[date] = coll
        coll[code].append(value)


def read_time_series_observation(f,
                                 sep=",",
                                 agg=_day_aggregate,
                                 min_sparsity=0):
    def std_norm(v, mean, std):
        if std == 0:
            return 1
        else:
            v = np.array(v)
            return (v - mean) / std

    with f as t_f:
        next(t_f)  # skip header

        observations = Observations()
        examples = defaultdict(_Example)
        values = defaultdict(list)
        code_pid = defaultdict(set)
        pid_count = set()

        for line in t_f:
            pid, date_str, code, value, label = line.strip().split(sep)
            date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            date_key = agg(date)
            examples[pid].add(date_key, code, float(value))
            examples[pid].label = label
            values[code].append(float(value))
            code_pid[code].add(pid)
            pid_count.add(pid)

        stats = {}
        for code, value in values.items():
            stats[code] = (np.mean(value), np.std(value))
        
        n_examples = float(len(pid_count))
        sparsity = {}
        for code, pids in code_pid.items():
            sparsity[code] = len(pids) / n_examples

        for pid, example in examples.items():
            observation = []
            for date, values in example.data.items():
                visit = []
                for code, value in values.items():
                    if sparsity[code] > min_sparsity:
                        v_sum = np.mean(std_norm(value, *stats[code]))
                        visit.append((code, v_sum))
                if visit:
                    observation.append(visit)
            if observation:
                observations.add(observation, example.label)
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
