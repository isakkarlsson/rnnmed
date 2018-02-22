import numpy as np


def timeseries_generator(timeseries):
    for x, y in timeseries:
        x = np.array(x).reshape(timeseries.n_timesteps, 1,
                                timeseries.n_dimensions)
        yield x, y
