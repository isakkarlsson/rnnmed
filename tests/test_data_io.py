import unittest

import rnnmed.data
import rnnmed.data.io
import rnnmed.data.timeseries as ts
import rnnmed.data.vectorize as vectorize


class TestData(unittest.TestCase):
    def test_observations(self):
        observations = rnnmed.data.io.Observations()
        data = [[["A"], ["B"]], [["C"], ["D"]]]

        for item in data:
            observations.add(item)

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations.n_features, 4)
        self.assertEqual(observations.n_labels, 0)
        self.assertEqual(observations.original_data, data)
        self.assertEqual(observations.data, [[[0], [1]], [[2], [3]]])
        import numpy as np
        a = np.argmax([[0, 1, 0], [1, 0, 0]], axis=1)
        print(a)
        print(observations.data_index.reverse_transform(a))

    def test_time_series(self):
        import numpy as np
        observations = rnnmed.data.io.Observations()
        data = [[["A"], ["B"]], [["C"], ["D"]]]

        for item in data:
            observations.add(item, 1)

        o_gen = vectorize.time_observation_generator(observations, n_visits=2)
        a_gen = vectorize.time_observation_generator(observations, n_visits=2)
        b_gen = vectorize.time_observation_generator(observations, n_visits=2)
        timeseries = rnnmed.data.io.read_time_series(
            open("test_data/synthetic_control.txt"))

        for x, y in o_gen:
            print(x)
        c_gen = rnnmed.data.concatenate_generator(
            [a_gen, b_gen], concat=lambda x: np.concatenate(x, axis=2))
        
        for x, y in c_gen:
            print(x)
            print(y)
        raise "ds"
        print(timeseries[0])
        generator = ts.timeseries_generator(timeseries)
        x_a, y_a = rnnmed.data.generate_time_batch(generator, batch_size=2)
        x_b, _ = rnnmed.data.generate_time_batch(o_gen, batch_size=2)
        print(np.concatenate([x_a, x_b], axis=2))
