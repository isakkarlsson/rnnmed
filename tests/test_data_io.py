import unittest

import rnnmed.data
import rnnmed.data.io
import rnnmed.data.timeseries as ts
import rnnmed.data.observations as observations


class TestData(unittest.TestCase):
    def test_observations(self):
        observations = rnnmed.data.observations.Observations()
        data = [[["A"], ["B"]], [["C"], ["D"]]]

        for item in data:
            observations.add(item)

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations.n_features, 4)
        self.assertEqual(observations.n_labels, 0)
        self.assertEqual(observations.original_data, data)
        self.assertEqual(observations.data,
                         [[[(0, 1)], [(1, 1)]], [[(2, 1)], [(3, 1)]]])

    def test_time_series(self):
        import numpy as np
        ob = rnnmed.data.observations.Observations()
        data = [[["A"], ["B"]], [["C"], ["D"]]]

        for item in data:
            ob.add(item, 1)

        o_gen = observations.time_observation_generator(ob, n_visits=2)
        a_gen = observations.time_observation_generator(ob, n_visits=2)
        b_gen = observations.time_observation_generator(ob, n_visits=2)
        timeseries = rnnmed.data.io.read_time_series(
            open("test_data/synthetic_control.txt"))

        c_gen = rnnmed.data.concatenate_generator(
            [a_gen, b_gen], concat=lambda x: np.concatenate(x, axis=2))

        for x, y in c_gen:
            print(x)
            print(y)

        print(timeseries[0])
        generator = ts.timeseries_generator(timeseries)
        x_a, y_a = rnnmed.data.generate_time_batch(generator, batch_size=2)
        x_b, _ = rnnmed.data.generate_time_batch(o_gen, batch_size=2)
        print(x_a)

    def test_time_series_observation(self):
        def week_agg(date):
            return date.year, date.isocalendar()[1]

        def month_agg(date):
            return date.year, date.month

        ob = rnnmed.data.io.read_time_series_observation(
            open("/mnt/veracrypt1/EHR_DATA/L270-90-raw-measurements.csv"),
            min_sparsity=0.1)

        import random
        random.seed(10)
        random.shuffle(ob)
        n_visits = 10
        generator = observations.time_observation_generator(
            ob, n_visits=n_visits)

        print(len(ob), ob.n_features)
        from rnnmed.visit2visit import visit2visit

        visit2visit(
            generator,
            n_features=ob.n_features,
            n_labels=ob.n_labels,
            n_timesteps=n_visits,
            n_hidden=128,
            max_iter=1000)
