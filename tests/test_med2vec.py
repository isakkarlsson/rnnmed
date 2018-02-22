import unittest

import numpy as np

import rnnmed.from.io
import rnnmed.from.vectorize as vectorize
import rnnmed.med2vec as med2vec
import rnnmed.visit2visit as visit2visit

import rnnmed.from.timeseries as ts


class TestMed2Vec(unittest.TestCase):
    def test_observations(self):
        m = 64  # code weight matrix dimension
        n = 32  # visit weight matrix dimension
        code_len = 2

        def transform(code):
            return code[:code_len]

        # uncomment to disable
        transform = None

        max_iter = 1000
        max_iter_progress_report = 250

        batch_size = 128

        observations = rnnmed.from.io.read_labeled_observations(
            "test_data/label_synthetic.seq")
        generator = vectorize.random_input_output_generator(
            observations, window_size=2, sample=3)

        m = med2vec.med2vec(
            generator,
            code_dim=observations.n_features,
            batch_size=batch_size,
            max_iter_progress_report=max_iter_progress_report,
            max_iter=max_iter,
            code_rep_dim=m,
            visit_rep_dim=n,
            l2_reg=0.01)

        x, y = vectorize.generate_time_batch(
            vectorize.time_observation_generator(observations, 4),
            batch_size=4)

        print(x.shape)
        
        print(np.round(m.transform(x), 3))

    def test_med_2_vec_predict(self):
        observations = rnnmed.from.io.read_labeled_observations(
            "test_data/mimic_demo.seq")
        # train_observations = observations[4:]
        # generator = rnnmed.vectorize.random_input_output_generator(
        #    train_observations, window_size=5, sample=3)
        
        # m = med2vec.med2vec(
        #     generator,
        #     code_dim=observations.n_features,
        #     batch_size=32,
        #     max_iter=1000,
        #     max_iter_progress_report=250,
        #     code_rep_dim=128,
        #     visit_rep_dim=256,
        #     l2_reg=0.01)
        np.set_printoptions(suppress=True)
        visit2visit.visit2visit(observations, n_visits=5, transform=None)

    def test_time_series(self):
        import random
        
        timeseries = rnnmed.from.io.read_time_series(
            open("test_data/synthetic_control.txt"))

        random.shuffle(timeseries)
        print(timeseries[0])
        generator = ts.timeseries_generator(timeseries)

        #  x, y = rnnmed.data.generate_time_batch(generator, batch_size=5)
        np.set_printoptions(suppress=True)
        visit2visit.visit2visit(
            generator,
            n_features=timeseries.n_dimensions,
            n_timesteps=timeseries.n_timesteps,
            n_labels=timeseries.n_labels,
            n_hidden=128,
            max_iter=400)
            
            
        
