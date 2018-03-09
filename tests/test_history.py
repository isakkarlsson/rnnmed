import unittest
import itertools

import rnnmed.data
import rnnmed.data.io
import rnnmed.data.timeseries as ts
import rnnmed.data.observations as observations
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

import numpy as np

class TestHistoryPredict(unittest.TestCase):
    def test_time_series_observation(self):
        ob = rnnmed.data.io.read_time_series_observation(
            open("/mnt/veracrypt1/EHR_DATA/D611-90-raw-measurements.csv"))

        import random
        random.seed(10)
        random.shuffle(ob)

        n_labels = ob.n_labels
        n_features = ob.n_features
        n_timesteps = 6
        generator = observations.time_observation_generator(ob, n_visits=n_timesteps)

        x_data, y_data = zip(*list(generator))

        x_data = np.concatenate(x_data, axis=1)
        y_data = np.vstack(y_data)
        print(x_data.shape)
        print(y_data.shape)

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        for train, test in skf.split(np.zeros(x_data.shape[1]), y_data):
            x_train = x_data[:, train, :]
            y_train = y_data[train, :]

            x_test = x_data[:, test, :]
            y_test = y_data[test, :]

        batches = list(
            rnnmed.data.collect_batch(
                generator, batcher=rnnmed.data.time_batch, batch_size=16))

        from rnnmed.visit2visit import HistoryPredictor

        X = tf.placeholder(
            tf.float32, shape=[n_timesteps, None, n_features])
        y = tf.placeholder(tf.int32, shape=[None])
        drop_prob = tf.placeholder_with_default(1.0, shape=())
        hp = HistoryPredictor(X, tf.one_hot(y, depth=n_labels), drop_prob)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(30):
                for b_x, b_y in batches:
                    _, _loss = sess.run(
                        [hp.optimize, hp.loss],
                        feed_dict={
                            X: b_x,
                            y: b_y.reshape(-1),
                            drop_prob: 0.4
                        })

                if epoch % 25 == 0:
                    print(_loss)
