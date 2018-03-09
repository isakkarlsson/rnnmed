import unittest
import itertools

import rnnmed.data
import rnnmed.data.io
import rnnmed.data.timeseries as ts
import rnnmed.data.observations as observations
import tensorflow as tf

class TestHistoryPredict(unittest.TestCase):
    def test_time_series_observation(self):
        ob = rnnmed.data.io.read_time_series(
            open("test_data/synthetic_control.txt"))

        import random
        random.seed(10)
        random.shuffle(ob)

        generator = itertools.cycle(ts.timeseries_generator(ob))

        from rnnmed.visit2visit import HistoryPredictor

        X = tf.placeholder(
            tf.float32, shape=[ob.n_timesteps, None, ob.n_dimensions])
        y = tf.placeholder(tf.int32, shape=[None])
        drop_prob = tf.placeholder_with_default(1.0, shape=())
        hp = HistoryPredictor(X, tf.one_hot(y, depth=ob.n_labels), drop_prob)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(100):
                b_x, b_y = rnnmed.data.generate_time_batch(generator)
                _, _loss = sess.run(
                    [hp.optimize, hp.loss],
                    feed_dict={
                        X: b_x,
                        y: b_y.reshape(-1),
                        drop_prob: 0.4
                    })
                if i % 25 == 0:
                    print(_loss)
