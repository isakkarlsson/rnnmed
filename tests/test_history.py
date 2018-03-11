import unittest
import itertools

import rnnmed.data
import rnnmed.data.io
import rnnmed.data.timeseries as ts
import rnnmed.data.observations as observations
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from rnnmed.visit2visit import HistoryPredictor
import numpy as np

def batch_generator(size, batch_size=32):
    for i in range(0, size, batch_size):
        yield slice(i, min(i+batch_size, size))


def train_test(x_train, y_train, x_test, y_test, n_timesteps, n_features,
               n_labels):
    return "Run of thingi"




class TestHistoryPredict(unittest.TestCase):

    def test_time_series_observation(self):
        ob = rnnmed.data.io.read_time_series_observation(
            open("/home/isak/D611-90-raw-measurements.csv"), min_sparsity=0.4)

        import random
        random.seed(10)
        random.shuffle(ob)

        n_labels = ob.n_labels
        n_features = ob.n_features
        n_timesteps = 10
        generator = observations.time_observation_generator(
            ob, n_visits=n_timesteps)

        x_data, y_data = zip(*list(generator))

        x_data = np.concatenate(x_data, axis=1)
        y_data = np.vstack(y_data)
        print(x_data.shape)
        print(y_data.shape)
        aucs = []
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for fold, (train, test) in enumerate(skf.split(np.zeros(x_data.shape[1]), y_data)):
            x_train = x_data[:, train, :]
            y_train = y_data[train, :]
            x_test = x_data[:, test, :]
            y_test = y_data[test, :]
            graph = tf.Graph()
            with graph.as_default():
                X = tf.placeholder(tf.float32, shape=[n_timesteps, None, n_features])
                y = tf.placeholder(tf.int32, shape=[None])
                drop_prob = tf.placeholder_with_default(1.0, shape=())
                hp = HistoryPredictor(X, tf.one_hot(y, depth=n_labels), drop_prob)
                init = tf.global_variables_initializer()

            with tf.Session(graph=graph) as sess:
                sess.run(init)
                for epoch in range(1000):
                    for idx in batch_generator(x_train.shape[1], 32):
                        _, _loss = sess.run(
                            [hp.optimize, hp.loss], feed_dict={
                                X: x_train[:, idx, :],
                                y: y_train[idx, :].reshape(-1),
                                drop_prob: 0.4
                            })

                    if epoch % 25 == 0:
                        print("Fold {}, epoch {} loss: {}".format(fold, epoch, _loss))

                y_pred = sess.run(hp.prediction, feed_dict={X: x_test})
                auc = roc_auc_score(y_test.reshape(-1), y_pred[:, 1])
                print(auc)
                aucs.append(auc)
            print("mean auc:", np.mean(aucs))
