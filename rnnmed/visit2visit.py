__author__ = "Isak Karlsson"

import sklearn.metrics as metrics
import numpy as np
import tensorflow as tf

import rnnmed.data

import itertools


class IdentityTransform:
    def a(self):
        pass

    def transform(self, x):
        return x

    def __call__(self, *args):
        return self.transform(*args)


def auc(a, p):
    return metrics.roc_auc_score(a.reshape(-1), p.reshape(-1))


def compute_stats(y_train, y_in_pred, y_test, y_out_pred):
    pass


class HistoryPredictor:
    def __init__(self,
                 x,
                 y,
                 drop_prob,
                 n_hidden=32,
                 learning_rate=0.01,
                 regularization=0.00015):
        """
        
        """
        self._x = x
        self._y = y
        self._drop_prob = drop_prob
        self._n_features = self._x.shape[2]
        self._n_timesteps = self._x.shape[0]
        self._n_labels = self._y.shape[1]
        self._n_hidden = n_hidden
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._loss = None
        self._logits = None
        self._prediction = None
        self._optimize = None
        self.loss
        self.logits
        self.prediction
        self.optimize

    @property
    def logits(self):
        if self._logits is None:
            layer = tf.layers.dense(
                tf.reshape(self._x, [-1, self._n_features]), self._n_hidden)
            layer = tf.reshape(layer, [self._n_timesteps, -1, self._n_hidden])

            lstm_cell = tf.nn.rnn_cell.LSTMCell(
                self._n_hidden,
                use_peepholes=True,
                forget_bias=1.0,
                activation=tf.nn.sigmoid)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=self._drop_prob)
            outputs, states = tf.nn.dynamic_rnn(
                lstm_cell, layer, time_major=True, dtype=tf.float32)
            last_output = outputs[-1]
            self._logits = tf.layers.dense(
                last_output, self._n_labels, use_bias=True)

        return self._logits

    @property
    def prediction(self):
        if self._prediction is None:
            self._prediction = tf.nn.softmax(self.logits)

        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            l2_vars = [
                tf.nn.l2_loss(var) for var in tf.trainable_variables()
                if not "noreg" in var.name or "Bias" in var.name
            ]
            l2 = sum(l2_vars)
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self._y,
                    logits=self.logits)) + self._regularization * l2
        return self._loss

    @property
    def optimize(self):
        if self._optimize is None:
            self._optimize = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate).minimize(self.loss)
        return self._optimize


def visit2visit(generator,
                n_features,
                n_labels,
                n_timesteps=1,
                n_hidden=32,
                max_iter=1000,
                transform=None):
    if transform is None:
        transform = IdentityTransform()

    x_test, y_test = rnnmed.data.generate_time_batch(generator, batch_size=64)
    print(x_test.shape)

    # use the rest for traning now ...
    generator = itertools.cycle(generator)

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, shape=[n_timesteps, None, n_features])
        y = tf.placeholder(tf.int32, shape=[None])
        keep_proba = tf.placeholder_with_default(1.0, shape=())

        layer = tf.layers.dense(tf.reshape(X, [-1, n_features]), 128)
        layer = tf.reshape(layer, [n_timesteps, -1, 128])

        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            n_hidden,
            use_peepholes=True,
            forget_bias=1.0,
            activation=tf.nn.sigmoid)
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(
            n_hidden,
            use_peepholes=True,
            forget_bias=1.0,
            activation=tf.nn.sigmoid)

        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=keep_proba)
        #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 3)
        outputs, states = tf.nn.dynamic_rnn(
            lstm_cell, layer, time_major=True, dtype=tf.float32)

        last_output = outputs[-1]
        pred_y = tf.layers.dense(last_output, n_labels, use_bias=True)

        prob_y = tf.nn.softmax(pred_y)

        #print(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
        l2_vars = [
            tf.nn.l2_loss(var) for var in tf.trainable_variables()
            if not "noreg" in var.name or "Bias" in var.name
        ]
        print(l2_vars)
        l2 = sum(l2_vars)

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=pred_y)) + 0 * l2
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(cost)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        for e in range(max_iter):
            x_batch, y_batch = rnnmed.data.generate_time_batch(
                generator, batch_size=128)
            _, loss = sess.run(
                [optimizer, cost],
                feed_dict={
                    X: transform(x_batch),
                    y: y_batch.reshape(-1),
                    keep_proba: 0.5
                })
            if e % 250 == 0:
                print("epoch: {}. loss: {}".format(e, loss))

        prob, loss = sess.run(
            [prob_y, cost],
            feed_dict={
                X: transform(x_test),
                y: y_test.reshape(-1)
            })

        print(np.round(prob, 4))
        pred = np.argmax(prob, axis=1)
        print(np.sum(pred.reshape(-1) == y_test.reshape(-1)) / y_test.shape[0])

        a, b = y_test.reshape(-1), prob[:, 1].reshape(-1)
        print(a)
        print(b)
        print("AUC:", metrics.roc_auc_score(a, b))

        print(pred.reshape(-1))
        print(y_test.reshape(-1))
        print(loss)


if __name__ == "__main__":
    visit2visit()
