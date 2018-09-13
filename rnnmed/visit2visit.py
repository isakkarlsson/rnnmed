"""This module implements a simple history predictor
"""
import tensorflow as tf

class HistoryPredictor:
    def __init__(self,
                 x,
                 y,
                 drop_prob,
                 n_hidden=32,
                 learning_rate=0.01,
                 regularization=0.00015):
        """A model that can predict based on historical records.
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
            outputs, _states = tf.nn.dynamic_rnn(
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
