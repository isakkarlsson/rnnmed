__author__ = "Isak Karlsson"

import numpy as np
import tensorflow as tf

import rnnmed.data

import itertools


class IdentityTransform:
    def transform(self, x):
        return x

    def __call__(self, *args):
        return self.transform(*args)


def visit2visit(generator,
                n_features,
                n_labels,
                n_timesteps=1,
                n_hidden=32,
                max_iter=1000,
                transform=None):
    if transform is None:
        transform = IdentityTransform()

    x_test, y_test = rnnmed.data.generate_time_batch(generator, batch_size=32)

    # use the rest for traning now ...
    generator = itertools.cycle(generator)

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, shape=[n_timesteps, None, n_features])
        y = tf.placeholder(tf.int32, shape=[None])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell, lstm_cell_1])
        outputs, states = tf.nn.static_rnn(
            cell, tf.unstack(X, axis=0), dtype=tf.float32)

        last_output = outputs[-1]
        pred_y = tf.layers.dense(last_output, n_labels, use_bias=True)

        prob_y = tf.nn.softmax(pred_y)
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=pred_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        for e in range(max_iter):
            x_batch, y_batch = rnnmed.data.generate_time_batch(
                generator, batch_size=32)
            _, loss = sess.run(
                [optimizer, cost],
                feed_dict={
                    X: transform(x_batch),
                    y: y_batch.reshape(-1)
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
        print(np.sum(pred == y_test.reshape(-1)/y_test.shape[0]))

        print(pred)
        print(y_test)
        print(loss)


if __name__ == "__main__":
    visit2visit()
