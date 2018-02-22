import itertools

import tensorflow as tf
import numpy as np

import data.io
import data.vectorize as vectorize
from data.vectorize import generate_input_batch


def initialize_decode_encode_layers(input_dim, layer_dim):
    """Inititialize the encoding and decoding layers.

    :param input_dim: size of the input dimension
    :param layer_dim: size of the hidden layers dimensions

    :returns: a tuple of encode weights, encode biases, decode weights
    decode biases

    """
    if not layer_dim:
        raise ValueError("need to specify hidden layer size")

    in_dim = input_dim

    encode_weights = []
    decode_weights = []
    encode_biases = []
    decode_biases = []

    for i in range(0, len(layer_dim)):
        out_dim = layer_dim[i]
        encode_weights.append(
            tf.get_variable(
                "encode-weight-{}".format(i), shape=[in_dim, out_dim]))
        encode_biases.append(
            tf.get_variable("encode-bias-{}".format(i), shape=[out_dim]))

        decode_weights.insert(0,
                              tf.get_variable(
                                  "decode-weight-{}".format(i),
                                  shape=[out_dim, in_dim]))
        decode_biases.insert(0,
                             tf.get_variable(
                                 "decode-bias-{}".format(i), shape=[in_dim]))
        in_dim = layer_dim[i]

    return encode_weights, encode_biases, decode_weights, decode_biases


def create_encode_decode_layer(x, weights, biases):
    weight = weights[0]
    bias = biases[0]
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))
    for i in range(1, len(weights)):
        weight = weights[i]
        bias = biases[i]
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, weight), bias))
    return layer


class Med2Med:
    def __init__(self, input_data, layer_dim, learning_rate=0.01):
        self.layer_dim = layer_dim
        self.learning_rate = learning_rate

        x_noise = input_data + (
            0.01 * tf.random_normal(shape=input_data.shape, mean=0, stddev=1))
        input_dim = input_data.shape[1]  # col
        layers = initialize_decode_encode_layers(input_dim, layer_dim)
        encode_weight, encode_bias, decode_weight, decode_bias = layers

        self.encode_ = create_encode_decode_layer(x_noise, encode_weight,
                                                  encode_bias)
        self.decode_ = create_encode_decode_layer(self.encode_, decode_weight,
                                                  decode_bias)
        y_pred = self.decode
        y_true = input_data
        self.loss_ = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_pred, labels=y_true))
        # self.loss_ = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
        self.optimize_ = tf.train.RMSPropOptimizer(
            self.learning_rate).minimize(self.loss)

    @property
    def decode(self):
        return self.decode_

    @property
    def encode(self):
        return self.encode_

    @property
    def loss(self):
        return self.loss_

    @property
    def optimize(self):
        return self.optimize_


def test(epochs=1000):
    observations, dictionary, reverse_dictionary = data.io.read_observations(
        "test_data/mimic_demo.seq")
    generator = itertools.cycle(
        vectorize.linear_input_generator(observations, dictionary))

    input_dim = len(dictionary)
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, input_dim])
        m2m = Med2Med(X, layer_dim=[256, 128])
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for i in range(epochs):
            batch_x = generate_input_batch(generator, 32)
            _, loss = sess.run(
                [m2m.optimize, m2m.loss], feed_dict={X: batch_x})
            if i % 25 == 0:
                print("epoch {} loss {}".format(i, loss))

        batch_x = generate_input_batch(generator, 2)
        ex = sess.run(m2m.decode, feed_dict={X: batch_x})
        print(np.round(ex, 0))
        print(batch_x)


if __name__ == "__main__":
    observations, dictionary, reverse_dictionary = data.io.read_observations(
        "test_data/mimic_demo.seq")
    # observations, dictionary, reverse_dictionary = data.io.read_observations(
    #    "/mnt/veracrypt1/val/slv_events.seq", transform=transform)
    generator = itertools.cycle(
        vectorize.linear_input_generator(observations, dictionary))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        input_dim = len(dictionary)
        X = tf.placeholder(tf.float32, [None, input_dim])
        m2m = Med2Med(X, layer_dim=[512, 256, 128, 64])

    m2m = Med2Med(generator, dictionary, layer_dim=[512, 256, 128, 64])
