import itertools

import tensorflow as tf

import data.io
import data.vectorize as vectorize
from data.vectorize import generate_input_batch


def initialize_decode_encode_layers(input_dim, layer_dim):
    in_dim = input_dim

    encode_weights = []
    decode_weights = []
    encode_biases = []
    decode_biases = []

    for i in range(0, len(layer_dim)):
        out_dim = layer_dim[i]
        encode_weights.append(
            tf.get_variable("encode-weight-{}".format(i),
                            shape=[in_dim, out_dim]))
        encode_biases.append(
            tf.get_variable("encode-bias-{}".format(i),
                            shape=[out_dim]))

        decode_weights.insert(
            0, tf.get_variable("decode-weight-{}".format(i),
                               shape=[out_dim, in_dim]))
        decode_biases.insert(
            0, tf.get_variable("decode-bias-{}".format(i),
                               shape=[in_dim]))
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

    def __init__(self, generator, dictionary, layer_dim=None):
        if layer_dim is None:
            layer_dim = [256, 128]  # last dim is output size

        if len(layer_dim) < 2:
            raise ValueError("needs at least 1 hidden layer")

        input_dim = len(dictionary)
        self.graph_ = tf.Graph()
        with self.graph_.as_default():
            encode_weight, encode_bias, decode_weight, decode_bias = \
                initialize_decode_encode_layers(input_dim, layer_dim)

            X = tf.placeholder("float", shape=[None, input_dim])
            encode_op = create_encode_decode_layer(X, encode_weight, encode_bias)
            decode_op = create_encode_decode_layer(encode_op, decode_weight,
                                                   decode_bias)

            y_pred = decode_op
            y_true = X
            loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)
            init = tf.global_variables_initializer()

        with tf.Session(graph=self.graph_) as sess:
            init.run()

            for i in range(1000):
                batch_x = generate_input_batch(generator, 32)
                _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
                print(l)



if __name__ == "__main__":
    observations, dictionary, reverse_dictionary = data.io.read_visits(
        "test_data/mimic_demo.seq")
    # observations, dictionary, reverse_dictionary = data.io.read_visits(
    #    "/mnt/veracrypt1/val/slv_events.seq", transform=transform)
    code_dim = len(dictionary)

    generator = itertools.cycle(vectorize.linear_input_generator(observations, dictionary))

    Med2Med(generator, dictionary, layer_dim=[512, 256, 128, 64])
