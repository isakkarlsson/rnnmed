import itertools

import numpy as np
import tensorflow as tf

from data.vectorize import generate_input_output_batch


def random_initializer(minval, maxval):
    def _init(shape, dtype=None, partition_info=None):
        return tf.random_uniform(shape=shape, minval=minval, maxval=maxval)
    return _init


class Med2Vec:

    def __init__(self, code_weight, code_bias, visit_weight, visit_bias):
        self.code_weight = code_weight
        self.code_bias = code_bias
        self.visit_bias = visit_bias
        self.visit_weight = visit_weight

    def transform(self, x):
        intermediate_visit = np.maximum(
            np.dot(x, self.code_weight) + self.code_bias, 0)
        return np.maximum(
            np.dot(intermediate_visit, self.visit_weight) + self.visit_bias, 0)


def med2vec(input_output_generator, code_dim, code_rep_dim=512,
            visit_rep_dim=1024, batch_size=64, max_iter=1000,
            max_iter_progress_report=100, start_learning_rate=0.5,
            l2_reg=0.001):
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)

    med2vec_graph = tf.Graph()
    with med2vec_graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, code_dim))
        y = tf.placeholder(tf.float32, shape=(None, code_dim))

        code_bias = tf.Variable(tf.zeros([code_rep_dim]),
                                name="code_bias")
        code_weight = tf.get_variable("code_weight",
                                      #initializer=random_initializer(0.1, 1),
                                      shape=(code_dim, code_rep_dim),
                                      regularizer=l2_regularizer)

        intermediate_visit = tf.nn.relu_layer(x, code_weight, code_bias)

        visit_bias = tf.Variable(tf.zeros([visit_rep_dim]))
        visit_weight = tf.get_variable("visit_weight",
                                       #initializer=random_initializer(0.1, 1),
                                       shape=(code_rep_dim, visit_rep_dim),
                                       regularizer=l2_regularizer)

        visit_rep = tf.nn.relu_layer(intermediate_visit, visit_weight,
                                     visit_bias)

        # dense layer representing the softmax classifier
        # we use tf.identity instead of `tf.nn.softmax`
        # so we can use the numerically stable
        # `tf.nn.softmax_cross_entropy_with_logits`
        # TODO: tf.identity if is_training else tf.softmax
        y_hat = tf.layers.dense(visit_rep, code_dim,
                                kernel_regularizer=tf.nn.l2_loss,
                                activation=tf.identity)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))

        reg_variables = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(
            l2_regularizer, reg_variables)

        loss += reg_term

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            start_learning_rate, global_step, 100000, 0.96, staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()

    with tf.Session(graph=med2vec_graph) as session:
        init.run()
        loss_values = []
        tmp_loss = []

        vec_cycle = itertools.cycle(input_output_generator)
        for i in range(max_iter):
            x_data, y_data = generate_input_output_batch(
                vec_cycle, batch_size=batch_size)
            feed_dict = {x: x_data, y: y_data}
            _, loss_value, y_hat_val, y_val = \
                session.run([optimizer, loss, y_hat, y], feed_dict=feed_dict)

            tmp_loss.append(loss_value)
            if i % max_iter_progress_report == 0:
                print("epoch:", i, "avg_loss:", np.mean(tmp_loss))
                loss_values.append(np.mean(tmp_loss))
                tmp_loss = []

        return Med2Vec(code_weight.eval(), code_bias.eval(),
                       visit_weight.eval(), visit_bias.eval())


if __name__ == "__main__":
    import data.io
    import data.vectorize as vectorize

    m = 128  # code weight matrix dimension
    n = 256  # visit weight matrix dimension
    code_len = 2


    def transform(code):
        return code[:code_len]


    # uncomment to disable
    transform = None

    max_iter = 10000
    max_iter_progress_report = 100

    batch_size = 128

    observations, dictionary, reverse_dictionary = data.io.read_visits(
        "test_data/mimic_demo.seq")
    observations, dictionary, reverse_dictionary = data.io.read_visits(
        "/mnt/veracrypt1/val/slv_events.seq", transform=transform)
    code_dim = len(dictionary)
    print("Data contains {} observations with {} different codes".format(
        len(observations), code_dim))

    input_output_generator = vectorize.random_input_output_generator(
        observations, dictionary, window_size=1, sample=3)
    m2v = med2vec(input_output_generator,
                  code_dim=code_dim,
                  max_iter=max_iter,
                  code_rep_dim=m,
                  visit_rep_dim=n,
                  l2_reg=0.0 )
