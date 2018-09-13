import itertools
import numpy as np
import tensorflow as tf

from rnnmed.data import generate_input_output_batch


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

    @property
    def n_features(self):
        return self.visit_weight.shape[1]

    def transform(self, x):
        intermediate_visit = np.maximum(
            np.dot(x, self.code_weight) + self.code_bias, 0)
        visit = np.dot(intermediate_visit, self.visit_weight) + self.visit_bias
        return np.maximum(visit, 0)

    def __call__(self, *args):
        return self.transform(*args)


class Med2Vec:
    def __init__(self,
                 x,
                 y,
                 n_hidden_code=512,
                 n_hidden_visit=1024,
                 learning_rate=0.5,
                 regularization=0.0001):
        self._x = x
        self._y = y
        self._n_hidden_code = n_hidden_code
        self._n_hidden_visit = n_hidden_visit
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._transform = None
        self._loss = None
        self._optimize = None
        self.transform
        self.loss
        self.optimize

    @property
    def transform(self):
        if self._tranform is None:
            c_bias = tf.Variable(
                tf.zeros([self._n_hidden_code]), name="code_bias")
            c_weight = tf.get_variable(
                "code_weight",
                initializer=random_initializer(0.01, 0.1),
                shape=[self._x.shape[1], self._n_hidden_code],
                regularizer=self._l2_regularizer)

            v_bias = tf.Variable(tf.zeros([self._n_hidden_visit]))
            v_weight = tf.get_variable(
                "visit_weight",
                initializer=random_initializer(0.01, 0.1),
                shape=[self._n_hidden_code, self._n_hidden_visit],
                regularizer=self._l2_regularizer)

            i_visit = tf.nn.relu_layer(self._x, c_weight, c_bias)
            v_representation = tf.nn.relu_layer(i_visit, v_weight, v_bias)

            # dense layer representing the softmax classifier
            # we use tf.identity instead of `tf.nn.softmax`
            # so we can use the numerically stable
            # `tf.nn.softmax_cross_entropy_with_logits`
            # TODO: tf.identity if is_training else tf.softmax
            self._transform = tf.layers.dense(
                v_representation,
                self._x.shape[1],
                kernel_regularizer=tf.nn.l2_loss,
                activation=tf.identity)
            self.softmax_transform = tf.nn.softmax(self._transform)

        return self._transform

    @property
    def loss(self):
        if self._loss is None:
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=self._l2_regularization)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self._transform, labels=self._y))

            reg_variables = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(
                l2_regularizer, reg_variables)

            self._loss = loss + reg_term

        return self._loss

    @property
    def optimize(self):
        if self._optimize is None:
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(
                self._learning_rate, global_step, 100000, 0.96, staircase=True)

            self._optimize = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(
                    self._loss, global_step=global_step)


if __name__ == "__main__":
    max_iter_progress_report = 100
    batch_size = 32
    max_epoch = 100
    med2vec_graph = tf.Graph()
    with med2vec_graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, code_dim))
        y = tf.placeholder(tf.float32, shape=(None, code_dim))
        m2v = Med2Vec(x, y)
        init = tf.global_variables_initializer()

    with tf.Session(graph=med2vec_graph) as session:
        init.run()
        loss_values = []
        tmp_loss = []
        # the input output to 
        vec_cycle = itertools.cycle(input_output_generator)
        for i in range(max_epoch):
            x_data, y_data = generate_input_output_batch(
                vec_cycle, batch_size=batch_size)
            feed_dict = {x: x_data, y: y_data}
            _, loss_value = session.run(
                [m2v.optimize, m2v.loss], feed_dict=feed_dict)

            tmp_loss.append(loss_value)
            if max_iter_progress_report is not None and \
               i % max_iter_progress_report == 0:
                print("epoch:", i, "avg_loss:", np.mean(tmp_loss))
                loss_values.append(np.mean(tmp_loss))
                tmp_loss = []

        m2v.transform
