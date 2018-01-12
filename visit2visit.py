__author__ = "Isak Karlsson"

import numpy as np
import tensorflow as tf


import data.io

import data.vectorize as vectorize

#from med2vec import med2vec




def visit2visit():
    visit2visit_graph = tf.Graph()
    with visit2visit_graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 10))
        inputs = tf.split(x, 5, axis=1)
        outputs = tf.unstack(x, axis=1)

    with tf.Session(graph=visit2visit_graph) as session:
        y_v = session.run(
            [outputs], feed_dict={x: np.linspace(-1, 1, 20).reshape(2, -1)})
        print(y_v)


if __name__ == "__main__":
    visit2visit()
