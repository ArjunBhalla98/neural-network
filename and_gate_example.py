import tensorflow as tf
import math 
import perceptron

T, F, bias = 1.0, -1.0, 1.0

train_in = [
    [bias, F, F],
    [bias, F, T],
    [bias, T, F],
    [bias, T, T]
]

train_out = [
    [F],
    [T],
    [T],
    [T]
]

percep = perceptron.Perceptron(tf.Variable(tf.ones((3,1))), tf.Session(), False)

print(percep.predict([train_in[0]]))
