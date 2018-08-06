import tensorflow as tf
import numpy as np
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
    [F],
    [F],
    [T]
]

percep = perceptron.Perceptron(tf.Variable(tf.ones((3,1)), dtype = "float32"), tf.Session(), logistic = False)
ALPHA = 0.1
TRAINING_ITERATIONS = 20

print("BEFORE: \n")
print("F F: " + str(percep.predict(train_in[0])))
print("F T: " + str(percep.predict(train_in[1])))
print("T F: " + str(percep.predict(train_in[2])))
print("T T: " + str(percep.predict(train_in[3])))
print()

for i in range(TRAINING_ITERATIONS):
    percep.train(train_in, train_out, ALPHA)

print("AFTER:\n")
print("F F: " + str(percep.predict(train_in[0])))
print("F T: " + str(percep.predict(train_in[1])))
print("T F: " + str(percep.predict(train_in[2])))
print("T T: " + str(percep.predict(train_in[3])))
