import tensorflow as tf
import numpy as np 
import math

class Perceptron(object):
    """
    This is a class that will model a perceptron in a neural network.

    Parameters:

    starting_weights: A tf.Variable object of shape (n, 1), where n is the number of inputs (ideally dtype = float32).

    session: tensorflow.Session() object to be used in training.
    
    logistic: A boolean which determines if you want to use the logistic regression function as
    the activation function (if false will use sign).
    """

    def __init__(self, starting_weights, session, logistic = True):
        
        assert isinstance(starting_weights, tf.Variable)
        assert isinstance(logistic, bool)
        self.weights = starting_weights
        self.logistic = logistic
        self.sess = session
    
    def get_weights(self):
        """
        Returns a matrix of weights.
        """
        return self.weights
    
    def set_weights(self, arg):
        """
        Sets the weights to a custom value. Must be a matrix in the shape of (?, 1).
        """
        assert (np.shape(arg) == np.shape(np.zeros((None, 1))))
        self.weights = arg
    
    def get_logistic(self):
        """
        Returns whether or not a logistic regression function is being used as the activation function.
        """
        return self.logistic
    
    def toggle_logistic(self):
        """
        Toggles between the logistic and the sign activation function.
        """
        self.logistic = not self.logistic

    def _sign_activate(self, x):
        """
        Will return the sign of the given parameter (int or float). 
        """
        assert (isinstance(x, float) or isinstance(x, int))
        if x < 0:
            return -1
        else:
            return 1
    
    def _logistic_activate(self, x):
        """
        Will run the sigmoid function on the given parameter (int or float).
        """
        assert (isinstance(x, float) or isinstance(x, int))
        return (1 / (1 + math.exp(-1*x)))
    
    def predict(self, x):
        """
        With the given activation function, predict what values the perceptron will give.
        x must be a vector (nx1 matrix) (tensorflow / numpy).
        """
        assert(isinstance(x, np.matrix))
        digit = tf.matmul(self.weights, x, transpose_b = True)[0][0]

        if self.logistic:
            return self._logistic_activate(digit)
        else:
            return self._sign_activate(digit)
    
    def train(self, x_values, output_vector, learning_rate):
        """
        Will train the perceptron according to the relevant (in-built) cost function.

        x_values: (1, m) np.matrix where m is the number of features in the example.
        output_vector: (1,) np.matrix which denotes the expected output value for this example.
        learning_rate: The alpha value for this perceptron.
        """

        if self.get_logistic:
            self._train_logistic(x_values, output_vector, learning_rate)
        else:
            self._train_sign(x_values, output_vector, learning_rate)
    
    def _train_logistic(x_values, output_vector, learning_rate):
        pass
    
    def _train_sign(x_values, output_vector, learning_rate):
        pass