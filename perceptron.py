import tensorflow as tf
import numpy as np 
import math

class Perceptron(object):
    """
    This is a class that will model a perceptron in a neural network.

    Parameters:

    starting_weights: A matrix of shape (n, 1), where n is the number of inputs (ideally dtype = float32).
    
    logistic: A boolean which determines if you want to use the logistic regression function as
    the activation function (if false will use sign).
    """

    def __init__(self, starting_weights, logistic = True):
        
        assert isinstance(starting_weights, np.matrix)
        assert isinstance(logistic, bool)
        self.weights = starting_weights
        self.logistic = logistic
    
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

    def sign_activate(self, x):
        """
        Will return the sign of the given parameter (int or float). 
        """
        assert (isinstance(x, float) or isinstance(x, int))
        if x < 0:
            return -1
        else:
            return 1
    
    def logistic_activate(self, x):
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
        digit = tf.matmul(self.weights, x, transpose_b = True)

        if self.logistic:
            return self.logistic_activate(digit)
        else:
            return self.sign_activate(digit)
        