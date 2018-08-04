import tensorflow as tf
import numpy as np 

class Perceptron(object):
    """
    This is a class that will model a perceptron in a neural network.

    Parameters:


    weights: A matrix of shape (n, 1), where n is the number of inputs (ideally dtype = float32).
    
    logistic: A boolean which determines if you want to use the logistic regression function (if false will use sign).

    """

    def __init__(self, weights, logistic = True):
        
        assert isinstance(weights, np.matrix)
        assert isinstance(logistic, bool)
        self.weights = weights
        self.logistic = logistic
