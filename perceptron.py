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
        # assert (isinstance(x, float) or isinstance(x, int))
        if x < 0:
            return -1
        else:
            return 1
    
    def _logistic_activate(self, x):
        """
        Will run the sigmoid function on the given parameter (int or float).

        Returns: float
        """
        assert (isinstance(x, float) or isinstance(x, int))
        return (1.0 / (1.0 + math.exp(-1.0*x)))
    
    def predict(self, x):
        """
        With the given activation function, predict what values the perceptron will give.
        x must be a vector (1xn matrix) (tensorflow / numpy).

        returns: float
        """
        print(x)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        digit = self.sess.run(tf.matmul(x, self.sess.run(self.weights)))[0][0]
        

        if self.logistic:
            return self._logistic_activate(digit)
        else:
            return self._sign_activate(digit)
    
    def train(self, x_values, output_vector, learning_rate):
        """
        Will train the perceptron according to the relevant (in-built) cost function.

        x_values: (1, n) np.matrix where n is the number of features in the example.
        output_vector: (1,) np.matrix which denotes the expected output value for this example.
        learning_rate: The alpha value for this perceptron.
        """

        if self.get_logistic:
            self._train_logistic(x_values, output_vector, learning_rate)
        else:
            self._train_sign(x_values, output_vector, learning_rate)
    
    def _train_logistic(self, x_values, output_vector, learning_rate):
        loss = math.sqrt(math.pow(self.predict(x_values) - output_vector[0], 2)) # Dubious about this step, check it out,
                                                                                    # Might need to use tensors for the GDO
        trainer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(trainer)
    
    def _train_sign(self, x_values, output_vector, learning_rate):
        loss = (self.predict(x_values) - output_vector[0])*learning_rate
        update_values = np.multiply(x_values, loss) # This is what to add to the weights, is the same shape as x_values (1xn)
        np.add(self.weights, np.transpose(update_values)) # weights + updates_transpose = 1xn + 1xn. This is the update.