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
        Sets the weights to a custom value. Must be a tf.Variable in the shape of (n, 1).
        """
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
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        digit = self.sess.run(tf.matmul([x], tf.cast(self.weights, tf.float32)))[0][0]
        

        if self.get_logistic():
            return self._logistic_activate(digit)
        else:
            return self._sign_activate(digit)
    
    def train(self, x_values, output_vector, learning_rate):
        """
        Will train the perceptron according to the relevant (in-built) cost function.

        x_values: (m, n) np.matrix where n is the number of features in the example, m is the # of examples.
        output_vector: (m, 1) matrix which denotes the expected output value for this example.
        learning_rate: The alpha value for this perceptron.
        """

        if self.get_logistic():
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
        
        predicted_output = tf.matmul(x_values, self.get_weights()) # Gets the what the perceptron thinks the output should be (Vector mx1)
        loss = tf.reduce_sum(tf.multiply(tf.subtract(predicted_output, output_vector), learning_rate))
        print(self.sess.run(loss))
        update_term = tf.multiply(x_values, self.sess.run(loss))
        self.set_weights(tf.add(self.get_weights(), update_term))

        # loss = (self.predict(x_values) - output_vector[0])*learning_rate
        # print("Loss: " + str(loss))
        # update_values = np.multiply(x_values, loss) # This is what to add to the weights, is the same shape as x_values (1xn)
        # print("Update vals: " + str(update_values))
        # update = tf.Variable(np.add(self.sess.run(self.weights), np.transpose(update_values))) # weights + updates_transpose = 1xn + 1xn. This is the update.
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        # print("New weights: " + str(self.sess.run(update)))
        # self.set_weights(update)