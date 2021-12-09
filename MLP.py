# Jack Brown
# 201056294
# Computational Intelligence CA - Part Two

import numpy as np


class MLP:
    """
    A multilayer perceptron (MLP).

    """

    def __init__(self, n_inputs, n_hidden_layer_neurons, n_outputs, activation_function="ReLU"):
        """"
        Constructor for MLP class.

        Parameters
            n_inputs (int) - number of input neurons
            n_hidden_layer_neurons (list) - number of neurons in each hidden layer
            n_outputs (int) - number of output neurons
            activation (string) - type of activation function, default = sigmoid

        """
        # User inputs for network structure + activation function
        self.n_inputs = n_inputs
        self.n_hidden_layer_neurons = n_hidden_layer_neurons
        self.n_outputs = n_outputs
        self.activation_function = activation_function

        # List of number of neurons in each layer
        layers = [n_inputs] + n_hidden_layer_neurons + [n_outputs]

        # Initialise activations for each neuron layer
        self.activations = []
        for i in range(len(layers)):
            nodes = np.zeros(layers[i])
            self.activations.append(nodes)

        # Initialise matrix of weights for each weights layer
        self.weights = []
        rng = np.random.default_rng()
        for i in range(len(layers) - 1):
            layer_weights = rng.uniform(size=(layers[i], layers[i + 1]))
            self.weights.append(layer_weights)

        # Initialise matrix of derivatives for each weights layer
        self.derivatives = []
        for i in range(len(layers) - 1):
            derivative = np.zeros(shape=(layers[i], layers[i + 1]))
            self.derivatives.append(derivative)


    def inspect(self):
        """
        Prints activation, weight, and derivative matrices.

        """
        print("Activations:\n" + str(self.activations))
        print("Weights:\n" + str(self.weights))
        print("Derivatives:\n " + str(self.derivatives))


    def sigmoid(self, net_inputs):
        """"
        Applies sigmoidal activation function to input array.

        Parameters
            net_inputs (np.ndarray) - vector of input values

        Returns
            outputs (np.ndarray) - vector of output values

        """
        output = 1.0 / (1.0 + np.exp(-net_inputs))
        return output


    def sigmoid_derivative(self, net_inputs):
        """"
        Applies derivative of sigmoidal activation function to input array.

        Parameters
            net_inputs (np.ndarray) - vector of input values

        Returns
            outputs (np.ndarray) - vector of output values

        """
        return net_inputs * (1.0 - net_inputs)


    def ReLU(self, net_inputs):
        """"
        Applies Rectified Linear Unit activation function to input array.

        Parameters
            net_inputs (np.ndarray) - vector of input values

        Returns
            outputs (np.ndarray) - vector of output values

        """
        return net_inputs * (net_inputs > 0)


    def ReLU_derivative(self, net_inputs):
        """
        Applies the derivative of the Rectified Linear Unit activation function to input array.

        Parameters
            net_inputs (np.ndarray) - vector of input values

        Returns
            outputs (np.ndarray) - vector of output values

        """
        return (net_inputs > 0) * 1


    def forward_pass(self, input):
        """
        Updates activations at each layer and returns the output layer activations.

        Parameters
            input (np.array) - input vector

        Returns
            activations (np.array) - output vector

        """
        # Begin forward propagation - activations equal to inputs
        activations = input

        # Update activations matrix with input values
        self.activations[0] = input

        i = 1
        # Iterate through each weight layer to propagate signal forward
        for layer_weights in self.weights:

            # Calculate input to next layer based on activations in current layer
            # and weight values at first layer of weight matrix

            # Calculate net inputs
            net_inputs = np.dot(activations, layer_weights)

            # Apply activation function
            if self.activation_function == "ReLU":
                activations = self.ReLU(net_inputs)
            elif self.activation_function == "sigmoid":
                activations = self.sigmoid(net_inputs)

            # Update activations matrix
            self.activations[i] = activations

            i += 1

        return activations


    def backward_pass(self, error):
        """
        Updates derivative matrix using backpropagation.

        Parameters
            error (np.array) - vector representing network error

        Returns
            None

        """
        # Iterate backwards over weight layers
        # "i" is index of weight layer.
        for i in reversed(range(len(self.derivatives))):

            # Get previous activation layer (index is "weight layer index + 1")
            activations = self.activations[i + 1]

            # Calculate delta -- error * activations, after activations used as input to
            # derivative of activation function
            if self.activation_function == "ReLU":
                delta = error * self.ReLU_derivative(activations)
            elif self.activation_function == "sigmoid":
                delta = error * self.sigmoid_derivative(activations)
            # Reshape delta for future matrix multiplication
            delta_reshape = delta.reshape(delta.shape[0], -1).T

            # Get activations for next activation layer (backwards) and reshape for matrix multiplication
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            # Update derivatives layer
            self.derivatives[i] = np.dot(current_activations, delta_reshape)

            # Backpropagate error to next layer (backwards)
            error = np.dot(delta, self.weights[i].T)


    def gradient_descent(self, eta=1.0):
        """
        Applies gradient descent to weight matrix.

        Parameters
            eta (float) - learning rate (0,1], controls speed of gradient descent

        Returns
            None

        """
        for i in range(len(self.weights)):
            # Match weight and derivatives layer
            weights_layer = self.weights[i]
            derivatives_layer = self.derivatives[i]

            # Update weight values for layer
            weights_layer = weights_layer + eta * derivatives_layer
            self.weights[i] = weights_layer


    def train(self, input_data, labels, n_iters, eta=0.1, verbose=False):
        """
        Trains network on a labelled dataset.

        Parameters
            X (np.ndarray) - training examples
            Y (np.array) - labels
            n_iters (int) - number of iterations over dataset
            eta (float (0,1]) - learning rate
            verbose (bool) - prints average network error after each iteration

        Returns
            None

        """
        print("Training model...\n")

        # Loop over iterations
        for iter in range(n_iters):

            # Track total mean squared error for iteration
            total_error = 0

            # Loop over training examples
            for input, label in zip(input_data, labels):
                output = self.forward_pass(input)

                error = label - output

                # Add mean squared error to total for iteration
                total_error += np.sum(error ** 2)

                self.backward_pass(error)

                self.gradient_descent(eta)

            avg_error = total_error / len(input_data)

            if verbose:
                print(f"Avg network error at iter={iter + 1} = {avg_error}")




if __name__ == "__main__":
    """
    Experiments below. Run to see experiment results.

    """

    ################
    # Experiment 1 # - Learning sum function
    ################

    print("Beginning experiment 1 - learning a sum function\n")

    from random import random

    # Create training dataset
    dataset_size = 100
    dataset = np.array([[random()/2 for _ in range(2)] for _ in range(dataset_size)])
    labels = np.array([[i[0] + i[1]] for i in dataset])

    # Create model, define structure
    model = MLP(2, [2], 1, activation_function="ReLU")

    # Train model
    model.train(dataset, labels, n_iters=20, eta=0.1, verbose=True)

    # Attempt sum function
    num1 = 1532
    num2 = 2341
    output = model.forward_pass(np.array([num1, num2]))

    print("Test output...")
    print(f"{num1} + {num2} = {output}\n")


    ################
    # Experiment 2 #  - Learning XOR function
    ################

    print("\nBeginning experiment 2 - learning XOR function\n")

    # Create training dataset
    dataset = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    labels = np.array([[1], [0], [0], [1]])

    # Create model, define structure
    model = MLP(2, [3], 1, activation_function="ReLU")

    # Train model
    model.train(dataset, labels, n_iters=10000, eta=0.1, verbose=False)

    # Attempt XOR function
    test_input1 = np.array([0, 1])
    test_input2 = np.array([1, 1])
    output1 = model.forward_pass(test_input1)
    output2 = model.forward_pass(test_input2)
    print(f"Input = {test_input1} : Output = {output1}")
    print(f"Input = {test_input2} : Output = {output2}")

