# shubham
import numpy as np


# np.random.seed(1)  #uncomment to get same values every time

class Layer_Dense:  # layer obeject
    def __init__(self, n_inputs, n_neurons):  # n_inputs = no. of features, n_neurons = no. of outputs
        self.weights = 3 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


def sigmoid(x):  # sigmoid function
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):  # derivative of sigmoid
    return x * (1 - x)


# training Data
training_inputs = np.array([[1, 1, 0],
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 0, 0],
                            [0, 1, 1],
                            [1, 0, 1],
                            [0, 0, 1],
                            [1, 0, 0]])

training_outputs = np.array([[1, 0, 1, 0, 0, 1, 0, 1]]).T
input_layer = Layer_Dense(3, 1)
no_of_iterations = 100000  #more the no. of iterations more the accuracy
for iteration in range(no_of_iterations):
    input_layer.forward(training_inputs)
    outputs = sigmoid(input_layer.output)
    error = training_outputs - outputs
    adjustments = error * d_sigmoid(outputs)
    input_layer.weights += np.dot(training_inputs.T, adjustments)
print(np.round(outputs,3))



