# shubham
import numpy as np
import time
# np.random.seed(1)  #uncomment to disable randomization

class Layer:  # layer obeject
    def __init__(self, inputs, neurons):  # inputs = no. of features, neurons = no. of outputs
        self.weights = 3 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

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


input_layer = Layer(3, 1)
no_of_iterations = 1000  #more the no. of iterations more the accuracy
pause_time = 1 #make this 0 for normal execution

for iteration in range(no_of_iterations):
    input_layer.forward(training_inputs)
    outputs = sigmoid(input_layer.output)
    
    error = training_outputs - outputs
    adjustments = error * d_sigmoid(outputs)
    input_layer.weights += np.dot(training_inputs.T, adjustments)
    
    time.sleep(pause_time) # for real time observation  
    #print('actual predicted outputs :', outputs.T)  #uncomment to see actual predicted values
    print('iteration no.',iteration+1)
    print('correct outputs : ', training_outputs.T)
    print('predicted outputs :', np.round(outputs).T)
    
    if list(training_outputs) == list(np.round(outputs)):
        print('correct output achieved at',str(iteration+1)+'th iteration.')
        break




