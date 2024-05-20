########################################################################################################################
# IMPORTS
########################################################################################################################


import numpy as np
import math

########################################################################################################################
# PART 1 // IMPLEMENT PCA (2 components) using ONLY scikit learn
########################################################################################################################


'''
Implement a simple feed forward neural network based ont he code below (copy & paste is not enough!)
-There should be: 2 inputs, a first hidden layer with 3 nodes, a second hidden layer with 4 nodes and one output node.
-Use random weights and biases.
-Make up a fixed target (e.g. 0.7) and inputs (e.g. 9 and 3) and calculate the loss 
    between the output of your network and the target 
    (in this case a simple difference is enough, e.g. 0.8 – 0.7 = 0.1)
-Run it multiple times and note down the weights and biases for the lowest loss. 
    (Next time we will learn how to use better-than-random weights and biases!)
'''


# ----------------------------------------------------------------------------------------------------------------------
# adapt the above code/input s.t. a forward pass of a NN with a 2-3-4-1 architecture can be calculated


def activation_sigmoid(val):
    """
    :param val: float (here a linear combination of neuron inputs, weights and biases) that should be activated
    :return: float
    """

    s = 1 / (1 + math.exp(-val))
    return s


def neuron_output(incoming_neuron_weights_and_bias_weight, incoming_values_and_bias):
    """
    :param incoming_neuron_weights_and_bias_weight: list of floats
    :param incoming_values_and_bias: list of floats
    """

    d = np.dot(incoming_neuron_weights_and_bias_weight, incoming_values_and_bias)
    s = activation_sigmoid(d)
    return s


def forward_propagation(neural_network, input_vector, verbosity=2):
    """
    :param neural_network: list of neurons, neurons are lists of weights and biases, weights and biases are floats
    :param input_vector: list of floats
    :param verbosity: controls print
    :return:
    """

    value_flow = input_vector
    layer_idx = 0

    for layer in neural_network:
        new_value_flow = []
        for incoming_neuron_weights in layer:
            incoming_values_and_bias = value_flow + [1.0]
            new_value_flow.append(neuron_output(incoming_neuron_weights, incoming_values_and_bias))

            if verbosity >= 2:
                print("Processing neuron: ", incoming_neuron_weights)

        value_flow = new_value_flow

        if verbosity >= 1:
            print("\nProcessing layer: ", layer_idx, layer)
            print("incoming values and bias:", value_flow)

        layer_idx += 1

    return value_flow


def randomArchitecture(architecture):
    nn = []

    for i in range(len(architecture) - 1):
        layer = []
        for j in range(architecture[i + 1]):
            neuron = []
            for k in range(architecture[i] + 1):
                neuron.append(np.random.randint(1, 10) / 10)
            layer.append(neuron)
        nn.append(layer)

    return (nn)


# ----------------------------------------------------------------------------------------------------------------------
# run the forward pass with random weights/biases, calculate the error
output = 0.7
input = [9, 3]
architecture = [2, 3, 4, 1]


for i in np.arange(100):
    print("Iteration " + str(i) + ": ")
    nn = randomArchitecture(architecture)
    out = forward_propagation(nn, input, verbosity=2)
    error = abs(out[0] - output)
    print("error: ", error)
    print()
