import numpy as np
import random


class NeuralNetwork:
    def __init__(self, size_of_inputs: int, layers=None, keep_buffer=False):
        self.size_of_inputs = size_of_inputs
        if layers is None:
            self.layers: list[DenseLayer] = []
        else:
            self.layers: list[DenseLayer] = layers
        self.inputs = []
        if keep_buffer:
            self.keep_buffer = True
            self.buffer = []
        else:
            self.keep_buffer = False

    def add_layer(self, layer):
        self.layers.append(layer)

    def forwards(self, inputs: list):
        if self.keep_buffer:
            return self.forwards_with_buffer(inputs)
        self.inputs = inputs
        last_layer_outputs = inputs
        for layer in self.layers:
            last_layer_outputs = layer.forwards(last_layer_outputs)
        return last_layer_outputs

    def forwards_with_buffer(self, inputs: list):
        values_snapshot = []
        self.inputs = inputs
        last_layer_outputs = inputs
        for i in range(len(self.layers)):
            values_snapshot.append(last_layer_outputs)
            last_layer_outputs = self.layers[i].forwards(last_layer_outputs)
        values_snapshot.append(last_layer_outputs)
        self.buffer.append(values_snapshot)
        return last_layer_outputs

    def backwards(self, loss_function_gradients):
        if self.keep_buffer:
            self.backwards_with_buffer(loss_function_gradients)
        current_layer_derivatives = loss_function_gradients
        for i in range(1, len(self.layers), 1):
            current_layer_derivatives = self.layers[-i].backwards(current_layer_derivatives,
                                                                  self.layers[-i - 1].values)
        # Input derivatives are for architectures where there may be multiple neural networks, and the results of one
        # NN may be the inputs to another.
        input_derivatives = self.layers[0].backwards(current_layer_derivatives,
                                 self.inputs)
        return input_derivatives

    def backwards_with_buffer(self, loss_function_gradients_list):
        for i in range(len(loss_function_gradients_list)):
            current_layer_derivatives = loss_function_gradients_list[i]
            for j in range(1, len(self.layers), 1):
                current_layer_derivatives = self.layers[-j].backwards(current_layer_derivatives,
                                                                      self.buffer[i][-j - 1])
            self.layers[0].backwards(current_layer_derivatives,
                                     self.buffer[i][0])

    def descend_the_gradient(self, learning_rate):
        for layer in self.layers:
            layer.descend_the_gradient(learning_rate)

    def save_weights(self, filename):
        with open(filename, 'w') as file:
            for layer in self.layers:
                for neuron in layer.neurons:
                    for weight in neuron.weights:
                        file.write(str(weight.value)+"\n")
                    file.write(str(neuron.bias.value)+"\n")

    def load_weights(self, filepath):
        with open(filepath, 'r') as file:
            weights = file.readlines()
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    weight.value = float(weights.pop(0))
                neuron.bias.value = float(weights.pop(0))




class DenseLayer:
    def __init__(self, number_of_neurons: int, size_of_inputs, activation_function):
        self.neurons = [Neuron(size_of_inputs, activation_function) for i in range(number_of_neurons)]
        self.values = []

    def forwards(self, inputs: list):
        self.values = []
        for neuron in self.neurons:
            self.values.append(neuron.forwards(inputs))
        return self.values

    def backwards(self, current_layer_derivatives, prev_layer_values):
        """
        Note to self: You must figure out the derivatives of the last layer directly given the loss function,
        and use those in the first call
        """
        prev_layer_derivatives = [0 for i in range(len(prev_layer_values))]
        for i in range(len(self.neurons)):
            neuron_contribution_to_derivatives = self.neurons[i].backwards(current_layer_derivatives[i],
                                                                           prev_layer_values)
            prev_layer_derivatives = [prev_derivative + neuron_contribution
                                      for prev_derivative, neuron_contribution
                                      in zip(prev_layer_derivatives, neuron_contribution_to_derivatives)]
        return prev_layer_derivatives

    def descend_the_gradient(self, learning_rate):
        for neuron in self.neurons:
            neuron.descend_the_gradient(learning_rate)


class Neuron:
    def __init__(self, size_of_inputs, activation_function):
        self.weights = [Weight() for i in range(size_of_inputs)]
        self.bias = Weight()
        self.activation_function: ActivationFunction = activation_function
        self.value = 0
        self.pre_activation_value = 0
        self.pre_activation_value_buffer = []

    def forwards(self, inputs: list):
        self.pre_activation_value = 0
        for i in range(len(inputs)):
            self.pre_activation_value += inputs[i] * self.weights[i].value
        self.pre_activation_value += self.bias.value
        self.pre_activation_value_buffer.append(self.pre_activation_value)
        self.value = self.activation_function.apply(self.pre_activation_value)
        return self.value

    def backwards(self, derivative, prev_layer_values):
        pre_activation_derivative = self.activation_function.derivative(derivative, self.pre_activation_value)
        self.bias.update_derivatives(pre_activation_derivative)
        prev_layer_derivatives = []
        for i in range(len(self.weights)):
            self.weights[i].update_derivatives(prev_layer_values[i] * pre_activation_derivative)
            prev_layer_derivatives.append(self.weights[i].value * pre_activation_derivative)
        return prev_layer_derivatives

    def descend_the_gradient(self, learning_rate):
        self.bias.descend_the_gradient(learning_rate)
        for weight in self.weights:
            weight.descend_the_gradient(learning_rate)


class Weight:
    def __init__(self):
        self.value = random.gauss(mu=0, sigma=1)
        # self.value = 2
        self.derivative = 0

    def update_derivatives(self, derivative):
        self.derivative += derivative

    def descend_the_gradient(self, learning_rate):
        self.value -= self.derivative * learning_rate
        self.derivative = 0


class ActivationFunction:
    def __init__(self):
        pass

    def apply(self, neuron_value):
        return neuron_value

    def derivative(self, neuron_derivative, pre_activation_value):
        return neuron_derivative


class Sigmoid(ActivationFunction):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def apply(self, neuron_value):
        return self.sigmoid(neuron_value)

    def derivative(self, neuron_derivative, pre_activation_value):
        return self.sigmoid(pre_activation_value) * self.sigmoid(1 - pre_activation_value) * neuron_derivative


class RectifiedLinearUnit(ActivationFunction):
    def __init__(self):
        super(RectifiedLinearUnit, self).__init__()

    def relu(self, x):
        if x > 0: return x
        return 0

    def apply(self, neuron_value):
        return self.relu(neuron_value)

    def derivative(self, neuron_derivative, pre_activation_value):
        if pre_activation_value > 0:
            return neuron_derivative
        return 0


class MeanSquaredLoss:
    def __init__(self):
        pass

    def calculate_loss(self, guesses: list, targets: list):
        return [(guesses[i] - targets[i]) ** 2 for i in range(len(guesses))]

    def calculate_derivatives(self, guesses: list, targets: list):
        return [2 * (guesses[i] - targets[i]) for i in range(len(guesses))]