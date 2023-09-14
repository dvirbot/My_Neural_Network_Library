from neuralnetworks import *

neural_network = NeuralNetwork(size_of_inputs=2)
neural_network.add_layer(DenseLayer(
    number_of_neurons=20,
    size_of_inputs=2,
    activation_function=RectifiedLinearUnit()))
neural_network.add_layer(DenseLayer(
    number_of_neurons=20,
    size_of_inputs=10,
    activation_function=RectifiedLinearUnit()))
neural_network.add_layer(DenseLayer(
    number_of_neurons=1,
    size_of_inputs=10,
    activation_function=ActivationFunction()))

loss_function = MeanSquaredLoss()

data = []
for i in range(100000):
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    value = (x**2/(y+6))+((x+2)/y)**3
    data.append(([x, y], value))


episodes = 100000
batch_size = 100
learning_rate = 0.001
report_frequency = 100
for i in range(episodes):
    for j in range(batch_size):
        input_values, target_value = random.choice(data)
        computed_value = neural_network.forwards(input_values)
        loss_function



