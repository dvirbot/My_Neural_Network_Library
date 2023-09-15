from neuralnetworks import *

neural_network = NeuralNetwork(size_of_inputs=2)
neural_network.add_layer(DenseLayer(
    number_of_neurons=100,
    size_of_inputs=2,
    activation_function=RectifiedLinearUnit()))
neural_network.add_layer(DenseLayer(
    number_of_neurons=1,
    size_of_inputs=100,
    activation_function=ActivationFunction()))

loss_function = MeanSquaredLoss()

training_data = []
for i in range(1000000):
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    value = ((x**2-y**2)/(x**2+y**2))
    training_data.append(([x, y], value))

test_data = []
for i in range(10000):
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    value = ((x**2-y**2)/(x**2+y**2))
    test_data.append(([x, y], value))

episodes = 100000
batch_size = 100
learning_rate = 0.0001
report_frequency = 100
for i in range(episodes):
    for j in range(batch_size):
        input_values, target_value = random.choice(training_data)
        computed_values = neural_network.forwards(input_values)
        loss_derivatives = loss_function.calculate_derivatives(guesses=computed_values, targets=[target_value])
        neural_network.backwards(loss_function_gradients=loss_derivatives)
    neural_network.descend_the_gradient(learning_rate=(learning_rate / batch_size))
    if i % report_frequency == 0:
        loss = 0
        for test in test_data:
            loss += loss_function.calculate_loss(
                guesses=neural_network.forwards(test[0]),
                targets=[test[1]]
            )[0]
        print(f"Episode: {i}, Avg Loss: {loss / 10000}")

# input_values, target_value = random.choice(training_data)
# computed_values = neural_network.forwards(input_values)
# print(input_values)
# print(target_value)
# print(computed_values)
# print(loss_function.calculate_loss(guesses=computed_values, targets=[target_value]))
# print(loss_function.calculate_derivatives(guesses=computed_values, targets=[target_value]))
