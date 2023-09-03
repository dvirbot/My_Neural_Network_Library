from neuralnetworks import *

neural_network = NeuralNetwork(number_of_inputs=2,loss_function=MeanSquaredLoss(), layers=[])

neural_network.add_layer(DenseLayer(3, 2, RectifiedLinearUnit()))
neural_network.add_layer(DenseLayer(1, 3, ActivationFunction()))

loss_function = MeanSquaredLoss()

guess = neural_network.forwards(inputs=[1,2])
print(guess)
# print(neural_network.layers[0].neurons[1].pre_activation_value)

print(loss_function.calculate_loss(guess, [40]))
loss_derivatives = loss_function.calculate_derivatives(guess, [40])
# print(loss_derivatives)

neural_network.backwards(loss_derivatives)
# print(neural_network.layers[0].neurons[1].weights[0].derivative)
# print(neural_network.layers[0].neurons[1].bias.derivative)
neural_network.descend_the_gradient(learning_rate=0.003)
guess = neural_network.forwards(inputs=[1, 2])
print(guess)
print(loss_function.calculate_loss(guess, [40]))
loss_derivatives = loss_function.calculate_derivatives(guess, [40])

neural_network.backwards(loss_derivatives)
neural_network.descend_the_gradient(learning_rate=0.003)
guess = neural_network.forwards(inputs=[1, 2])
print(guess)
print(loss_function.calculate_loss(guess, [40]))
loss_derivatives = loss_function.calculate_derivatives(guess, [40])
print(loss_derivatives)

neural_network.backwards(loss_derivatives)
neural_network.descend_the_gradient(learning_rate=0.003)
guess = neural_network.forwards(inputs=[1, 2])
print(guess)
print(loss_function.calculate_loss(guess, [40]))



