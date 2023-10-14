import neuralnetworks
import reinforcement_learning
import gymnasium as gym
import random
import numpy as np

neural_network = neuralnetworks.NeuralNetwork(size_of_inputs=4, keep_buffer=True)
neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=10,
                                                   size_of_inputs=4,
                                                   activation_function=neuralnetworks.Sigmoid()))
neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=10,
                                                   size_of_inputs=10,
                                                   activation_function=neuralnetworks.Sigmoid()))
neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=2,
                                                   size_of_inputs=10,
                                                   activation_function=neuralnetworks.ActivationFunction()))
reinforce_agent = reinforcement_learning.ReinforceAgent(neural_network=neural_network,
                                                        future_discount_factor=1)

cartpole_env = gym.make("CartPole-v1", render_mode="human")

for j in range(10):
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    cartpole_env.render()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not terminated or truncated:
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=reinforce_agent.take_step(observations=observation))
        observation = observation.tolist()
    reinforce_agent.episode_reset()

cartpole_env.close()

cartpole_env = gym.make("CartPole-v1")

episodes = 100001
learning_rate = 2**(-13)
report_frequency = 1000
total_reward = 0

for i in range(episodes):
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=reinforce_agent.take_step(observations=observation, reward=reward))
        observation = observation.tolist()
        total_reward += reward
        # cartpole_env.render()
    reinforce_agent.get_final_reward(reward)
    reinforce_agent.update_weights(learning_rate)
    reinforce_agent.episode_reset()
    if i%report_frequency == 0:
        print(f"Episode: {i}, avg_reward: {total_reward / report_frequency} ")
        if total_reward/1000 > 60:
            cartpole_env.reset()
            break
        total_reward = 0

cartpole_env.close()
cartpole_env = gym.make("CartPole-v1", render_mode="human")

for j in range(10):
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    cartpole_env.render()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not terminated or truncated:
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=reinforce_agent.take_step(observations=observation))
        observation = observation.tolist()
    reinforce_agent.episode_reset()



