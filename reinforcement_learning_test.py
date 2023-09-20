import neuralnetworks
import reinforcement_learning
import gymnasium as gym
import random
import numpy as np

neural_network = neuralnetworks.NeuralNetwork(size_of_inputs=4, keep_buffer=True)
neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=10,
                                                   size_of_inputs=4,
                                                   activation_function=neuralnetworks.RectifiedLinearUnit()))
neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=10,
                                                   size_of_inputs=10,
                                                   activation_function=neuralnetworks.RectifiedLinearUnit()))
neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=2,
                                                   size_of_inputs=10,
                                                   activation_function=neuralnetworks.ActivationFunction()))
reinforce_agent = reinforcement_learning.ReinforceAgent(neural_network=neural_network,
                                                        num_possible_actions=2,
                                                        future_discount_factor=0.99)



cartpole_env = gym.make("CartPole-v1", render_mode="human")

for j in range(10):
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not terminated or truncated:
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=reinforce_agent.take_step(observations=observation))
        cartpole_env.render()
        observation = observation.tolist()
    reinforce_agent.episode_reset()

cartpole_env.close()

cartpole_env = gym.make("CartPole-v1")

episodes = 10001
learning_rate = 5e-6
report_frequency = 1000

for i in range(episodes):
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not terminated or truncated:
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=reinforce_agent.take_step(observations=observation, reward=reward))
        observation = observation.tolist()
        # cartpole_env.render()
    reinforce_agent.get_final_reward(reward)
    reinforce_agent.update_weights(learning_rate=learning_rate)
    reinforce_agent.episode_reset()
    if i%report_frequency == 0:
        total_reward = 0
        for j in range(100):
            observation: np.ndarray
            observation, info = cartpole_env.reset()
            observation: list = observation.tolist()
            terminated = False
            truncated = False
            reward = None
            while not terminated or truncated:
                observation, reward, terminated, truncated, info = cartpole_env.step(
                    action=reinforce_agent.take_step(observations=observation))
                total_reward += reward
                observation = observation.tolist()
                # cartpole_env.render()

            reinforce_agent.episode_reset()
        print(f"Episode: {i}, avg_reward: {total_reward/100} ")
        # if total_reward/100 > 60:
        #     cartpole_env.reset()
        #     break

cartpole_env.close()
cartpole_env = gym.make("CartPole-v1", render_mode="human")

for j in range(10):
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not terminated or truncated:
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=reinforce_agent.take_step(observations=observation))
        cartpole_env.render()
        observation = observation.tolist()
    reinforce_agent.episode_reset()



