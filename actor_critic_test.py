import neuralnetworks
import reinforcement_learning
import gymnasium as gym
import random
import numpy as np

policy_neural_network = neuralnetworks.NeuralNetwork(size_of_inputs=4, keep_buffer=False)
policy_neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=20,
                                                          size_of_inputs=4,
                                                          activation_function=neuralnetworks.Sigmoid()))
policy_neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=20,
                                                          size_of_inputs=20,
                                                          activation_function=neuralnetworks.Sigmoid()))
policy_neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=2,
                                                          size_of_inputs=20,
                                                          activation_function=neuralnetworks.ActivationFunction()))
value_neural_network = neuralnetworks.NeuralNetwork(size_of_inputs=4, keep_buffer=False)
value_neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=15,
                                                         size_of_inputs=4,
                                                         activation_function=neuralnetworks.Sigmoid()))
value_neural_network.add_layer(neuralnetworks.DenseLayer(number_of_neurons=1,
                                                         size_of_inputs=15,
                                                         activation_function=neuralnetworks.ActivationFunction()))
# neural_network.load_weights("First DQN Saved Eons (19_10_23)/DQN_Eon_5")
actor_critic_agent = reinforcement_learning.ActorCriticAgent(policy_network=policy_neural_network,
                                                             value_network=value_neural_network,
                                                             future_discount_factor=0.98, replay_buffer_size=10000)

# cartpole_env = gym.make("CartPole-v1", render_mode="human")
#
# for j in range(10):
#     observation: np.ndarray
#     observation, info = cartpole_env.reset()
#     cartpole_env.render()
#     observation: list = observation.tolist()
#     terminated = False
#     truncated = False
#     reward = None
#     while not (terminated or truncated):
#         observation, reward, terminated, truncated, info = cartpole_env.step(
#             action=actor_critic_agent.eval_take_action(observation=observation))
#         observation = observation.tolist()
#     actor_critic_agent.episode_reset()
#
# cartpole_env.close()

cartpole_env = gym.make("CartPole-v1")

episodes = 100001
policy_learning_rate = 2 ** (-11)
value_learning_rate = 2 ** (-10)
report_frequency = 100
eval_frequency = 1000
total_reward = 0
step = 0
learn_frequency = 32
update_target_frequency = 100
automatic_eon_threshold = 100
eon = 0

for i in range(episodes):
    # if i % 50 == 0:
    #     print(f"episode {i}")
    observation: np.ndarray
    observation, info = cartpole_env.reset()
    observation: list = observation.tolist()
    terminated = False
    truncated = False
    reward = None
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = cartpole_env.step(
            action=actor_critic_agent.take_step(observation=observation, reward=reward))
        observation = observation.tolist()
        total_reward += reward
        step += 1
        if step % learn_frequency == 0 and step > 10000:
            actor_critic_agent.update_weights(policy_learning_rate=policy_learning_rate, value_learning_rate=value_learning_rate)
        if step % update_target_frequency == 0 and step > 10000:
            actor_critic_agent.update_target()
        # cartpole_env.render()
    actor_critic_agent.get_final_reward(reward)
    # actor_critic_agent.update_weights(policy_learning_rate=policy_learning_rate, value_learning_rate=value_learning_rate)
    actor_critic_agent.episode_reset()
    if i % report_frequency == 0:
        print(f"Episode: {i}, avg_reward: {total_reward / report_frequency} ")
        # if total_reward/1000 > 60:
        #     cartpole_env.reset()
        #     break
        total_reward = 0
    if i % eval_frequency == 0 and i > 1 or total_reward / report_frequency > automatic_eon_threshold:
        eon += 1
        eval_reward = 0
        for j in range(1000):
            observation: np.ndarray
            observation, info = cartpole_env.reset()
            observation: list = observation.tolist()
            terminated = False
            truncated = False
            reward = None
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = cartpole_env.step(
                    action=actor_critic_agent.eval_take_action(observation=observation))
                eval_reward += reward
        print(f"Eon: {eon}, avg_reward: {eval_reward / 1000} ")
        if total_reward / report_frequency > automatic_eon_threshold:
            automatic_eon_threshold *= 1.5
        # actor_critic_agent.current_q_network.save_weights(f"DQN_Eon_{eon}")

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
            action=actor_critic_agent.take_step(observation=observation))
        observation = observation.tolist()
    actor_critic_agent.episode_reset()
