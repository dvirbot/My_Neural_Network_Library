import numpy as np
import random
import copy
import neuralnetworks


def softmax(values):
    exponentiated_probabilities_summed = 0
    for value in values:
        exponentiated_probabilities_summed += np.exp(value)
    return [np.exp(value) / exponentiated_probabilities_summed for value in values]


def softmax_gradient(prob_distribution, action):
    derivatives = []
    for potential_action in range(len(prob_distribution)):
        if potential_action == action:
            derivatives.append(prob_distribution[action] * (1 - prob_distribution[action]))
        else:
            derivatives.append(-prob_distribution[action] * prob_distribution[potential_action])
    return derivatives


class ReinforceAgent:
    def __init__(self, neural_network: neuralnetworks.NeuralNetwork, future_discount_factor=0.99):
        self.neural_network = neural_network
        self.action_space = [i for i in range(neural_network.layers[-1].number_of_neurons)]
        self.future_discount_factor = future_discount_factor
        self.episode_reset()

    def episode_reset(self):
        self.rewards = []
        self.previous_step_probabilities = []
        self.previous_actions = []
        self.episode_len = 0

    def take_step(self, observations, reward=None):
        self.episode_len += 1
        if reward is not None:
            self.rewards.append(reward)
        relative_action_values = self.neural_network.forwards(inputs=observations)
        action_probability_distribution = softmax(relative_action_values)
        self.previous_step_probabilities.append(action_probability_distribution)
        action = random.choices(population=self.action_space, weights=action_probability_distribution)[0]
        self.previous_actions.append(action)
        return action

    def get_final_reward(self, reward):
        self.rewards.append(reward)

    def eval_take_action(self, observations):
        relative_action_values = self.neural_network.forwards(inputs=observations)
        action_probability_distribution = softmax(relative_action_values)
        action = random.choices(population=self.action_space, weights=action_probability_distribution)[0]
        return action

    def update_weights(self, learning_rate):
        # !! Time limits make an environment not a Markov Decision Process unless the time limit is part of the state
        # of the environment, in which case an RL agent may be interested in it!
        learning_rate *= -1
        gradients = []
        value_at_timestep = 0
        for i in range(1, self.episode_len + 1):
            value_at_timestep += self.rewards[-i] * self.future_discount_factor ** (self.episode_len - i)
            action_probabilities_at_timestep = self.previous_step_probabilities[-i]
            action_at_timestep = self.previous_actions[-i]
            gradients.append(np.multiply((value_at_timestep / action_probabilities_at_timestep[action_at_timestep]),
                                         np.array(softmax_gradient(prob_distribution=action_probabilities_at_timestep,
                                                                   action=action_at_timestep))).tolist()
                             )
        gradients.reverse()
        self.neural_network.backwards(loss_function_gradients=gradients)
        self.neural_network.descend_the_gradient(learning_rate=learning_rate)

class Experience:
    def __init__(self, observation, action, reward, next_observation):
        self.bservation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer : list[Experience] = []
        self.max_size = max_size
        self.index = -1
        self.full = False
        self.observation = None
        self.action = None

    def update_buffer(self, observation, action, reward):
        if self.full:
            self.buffer[self.index] = Experience(self.observation, self.action, reward, observation)
        elif not self.index == -1:
            self.buffer.append(Experience(self.observation, self.action, reward, observation))
        self.observation = observation
        self.action = action
        self.index += 1
        if self.index == self.max_size:
            self.index = 0


class DQNAgent:
    def __init__(self, neural_network: neuralnetworks.NeuralNetwork, future_discount_factor, replay_buffer_size):
        self.current_q_network = neural_network
        self.previous_q_network = copy.deepcopy(self.current_q_network)
        self.action_space = [i for i in range(neural_network.layers[-1].number_of_neurons)]
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    def take_step(self, observation, reward=None):
        self.current_q_network.forwards()


