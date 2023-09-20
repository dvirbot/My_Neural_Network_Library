import numpy as np
import random
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
    def __init__(self, neural_network: neuralnetworks.NeuralNetwork, num_possible_actions, future_discount_factor=0.99):
        self.neural_network = neural_network
        self.action_space = [i for i in range(num_possible_actions)]
        self.future_discount_factor = future_discount_factor
        self.rewards = []
        self.previous_step_probabilities = []
        self.previous_actions = []
        self.episode_len = 0

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
            value_at_timestep += self.rewards[-i] * self.future_discount_factor ** self.episode_len + 1 - i
            action_probabilities_at_timestep = self.previous_step_probabilities[-i]
            action_at_timestep = self.previous_actions[-i]
            gradients.append(value_at_timestep / action_probabilities_at_timestep[action_at_timestep]
                               * np.array(softmax_gradient(prob_distribution=action_probabilities_at_timestep,
                                                           action=action_at_timestep))
                               )
        gradients.reverse()
        self.neural_network.backwards_with_buffer(loss_function_gradients_list=gradients)
        self.neural_network.descend_the_gradient(learning_rate=learning_rate)
