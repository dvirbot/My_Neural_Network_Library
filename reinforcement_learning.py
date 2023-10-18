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
        self.replay_buffer = ReplayBuffer()

    def episode_reset(self):
        self.episode_len = 0

    def take_step(self, observation, reward=None):
        self.episode_len += 1
        relative_action_values = self.neural_network.forwards(inputs=observation)
        action_probability_distribution = softmax(relative_action_values)
        action = random.choices(population=self.action_space, weights=action_probability_distribution, k=1)[0]
        if reward is not None:
            self.replay_buffer.update_buffer(observation, action, reward)
            return action
        self.replay_buffer.first_step_update_buffer(observation, action)
        return action

    def get_final_reward(self, reward):
        self.replay_buffer.last_step_update_buffer(reward)

    def eval_take_action(self, observations):
        relative_action_values = self.neural_network.forwards(inputs=observations)
        action_probability_distribution = softmax(relative_action_values)
        action = random.choices(population=self.action_space, weights=action_probability_distribution, k=1)[0]
        return action

    def update_weights(self, learning_rate):
        # !! Time limits make an environment not a Markov Decision Process unless the time limit is part of the state
        # of the environment, in which case an RL agent may be interested in it!
        learning_rate *= -1
        value_at_timestep = 0
        self.replay_buffer.buffer.reverse()
        for i in range(0, len(self.replay_buffer.buffer)):
            current_experience: Experience = self.replay_buffer.buffer[-i]
            value_at_timestep += current_experience.reward * self.future_discount_factor ** (self.episode_len - i - 1)
            action_probabilities_at_timestep = self.neural_network.forwards(current_experience.observation)
            action_at_timestep = current_experience.action
            gradient = np.multiply((value_at_timestep / action_probabilities_at_timestep[action_at_timestep]),
                                    np.array(softmax_gradient(prob_distribution=action_probabilities_at_timestep,
                                                              action=action_at_timestep))).tolist()
            self.neural_network.backwards(loss_function_gradients=gradient)
            self.neural_network.descend_the_gradient(learning_rate=learning_rate)
        self.replay_buffer.clear_buffer()


class Experience:
    def __init__(self, observation, action, reward, next_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer: list[Experience] = []
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
            if len(self.buffer) >= self.max_size:
                self.full = True
                self.index = 0
        self.observation = observation
        self.action = action
        self.index += 1
        if self.index == self.max_size:
            self.index = 0

    def first_step_update_buffer(self, observation, action):
        self.observation = observation
        self.action = action

    def last_step_update_buffer(self, reward):
        if self.full:
            self.buffer[self.index] = Experience(self.observation, self.action, reward, None)
            self.index += 1
            if self.index == self.max_size:
                self.index = 0
            return
        self.buffer.append(Experience(self.observation, self.action, reward, None))
        if len(self.buffer) >= self.max_size:
            self.full = True
            self.index = 0


    def clear_buffer(self):
        self.buffer: list[Experience] = []


class DQNAgent:
    def __init__(self, neural_network: neuralnetworks.NeuralNetwork, future_discount_factor, replay_buffer_size, epsilon=0.05):
        self.current_q_network = neural_network
        self.target_q_network = copy.deepcopy(self.current_q_network)
        self.action_space = [i for i in range(neural_network.layers[-1].number_of_neurons)]
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.epsilon = epsilon
        self.loss_function = neuralnetworks.MeanSquaredLoss()
        self.future_discount_factor = future_discount_factor

    def episode_reset(self):
        pass

    def take_step(self, observation, reward=None):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.current_q_network.forwards(inputs=observation))
        if reward is not None:
            self.replay_buffer.update_buffer(observation, action, reward)
            return action
        self.replay_buffer.first_step_update_buffer(observation, action)
        return action

    def get_final_reward(self, reward):
        self.replay_buffer.last_step_update_buffer(reward)

    def eval_take_action(self, observation):
        action = np.argmax(self.current_q_network.forwards(inputs=observation))
        return action

    def update_target(self):
        self.target_q_network = copy.deepcopy(self.current_q_network)

    def update_weights(self, learning_rate):
        learning_rate *= 1
        empty_gradient = [0 for action in self.action_space]
        for experience in random.choices(population=self.replay_buffer.buffer, k=32):
            if experience.next_observation is None:
                gradient = copy.deepcopy(empty_gradient)
                gradient[experience.action] = self.loss_function.calculate_derivatives(
                    guesses=[self.current_q_network.forwards(experience.observation)[experience.action]],
                    targets=[experience.reward])[0]
                self.current_q_network.backwards(gradient)
                continue
            gradient = copy.deepcopy(empty_gradient)
            gradient[experience.action] = self.loss_function.calculate_derivatives(
                guesses=[self.current_q_network.forwards(experience.observation)[experience.action]],
                targets=[experience.reward + self.future_discount_factor * self.target_q_network.forwards(
                    experience.next_observation)[experience.action]])[0]
            self.current_q_network.backwards(gradient)
        self.current_q_network.descend_the_gradient(learning_rate)




