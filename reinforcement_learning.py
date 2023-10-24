import numpy as np
import random
import copy
import neuralnetworks
import _pickle


def softmax(values):
    values = np.array(values)
    max_value = values.max()
    values -= max_value
    ret: np.ndarray = np.exp(values) / np.exp(values).sum()
    return ret.tolist()


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
    def __init__(self, neural_network: neuralnetworks.NeuralNetwork, future_discount_factor, replay_buffer_size,
                 epsilon=0.05):
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
        # self.target_q_network = copy.deepcopy(self.current_q_network)
        # I am using this instead of the commented deepcopy code because deepcopy was very slow for some reason
        self.target_q_network = _pickle.loads(_pickle.dumps(self.current_q_network))

    def update_weights(self, learning_rate):
        empty_gradient = [0 for action in self.action_space]
        for experience in random.choices(population=self.replay_buffer.buffer, k=32):
            if experience.next_observation is None:
                gradient = [0 for action in self.action_space]
                gradient[experience.action] = self.loss_function.calculate_derivatives(
                    guesses=[self.current_q_network.forwards(experience.observation)[experience.action]],
                    targets=[experience.reward])[0]
                self.current_q_network.backwards(gradient)
                continue
            gradient = [0 for action in self.action_space]
            gradient[experience.action] = self.loss_function.calculate_derivatives(
                guesses=[self.current_q_network.forwards(experience.observation)[experience.action]],
                targets=[experience.reward + self.future_discount_factor * self.target_q_network.forwards(
                    experience.next_observation)[experience.action]])[0]
            self.current_q_network.backwards(gradient)
        self.current_q_network.descend_the_gradient(learning_rate)

    def epsilon_decay(self, factor):
        self.epsilon *= factor


class ActorCriticAgent:
    def __init__(self, policy_network: neuralnetworks.NeuralNetwork, value_network: neuralnetworks.NeuralNetwork,
                 future_discount_factor, replay_buffer_size):
        self.current_value_network = value_network
        self.target_value_network = copy.deepcopy(self.current_value_network)
        self.policy_network = policy_network
        self.action_space = [i for i in range(self.policy_network.layers[-1].number_of_neurons)]
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.loss_function = neuralnetworks.MeanSquaredLoss()
        self.future_discount_factor = future_discount_factor

    def episode_reset(self):
        pass

    def take_step(self, observation, reward=None):
        action_outputs = self.policy_network.forwards(observation)
        action_probabilities = softmax(action_outputs)
        action = random.choices(population=self.action_space, weights=action_probabilities, k=1)[0]
        if reward is not None:
            self.replay_buffer.update_buffer(observation, action, reward)
            return action
        self.replay_buffer.first_step_update_buffer(observation, action)
        return action

    def get_final_reward(self, reward):
        self.replay_buffer.last_step_update_buffer(reward)

    def eval_take_action(self, observation):
        action_outputs = self.policy_network.forwards(observation)
        action_probabilities = softmax(action_outputs)
        action = random.choices(population=self.action_space, weights=action_probabilities, k=1)[0]
        return action

    def update_target(self):
        # self.target_q_network = copy.deepcopy(self.current_q_network)
        # I am using this instead of the commented deepcopy code because deepcopy was very slow for some reason
        self.target_value_network = _pickle.loads(_pickle.dumps(self.current_value_network))

    def update_weights(self, policy_learning_rate, value_learning_rate):
        policy_learning_rate *= -1
        for experience in random.choices(population=self.replay_buffer.buffer, k=32):
            if experience.next_observation is None:
                action_probabilities = self.policy_network.forwards(experience.observation)
                learning_factor = (experience.reward - self.current_value_network(experience.observation)[0]) / \
                                  action_probabilities[experience.action]
                policy_gradients: np.ndarray = np.multiply(
                    softmax_gradient(prob_distribution=action_probabilities, action=experience.action),
                    learning_factor).tolist()
                self.policy_network.backwards(policy_gradients)
                continue
            action_probabilities = self.policy_network.forwards(experience.observation)
            learning_factor = (experience.reward + self.future_discount_factor * self.current_value_network(
                experience.next_observation)[0] - self.current_value_network(experience.observation)[0]) / \
                              action_probabilities[experience.action]
            policy_gradients: np.array = np.multiply(
                softmax_gradient(prob_distribution=action_probabilities, action=experience.action),
                learning_factor).tolist()
            self.policy_network.backwards(policy_gradients)
        for experience in random.choices(population=self.replay_buffer.buffer, k=128):
            if experience.next_observation is None:
                value_gradients = self.loss_function.calculate_derivatives(
                    guesses=self.current_value_network.forwards(experience.observation), targets=[experience.reward])
                self.current_value_network.backwards(value_gradients)
                continue
            value_gradients = self.loss_function.calculate_derivatives(
                guesses=self.current_value_network.forwards(experience.observation),
                targets=[experience.reward + self.future_discount_factor * self.current_value_network(
                    experience.next_observation)[0]])
            self.current_value_network.backwards(value_gradients)
        self.policy_network.descend_the_gradient(policy_learning_rate)
        self.current_value_network.descend_the_gradient(value_learning_rate)

    def epsilon_decay(self, factor):
        self.epsilon *= factor
