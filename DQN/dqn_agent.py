import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from env_config import *


def build_model(input_dim, output_dim, num_mds):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim * num_mds),
    )


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        action_indices = []
        for a in action:
            if a == 0.0:
                action_indices.append(0)
            else:
                idx = DISCRETE_POWERS.index(a)
                action_indices.append(idx)
        self.buffer.append((state, action_indices, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env,
                 state_dim,
                 action_space_size,
                 max_epsilon,
                 min_epsilon,
                 epsilon_decay_rate=0.9,
                 lr=0.001,
                 gamma=0.9,
                 batch_size=32,
                 buffer_capacity=10000,
                 target_update_freq=100,
                 optimizer_type='adam'):
        self.env = env
        self.num_mds, self.discrete_powers = env.get_env_params()

        self.state_space_size = state_dim
        self.action_space_size = action_space_size
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy params
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = max_epsilon

        # Q network (main)
        self.q_network = build_model(state_dim, action_space_size, self.num_mds)

        # Q Network (target)
        self.target_network = build_model(state_dim, action_space_size, self.num_mds)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.q_network.parameters(), lr=lr, momentum=0.9)

        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.step_count = 0

    def standardize_state(self, state):
        """
        Standardize state (d_md & h_mk)
        :param state:
        :return:
        """
        state = np.array(state)
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def choose_action(self, state, episode):
        """
        Choose action based on epsilon-greedy policy and Q network
        :param state: current state
        :param episode: current episode
        :return: [index of action for each MD]
        """
        # Epsilon decay
        # self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) / (1 + np.log(1 + episode))
        self.epsilon = max(self.min_epsilon, self.max_epsilon * (self.epsilon_decay_rate ** episode))

        d_md_remaining = state[:self.num_mds]

        # state = self.standardize_state(state)
        # Epsilon-greedy
        if np.random.uniform(0, 1) < self.epsilon:
            indices = np.random.choice(len(self.discrete_powers), self.num_mds)  # Explore
            return np.array([self.discrete_powers[i] if d_md_remaining[m] > 0 else 0.0 for m, i in enumerate(indices)])

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)  # shape: (1, num_MDs * |action space|)

        q_values = q_values.view(self.num_mds, len(self.discrete_powers))
        # print(q_values)
        best_action_indices = np.argmax(q_values.numpy(), axis=-1)
        actions = [self.discrete_powers[i] if d_md_remaining[m] > 0 else 0.0 for m, i in enumerate(best_action_indices)]  # Exploit
        # print(actions)
        return actions

    def update(self):
        """
        Update Q network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size=self.batch_size)

        # Convert to torch tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        q_values = self.q_network(states).view(self.batch_size, self.num_mds, len(self.discrete_powers))
        # q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        gathered_q_values = []
        for md_idx in range(self.num_mds):
            md_actions = actions[:, md_idx].unsqueeze(1)  # [batch_size, 1]
            md_q_values = q_values[:, md_idx, :].gather(1, md_actions)  # [batch_size, 1]
            gathered_q_values.append(md_q_values)

        q_values = torch.cat(gathered_q_values, dim=1)  # [batch_size, num_mds]

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).view(self.batch_size, self.num_mds, len(self.discrete_powers))
            best_next_q = torch.max(next_q_values, dim=-1)[0]
            target_q_values = rewards.unsqueeze(1) + self.gamma * best_next_q

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
