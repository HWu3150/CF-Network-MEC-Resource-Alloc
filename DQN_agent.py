from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(self, env,
                 state_space_size,
                 action_space_size,
                 max_epsilon,
                 min_epsilon,
                 epsilon_decay_rate=0.001,
                 lr=0.001,
                 gamma=0.9):
        self.env = env
        self.num_mds, self.discrete_powers = env.get_env_params()

        self.memory = ReplayBuffer(capacity=10000)

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        # Epsilon-greedy params
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = max_epsilon
        # Q network
        self.model = build_model(state_space_size, action_space_size, self.num_mds)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state, episode):
        """
        Choose action based on epsilon-greedy policy and Q network
        state: current state
        episode: current episode
        return: [index of action for each MD]
        """
        # Epsilon decay
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -self.epsilon_decay_rate * episode)
        # Epsilon-greedy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(len(self.discrete_powers), self.num_mds)  # Explore

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)  # shape: (1, num_MDs * |action space|)

        q_values = q_values.view(self.num_mds, len(self.discrete_powers))
        return np.argmax(q_values.numpy(), axis=-1)  # Exploit

    def update(self, state, actions, reward, next_state):
        """
        Update Q network
        state: current state
        actions: action taken by each MD
        reward: reward received
        next_state: next state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])

        q_values = self.model(state_tensor).view(self.num_mds, len(self.discrete_powers))
        next_q_values = self.model(next_state_tensor).detach().view(self.num_mds, len(self.discrete_powers))

        target_q_values = q_values.clone()
        # Compute Q*(s_t+1, a)
        best_next_q = torch.max(next_q_values, axis=-1)[0]
        # Compute r_t + gamma * Q*(s_t+1, a)
        for i in range(self.num_mds):
            target_q_values[i, actions[i]] = reward + self.gamma * best_next_q[i]

        # Update Q network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
