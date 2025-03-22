import numpy as np
import torch
import torch.optim as optim
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer

import TrajectoryBuffer

# first_trajectory = trajectories[0]
# print(len(trajectories))
# print(f"First trajectory keys: {first_trajectory.keys()}")
#
# states = first_trajectory["states"]
# actions = first_trajectory["actions"]
# rewards = first_trajectory["rewards"]
#
# print(f"States shape: {states.shape}")
# print(f"Actions shape: {actions.shape}")
# print(f"Rewards shape: {rewards.shape}")
#
# print(first_trajectory)

states, actions, rewards, rtg, time_steps = TrajectoryBuffer.load("DQN/trajectories.npy")

states = torch.tensor(np.concatenate(states), dtype=torch.float32)
actions = torch.tensor(np.concatenate(actions), dtype=torch.float32)
rewards = torch.tensor(np.concatenate(rewards), dtype=torch.float32)
rtg = torch.tensor(np.concatenate(rtg), dtype=torch.float32)
time_steps = torch.tensor(np.concatenate(time_steps), dtype=torch.long)

# Hyperparameters
state_dim = states.shape[-1]
act_dim = actions.shape[-1]
hidden_size = 128
max_length = 20

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    hidden_size=hidden_size,
    max_length=max_length
)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

trainer = SequenceTrainer(
    model=model,
    optimizer=optimizer,
    batch_size=64,
    get_batch=lambda batch_size: (states[:batch_size], actions[:batch_size], rewards[:batch_size], rtg[:batch_size], timesteps[:batch_size], None),
    loss_fn=loss_fn
)

for epoch in range(10):
    trainer.train_iteration(num_steps=1000, iter_num=epoch, print_logs=True)
