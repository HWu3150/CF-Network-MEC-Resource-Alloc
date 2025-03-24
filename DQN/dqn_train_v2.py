from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from plot import EpisodeRewardCallback
from gym_mec_env import MECEnvGym
from env_config import *

env = MECEnvGym(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1
)

episode_callback = EpisodeRewardCallback(verbose=1)
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./models/')

callbacks = [episode_callback, checkpoint_callback]

# Train
model.learn(total_timesteps=200000, callback=callbacks)

model.save("dqn_mec_final")

# Plot and save the reward curve
episode_callback.plot_rewards(save_path="training_rewards.png")
