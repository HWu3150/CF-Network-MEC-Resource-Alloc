import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gym_mec_env_cont import MECEnvGymContinuous
from env_config import *
from DQN.plot import EpisodeRewardCallback

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("models/", exist_ok=True)

# Create environment
env = MECEnvGymContinuous(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                         BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)

# # Add Monitor wrapper for logging
# env = Monitor(env, log_dir)
#
# # Create evaluation environment
# eval_env = MECEnvGymContinuous(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
#                               BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)

# Action noise for exploration
n_actions = env.action_space.shape[0]

# Two options for noise - choose one:
# Option 1: Normal (Gaussian) noise
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)  # Start with higher exploration
)

# Option 2: Ornstein-Uhlenbeck process (often used in DDPG)
# action_noise = OrnsteinUhlenbeckActionNoise(
#     mean=np.zeros(n_actions),
#     sigma=0.2 * np.ones(n_actions),
#     theta=0.15
# )

# Create DDPG model
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=3e-4,       # Slightly lower learning rate for stability
    buffer_size=100000,       # Larger buffer for more diverse experiences
    learning_starts=10000,    # Collect more transitions before learning
    batch_size=256,           # Larger batch size for better gradient estimates
    gamma=0.99,               # Discount factor
    tau=0.005,                # Soft update coefficient
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],    # Actor network
            qf=[256, 256]     # Critic network
        )
    ),
    verbose=1
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="models/",
    name_prefix="ddpg_mec"
)
episode_callback = EpisodeRewardCallback(verbose=1)

# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path="models/best/",
#     log_path="logs/eval/",
#     eval_freq=5000,
#     n_eval_episodes=10,
#     deterministic=True,
#     render=False
# )

# Train the model
model.learn(
    total_timesteps=200000,
    callback=[checkpoint_callback, episode_callback],
    log_interval=100
)

# Save the final model
model.save("models/ddpg_mec_final")

episode_callback.plot_rewards(save_path="training_rewards.png")

# Evaluate the model
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
#
# # Plot learning curve
# from stable_baselines3.common.results_plotter import load_results, ts2xy
#
# # Plot learning curve
# x, y = ts2xy(load_results(log_dir), 'timesteps')
# plt.figure(figsize=(12, 6))
# plt.plot(x, y)
# plt.xlabel('Timesteps')
# plt.ylabel('Rewards')
# plt.title('DDPG Learning Curve')
# plt.savefig('ddpg_learning_curve.png')
# plt.show()