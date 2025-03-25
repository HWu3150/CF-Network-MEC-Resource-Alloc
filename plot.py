import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardCallback(BaseCallback):
    """
    Callback for saving episode rewards during training
    """

    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.cumulative_reward = 0
        self.episode_timesteps = 0

    def _on_step(self) -> bool:
        # Update counters
        self.episode_timesteps += 1

        # Get the most recent reward
        reward = self.locals['rewards'][0]  # For single environment
        self.cumulative_reward += reward

        # If episode is done, save the total reward and reset counters
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_lengths.append(self.episode_timesteps)
            self.episode_count += 1

            # Print info
            if self.verbose > 0 and self.episode_count % 10 == 0:
                print(f"Episode {self.episode_count}: reward = {self.cumulative_reward}")

            # Reset counters
            self.cumulative_reward = 0
            self.episode_timesteps = 0

        return True

    def plot_rewards(self, save_path=None, window=10):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards, label='Episode reward')
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Episode Rewards during Training")

        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(self.episode_rewards)),
                     moving_avg,
                     color='red',
                     label=f'{window}-episode moving average')
            plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()
