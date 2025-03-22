import numpy as np


def load(filepath):
    """
    Load trajectories.
    """
    trajectories = np.load(filepath, allow_pickle=True)

    all_states, all_actions, all_rewards, all_rtg, all_time_steps = [], [], [], [], []

    for trajectory in trajectories:
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]

        # Compute RTG
        rtg = np.zeros_like(rewards, dtype=np.float32)
        total_return = 0
        for t in reversed(range(len(rewards))):
            total_return += rewards[t]
            rtg[t] = total_return

        # Time step index
        time_steps = np.arange(len(states))

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_rtg.append(rtg)
        all_time_steps.append(time_steps)

    return all_states, all_actions, all_rewards, all_rtg, all_time_steps


class TrajectoryBuffer:
    def __init__(self):
        self.trajectories = []

    def add_trajectory(self, states, actions, rewards):
        """
        Append a trajectory.
        """
        rtg = self.compute_rtg(rewards)     # Compute RTG
        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "rtg": np.array(rtg)
        }
        self.trajectories.append(trajectory)

    def compute_rtg(self, rewards):
        """
        Compute Return-to-Go (RTG), total reward starting at the
        current time step.
        """
        rtg = np.zeros_like(rewards, dtype=np.float32)
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total += rewards[t]
            rtg[t] = running_total
        return rtg

    def save(self, filename):
        """
        Save trajectories to file.
        """
        np.save(filename, self.trajectories)
