import numpy as np


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

    def load(self, filename):
        """
        Read trajectories.
        """
        self.trajectories = np.load(filename, allow_pickle=True)
