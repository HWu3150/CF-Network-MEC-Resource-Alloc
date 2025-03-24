from typing import Optional, Dict, Any

from gymnasium import Env, spaces
import numpy as np

from mec_env import MECEnv


# Encapsulate MEC environment as gym env
class MECEnvGym(Env):
    def __init__(self, num_mds, num_aps, data_size, small_scale_fading,
                 bandwidth, channel_noise, g, discrete_powers, t_length):
        super(MECEnvGym, self).__init__()

        # Initialize MEC env
        self.env = MECEnv(num_mds, num_aps, data_size, small_scale_fading,
                          bandwidth, channel_noise, g, discrete_powers, t_length)

        # Define action space
        self.action_space = spaces.Discrete(len(discrete_powers) ** num_mds)

        self.num_mds = num_mds
        self.discrete_powers = discrete_powers

        # Define state space
        obs_dim = len(self.env._get_state())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs = np.array(self.env.reset(), dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Convert action space # to array of actions
        powers = []
        remaining_action = action

        for i in range(self.num_mds):
            # Extract action index of each MD
            power_idx = remaining_action % len(self.discrete_powers)
            remaining_action = remaining_action // len(self.discrete_powers)

            powers.append(self.discrete_powers[power_idx])

        next_state, reward, done = self.env.step(powers)
        return np.array(next_state, dtype=np.float32), reward, done, False, {}  # Truncated & info(debug)
