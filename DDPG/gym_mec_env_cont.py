from typing import Optional, Dict, Any

from gymnasium import Env, spaces
import numpy as np

from mec_env import MECEnv


class MECEnvGymContinuous(Env):
    def __init__(self, num_mds, num_aps, data_size, small_scale_fading,
                 bandwidth, channel_noise, g, discrete_powers, t_length):
        super(MECEnvGymContinuous, self).__init__()

        # Initialize MEC env
        self.env = MECEnv(num_mds, num_aps, data_size, small_scale_fading,
                          bandwidth, channel_noise, g, discrete_powers, t_length)

        # Get max power level
        self.max_power = max(discrete_powers)

        # Define continuous action space (one value per MD)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,  # Will be scaled to actual power range
            shape=(num_mds,),
            dtype=np.float32
        )

        # Save parameters
        self.num_mds = num_mds
        self.discrete_powers = discrete_powers

        # Define state space
        obs_dim = len(self.env._get_state())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs = np.array(self.env.reset(), dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Scale actions from [0,1] to [0,max_power]
        scaled_action = action * self.max_power

        # Apply zero power to MDs with no data
        for i in range(self.num_mds):
            if self.env.d_md_percent[i] <= 0:
                scaled_action[i] = 0.0

        # Take step in environment with scaled actions
        next_state, reward, done = self.env.step(scaled_action)

        return np.array(next_state, dtype=np.float32), reward, done, False, {}