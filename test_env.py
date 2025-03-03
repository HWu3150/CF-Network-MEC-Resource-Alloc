from mec_env import *
from env_config import *

powers = np.array([1.0, 1.2])
h_mk = np.array([[0.5, 0.7], [0.6, 0.8]])
G_mk = np.array([[2.0, 1.5], [1.8, 1.2]])
channel_noise = 0.1
sinrs = compute_sinr(powers, h_mk, G_mk, channel_noise)
print(sinrs)
print(compute_transmission_rates(sinrs, bandwidth=BANDWIDTH) / 8 / 1e6)

env = MECEnv(NUM_MDS,
             NUM_APS,
             DATA_SIZE,
             SMALL_SCALE_FADING,
             BANDWIDTH,
             CHANNEL_NOISE,
             G,
             DISCRETE_POWERS,
             t)

print(env._get_state())
env.step([2, 3])
print(env._get_state())
env.step([5, 6])
print(env._get_state())