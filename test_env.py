from mec_env import *
from env_config import *

powers = np.array([0.8, 0.1])
powers_low = np.array([0.8, 0.8])
h_mk = np.array([[0.5, 0.5], [0.5, 0.5]])
G_mk = np.array([[-90, -90], [-90, -90]])
channel_noise = 0.1
sinrs = compute_sinr(powers, h_mk, G_mk, channel_noise)
print("sinr by vec", sinrs)
# print("sinr by iter", compute_sinr(powers, h_mk, G_mk, channel_noise))
print("0.8 power trans rate: ", compute_transmission_rates(sinrs, bandwidth=BANDWIDTH))
print("0.1 power trans rate: ", compute_transmission_rates(compute_sinr(powers_low, h_mk, G_mk, channel_noise), bandwidth=BANDWIDTH))

env = MECEnv(NUM_MDS,
             NUM_APS,
             DATA_SIZE,
             SMALL_SCALE_FADING,
             BANDWIDTH,
             CHANNEL_NOISE,
             G,
             DISCRETE_POWERS,
             t)

# print(env.h_idx)
# print(env.h_mk)
# print(env.G_mk)
# print(env.d_md)
#
# print(env._get_state())
# print(env.step([2, 3])[1])
# print(env._get_state())
# print(env.step([5, 6])[1])
# print(env._get_state())

def test_fixed_power_2_md(power_values):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        d_md_remaining = env.d_md
        actions = power_values
        if d_md_remaining[0] == 0:
            actions[0] = 0.
            actions[1] = 0.8
        elif d_md_remaining[1] == 0:
            actions[1] = 0.
            actions[0] = 0.8

        state, reward, done = env.step(actions)
        total_reward += reward
        steps += 1

    print(f"Power: {power_values}, Steps: {steps}, Total Reward: {total_reward:.4f}")
    return steps, total_reward


# 测试 MD 使用最大功率 (0.8)
test_fixed_power_2_md(powers)

# 测试 MD 使用最小功率 (0.1)
test_fixed_power_2_md(powers_low)