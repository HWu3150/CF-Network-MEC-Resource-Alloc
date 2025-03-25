import numpy as np
from stable_baselines3 import DDPG, DQN
from DDPG.gym_mec_env_cont import MECEnvGymContinuous
from DQN.gym_mec_env import MECEnvGym
from mec_env import MECEnv
from env_config import *
import os

NUM_EPISODES = 30
EPISODE_LENGTH = 50


# ---------- Load Trained Models ----------
ddpg_model_path = os.path.join("DDPG", "models", "ddpg_mec_final")
dqn_model_path = os.path.join("DQN", "models", "dqn_mec_final")
ddpg_model = DDPG.load(ddpg_model_path)
dqn_model = DQN.load(dqn_model_path)

# ---------- Create Environments ----------
ddpg_env = MECEnvGymContinuous(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                               BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)
dqn_env = MECEnvGym(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                    BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)
baseline_env = MECEnv(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                      BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)

# ---------- Evaluation Function ----------
def evaluate_policy(model, env, deterministic=True):
    """
    Evaluate with model policy.
    :param model:
    :param env:
    :param deterministic:
    :return:
    """
    rewards = []
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(EPISODE_LENGTH):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards


def evaluate_random(env: MECEnv):
    """
    Evaluate with random power level allocations.
    :param env:
    :return:
    """
    rewards = []
    for ep in range(NUM_EPISODES):
        env.reset()
        total_reward = 0
        for _ in range(EPISODE_LENGTH):
            actions = np.random.uniform(0, max(DISCRETE_POWERS), size=NUM_MDS)
            _, reward, done = env.step(actions)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards


def evaluate_greedy(env: MECEnv):
    """
    Evaluate with greedy policy.
    :param env:
    :return:
    """
    rewards = []
    for ep in range(NUM_EPISODES):
        env.reset()
        total_reward = 0
        for _ in range(EPISODE_LENGTH):
            actions = np.zeros(NUM_MDS)
            max_idx = np.argmax(env.d_md_percent)
            actions[max_idx] = max(DISCRETE_POWERS)
            _, reward, done = env.step(actions)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards

# ---------- Run Evaluation ----------
print("Evaluating DDPG...")
ddpg_rewards = evaluate_policy(ddpg_model, ddpg_env)
print(f"DDPG: Mean = {np.mean(ddpg_rewards):.2f}, Std = {np.std(ddpg_rewards):.2f}")

print("Evaluating DQN...")
dqn_rewards = evaluate_policy(dqn_model, dqn_env)
print(f"DQN: Mean = {np.mean(dqn_rewards):.2f}, Std = {np.std(dqn_rewards):.2f}")

print("Evaluating Random baseline...")
random_rewards = evaluate_random(baseline_env)
print(f"Random: Mean = {np.mean(random_rewards):.2f}, Std = {np.std(random_rewards):.2f}")

print("Evaluating Greedy baseline...")
greedy_rewards = evaluate_greedy(baseline_env)
print(f"Greedy: Mean = {np.mean(greedy_rewards):.2f}, Std = {np.std(greedy_rewards):.2f}")
