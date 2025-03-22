import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import DQN
from gym_mec_env import MECEnvGym
from env_config import *


def evaluate_model(model_path, env, num_episodes=100):
    """Evaluate model performance"""
    model = DQN.load(model_path)

    rewards = []
    steps = []
    completion_times = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

        while not done:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, done, _, info = env.step(action)

            total_reward += reward
            episode_steps += 1

        rewards.append(total_reward)
        steps.append(episode_steps)

        # 如果环境提供了完成时间指标
        if hasattr(env.unwrapped, 'time_step'):
            completion_times.append(env.unwrapped.time_step)

    return {
        'rewards': rewards,
        'steps': steps,
        'completion_times': completion_times
    }


def compare_with_random(env, trained_model_path, num_episodes=50):
    """比较训练模型与随机策略"""
    # 创建环境
    env_trained = MECEnvGym(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                            BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)
    env_random = MECEnvGym(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                           BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)

    # 加载训练好的模型
    model = DQN.load(trained_model_path)

    trained_rewards = []
    random_rewards = []

    # 设置相同的随机种子
    seed = 42

    for ep in tqdm(range(num_episodes), desc="Comparing"):
        # 评估训练模型
        obs, _ = env_trained.reset(seed=seed + ep)
        done = False
        trained_total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env_trained.step(action)
            trained_total_reward += reward

        trained_rewards.append(trained_total_reward)

        # 评估随机策略
        obs, _ = env_random.reset(seed=seed + ep)
        done = False
        random_total_reward = 0

        while not done:
            action = env_random.action_space.sample()  # 随机动作
            obs, reward, done, _, _ = env_random.step(action)
            random_total_reward += reward

        random_rewards.append(random_total_reward)

    return trained_rewards, random_rewards


def plot_results(results, title="Model Performance"):
    """绘制评估结果"""
    plt.figure(figsize=(15, 10))

    # 绘制奖励图
    plt.subplot(2, 2, 1)
    plt.plot(results['rewards'])
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # 绘制直方图
    plt.subplot(2, 2, 2)
    plt.hist(results['rewards'], bins=20)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')

    # 绘制步数图
    plt.subplot(2, 2, 3)
    plt.plot(results['steps'])
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    # 添加统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats = f"Average Reward: {np.mean(results['rewards']):.2f}\n"
    stats += f"Std Dev Reward: {np.std(results['rewards']):.2f}\n"
    stats += f"Min Reward: {np.min(results['rewards']):.2f}\n"
    stats += f"Max Reward: {np.max(results['rewards']):.2f}\n"
    stats += f"Average Steps: {np.mean(results['steps']):.2f}\n"
    if 'completion_times' in results:
        stats += f"Average Completion Time: {np.mean(results['completion_times']):.2f}"

    plt.text(0.1, 0.5, stats, fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def plot_comparison(trained_rewards, random_rewards):
    """绘制训练模型与随机策略的比较"""
    plt.figure(figsize=(12, 6))

    # 绘制奖励比较
    plt.subplot(1, 2, 1)
    plt.plot(trained_rewards, label='Trained Model')
    plt.plot(random_rewards, label='Random Policy')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    # 绘制箱线图比较
    plt.subplot(1, 2, 2)
    plt.boxplot([trained_rewards, random_rewards], labels=['Trained Model', 'Random Policy'])
    plt.title('Reward Distribution')
    plt.ylabel('Total Reward')

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()


if __name__ == "__main__":
    env = MECEnvGym(NUM_MDS, NUM_APS, DATA_SIZE, SMALL_SCALE_FADING,
                    BANDWIDTH, CHANNEL_NOISE, G, DISCRETE_POWERS, t)

    results = evaluate_model("dqn_mec_final", env, num_episodes=1000)
    plot_results(results, title="DQN Model Performance")

    trained_rewards, random_rewards = compare_with_random(env, "dqn_mec_final")
    plot_comparison(trained_rewards, random_rewards)
