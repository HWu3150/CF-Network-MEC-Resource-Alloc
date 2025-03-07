from DQN.dqn_agent import DQNAgent
from env_config import *
from mec_env import MECEnv

# Training params
episodes = 10000
max_epsilon = 1.0
min_epsilon = 0.05
epsilon_decay_rate = 0.0005

env = MECEnv(NUM_MDS,
             NUM_APS,
             DATA_SIZE,
             SMALL_SCALE_FADING,
             BANDWIDTH,
             CHANNEL_NOISE,
             G,
             DISCRETE_POWERS,
             t)

agent = DQNAgent(env,
                 state_dim=len(env._get_state()),
                 action_space_size=len(DISCRETE_POWERS),
                 max_epsilon=max_epsilon,
                 min_epsilon=min_epsilon)

# Training
for episode in range(episodes):
    # Reset environment
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Learn
    while not done:
        action = agent.choose_action(state, episode)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)

        # print(f"Step {step}: Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

        state = next_state
        total_reward += reward
        step += 1

    if episode % 50 == 0:
        print(f"Episode: {episode}, Total reward: {total_reward:.4f}, steps taken: {step}")
