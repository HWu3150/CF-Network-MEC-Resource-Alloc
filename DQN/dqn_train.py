from DQN.dqn_agent import DQNAgent
from env_config import *
from mec_env import MECEnv
from TrajectoryBuffer import TrajectoryBuffer

# Training params
episodes = 10000
max_epsilon = 1.0
min_epsilon = 0.05
epsilon_decay_rate = 0.9998

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
                 min_epsilon=min_epsilon,
                 epsilon_decay_rate=epsilon_decay_rate)

trajectory_buffer = TrajectoryBuffer()

# Training
for episode in range(episodes):
    states, actions, rewards = [], [], []
    # Reset environment
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Learn
    while not done:
        action = agent.choose_action(state, episode)
        next_state, reward, done = env.step(action)

        agent.replay_buffer.push(state, action, reward, next_state)
        agent.update()

        # print(f"Step {step}: Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        total_reward += reward
        step += 1

    trajectory_buffer.add_trajectory(states, actions, rewards)

    if episode % 50 == 0:
        print(f"Episode: {episode}, Total reward: {total_reward:.4f}, epsilon: {agent.epsilon}")

trajectory_buffer.save("trajectories")
