from ddpg_agent import DDPGAgent
from env_config import *
from mec_env import MECEnv

episodes = 1000
batch_size = 32

env = MECEnv(NUM_MDS,
             NUM_APS,
             DATA_SIZE,
             SMALL_SCALE_FADING,
             BANDWIDTH,
             CHANNEL_NOISE,
             G,
             DISCRETE_POWERS,
             t)

state_dim = len(env._get_state())
action_dim = NUM_MDS
max_action = max(DISCRETE_POWERS)

agent = DDPGAgent(env, state_dim, action_dim, max_action)
print("Hello")

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update(batch_size)

        state = next_state
        total_reward += reward
        step += 1

    # if episode % 50 == 0:
    #     print(f"Episode {episode}, Total reward: {total_reward:.4f}, steps: {step}")
    print(f"Episode {episode}, Total reward: {total_reward:.4f}, steps: {step}")
