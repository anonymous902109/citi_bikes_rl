import numpy as np

def evaluate(env, policy, n_episodes=100):
    episode_rewards = []
    for ep_i in range(n_episodes):

        obs, _ = env.reset()
        done = False
        rew = 0.0
        while not done:
            action, _ = policy(obs)
            obs, rew, done, trunc, info = env.step(action)

            rew += -info['bike_shortage']

        episode_rewards.append(rew)

    return np.mean(episode_rewards)