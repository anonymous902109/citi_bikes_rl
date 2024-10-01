import numpy as np


def evaluate(env, policy, n_episodes=10):
    episode_rewards = []
    for ep_i in range(n_episodes):

        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = policy(obs)
            obs, rew, done, trunc, info = env.step(action)

            ep_rew += rew

        episode_rewards.append(ep_rew)

    return np.mean(episode_rewards)