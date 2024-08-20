from gym.wrappers import NormalizeReward
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from baselines import random_policy, greedy_policy, noop_policy
from citibikes import CitiBikes
from evaluate import evaluate


def print_hi():
    env = CitiBikes()

    n_env = NormalizeReward(env)
    n_env = NormalizeObservation(n_env)

    n_env = Monitor(n_env, "./tensorboard/", allow_early_resets=True)
    n_env = DummyVecEnv([lambda: n_env])

    baselines = [random_policy,  noop_policy]
    baseline_names = ['RANDOM', 'NOOP']

    # for i, policy in enumerate(baselines):
    #     mean_rew = evaluate(n_env, policy, n_episodes=100)
    #     print('{}: Mean reward = {}'.format(baseline_names[i], mean_rew))

    # model = PPO.load('ppo.zip')
    model = PPO('MlpPolicy', n_env, verbose=1,
                policy_kwargs={'net_arch': [256, 256]}, learning_rate=0.0001, batch_size=512, gamma=0.9)
    model.learn(total_timesteps=5e5)
    model.save('ppo_1e6.zip')

    evaluate(n_env, model.predict)

    n_ep = 10
    fros = {station: 0 for station in range(n_env.envs[0].num_stations)}
    tos = {station: 0 for station in range(n_env.envs[0].num_stations)}

    ep_steps = []
    ep_rewards = []
    for i in tqdm(range(n_ep)):
        done = False
        obs = n_env.reset()
        n_steps = 0.0
        rew = 0.0
        while not done:
            n_steps += 1
            action, _ = model.predict(obs)

            fro, to, n = action[0]

            fros[fro] += n
            tos[to] += n

            obs, reward, done, info = n_env.step(action)
            rew += reward

        ep_steps.append(n_steps)
        ep_rewards.append(rew)

    print('Episode length {}'.format(ep_steps))
    print('Episode reward: {}'.format(ep_rewards))

    print('FROM: {}'.format(fros))
    print('TO: {}'.format(tos))


if __name__ == '__main__':
    print_hi()

