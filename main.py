from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines import random_policy, greedy_policy
from citibikes import CitiBikes
from evaluate import evaluate


def print_hi():
    env = CitiBikes()

    env = Monitor(env, "./tensorboard/", allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    baselines = [random_policy, greedy_policy]
    baseline_names = ['RANDOM', 'GREEDY']

    for i, policy in enumerate(baselines):
        mean_rew = evaluate(env.envs[0], policy, n_episodes=100)
        print('{}: Mean reward = {}'.format(baseline_names[i], mean_rew))

    policy_kwargs = {'net_arch': [1028, 1028, 1028, 1028]}
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                batch_size=512,
                learning_rate=0.001,
                gamma=0.9,
                verbose=1,
                tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=1000000, tb_log_name="run")
    model.save("trained_models/citibikes")

    mean_rew = evaluate(env.envs[0], model.predict)
    print('PPO: Mean reward = {}'.format(mean_rew))


if __name__ == '__main__':
    print_hi()

