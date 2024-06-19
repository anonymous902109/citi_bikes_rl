import gymnasium as gym
import maro
from gymnasium.spaces.multi_discrete import MultiDiscrete
from gymnasium.spaces import Box
from maro.simulator import Env
from maro.simulator.scenarios.citi_bike.common import Action, DecisionType
import numpy as np


class CitiBikes(gym.Env):

    def __init__(self):
        self.gym_env = Env(scenario="citi_bike", topology='toy.5s_6t', start_tick=0, durations=1440, snapshot_resolution=10)

        self.features = ["bikes",
                         "decision_type", "decision_station_idx", "frame_idx",
                         "holiday", "temperature", "weather", "weekday"]

        self.station_features = self.features[0:1]
        self.decision_features = self.features[1:4]
        self.shared_features = self.features[4:]

        self.num_stations = len(self.gym_env.current_frame.stations)
        self.max_bike_transfer = 10

        self.state_dim = self.num_stations * len(self.station_features) + len(self.decision_features) +  len(self.shared_features)

        self.action_space = MultiDiscrete([self.num_stations, self.num_stations, self.max_bike_transfer])
        self.observation_space = Box(low=np.array([0]*self.state_dim), high=np.array([np.inf]*self.state_dim), shape=(self.state_dim, ))

        self.max_penalty = sum([station.capacity for station in self.gym_env.current_frame.stations])

        self.penalties = {
            'IMPOSSIBLE_ACTION': -10,
            'ZERO ACTION': 0,
            'UNNECESSARY ACTION': -100
        }

        self.decision_types = {
            'NONE': 0,
            'SUPPLY': 1,
            'DEMAND': 2
        }

    def step(self, action):
        start_station, end_station, number = action
        action = Action(
            from_station_idx=start_station,
            to_station_idx=end_station,
            number=number
        )

        if self.obs[0] == self.decision_types['SUPPLY']:
            action = None

        metric, decision_event, is_done = self.gym_env.step(action)

        rew = self.calculate_reward(metric, self.obs, action)

        obs = self.generate_obs(decision_event)

        self.obs = obs

        return obs, rew, is_done, False, {'bike_shortage': metric['bike_shortage']}

    def calculate_reward(self, metric, obs, action):
        bike_shortage = metric['bike_shortage']
        rew = -bike_shortage

        # # check if action was possible if it wants to move more bikes than there are at the station
        # bikes_start = self.gym_env.current_frame.stations[action.from_station_idx].bikes
        # transfer = action.number
        #
        # # if agent wasn't supposed to act because transfering more than they can
        # if bikes_start < transfer:
        #     rew += self.penalties['IMPOSSIBLE_ACTION']
        #
        # if transfer == 0:
        #     rew += self.penalties['ZERO ACTION']

        return rew

    def reset(self, seed=0):
        self.gym_env.reset()
        self.gym_env.step(None)[0]
        obs = self.generate_obs(None)

        self.obs = obs

        return np.array(obs).squeeze(), None

    def generate_obs(self, decision_event):
        obs = []

        stations = self.gym_env.current_frame.stations

        if decision_event is None:
            obs.append(self.decision_types['NONE'])
        elif decision_event.type == DecisionType.Supply:
            obs.append(self.decision_types['SUPPLY'])
        elif decision_event.type == DecisionType.Demand:
            obs.append(self.decision_types['DEMAND'])

        if decision_event is not None:
            obs.append(decision_event.station_idx)
            obs.append(decision_event.frame_index)

            for station_id in range(self.num_stations):
                obs.append(decision_event.action_scope[station_id])

        else:
            obs.append(-1)
            obs.append(self.gym_env.frame_index)
            for s in stations:
                obs.append(-1)
                # obs.append(s.bikes)

        # adding features same for all stations such as weather
        for f in self.shared_features: # tick cannot be added this way
            obs.append(getattr(stations[0], f))

        return obs

    def render(self):
        print(self.gym_env.summary)

    def close(self):
        pass