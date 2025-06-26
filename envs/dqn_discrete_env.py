from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np
from envs.aquaculture_env import AquacultureEnv

class DiscretizedAquacultureEnv(Env):
    def __init__(self, region="guangdong"):
        self.base_env = AquacultureEnv(region=region)
        
        self.feed_bins = 40
        self.temp_bins = 16
        self.air_bins = 10

        self.discrete_actions = [
            (feed, temp, air)
            for feed in np.linspace(self.base_env.action_space.low[0], self.base_env.action_space.high[0], self.feed_bins)
            for temp in np.linspace(self.base_env.action_space.low[1], self.base_env.action_space.high[1], self.temp_bins)
            for air in np.linspace(self.base_env.action_space.low[2], self.base_env.action_space.high[2], self.air_bins)
        ]
        
        self.action_space = Discrete(len(self.discrete_actions))
        self.observation_space = self.base_env.observation_space

    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        return obs, info

    def step(self, action_idx):
        action = np.array(self.discrete_actions[action_idx], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.base_env.render()

    def close(self):
        self.base_env.close()

    def __getattr__(self, name):
        return getattr(self.base_env, name)