import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from envs.renderer import Renderer
from model.fish import Fish, FishStage
from model.individual_growth_model import IndividualGrowthModel
from model.uia_model import UIAModel
from model.temperature_model import TemperatureModel
from model.reward_cost import RewardCost

class AquacultureEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    ALLOWED_REGIONS = ["guangdong", "north_sulawesi", "kafr_el_sheikh"]

    def __init__(self, region="guangdong"):
        if region not in self.ALLOWED_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Allowed regions: {self.ALLOWED_REGIONS}")
        super().__init__()

        self.region = region

        # State space (observation) boundaries:
        # State variables include:
        # [0] Total Biomass (ξ): range 0.05 kg - 30000 kg → scaled here to [50, 3e7] grams
        # [1] Fish Count (p): range 50 - 500 units
        # [2] Temperature (T): range 24°C - 40°C
        # [3] Dissolved Oxygen (DO): range 0.3 mg/L - 1 mg/L
        # [4] Un-ionized Ammonia (UIA): range 0.06 mg/L - 1.8 mg/L
        self.obs_low  = np.array([50, 50, 24, 0.3, 0.06], dtype=np.float32)
        self.obs_high = np.array([3e7, 500, 40, 1.0, 1.8 ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low = np.zeros_like(self.obs_low),
            high= np.ones_like(self.obs_high),
           dtype=np.float32
        )

        # Action space (continuous control variables):
        # [0] Feeding Rate (f): daily feed ratio [0, 1] relative to fish body weight
        # [1] Temperature Control (T_set): desired water temperature [24°C - 40°C]
        # [2] Aeration Rate (DO_set): target dissolved oxygen level [0.3 mg/L - 1 mg/L]
        self.action_space = spaces.Box(
            low = np.array([0.0, 24.0, 0.3], dtype=np.float32),
            high= np.array([1.0, 40.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.initial_fish_count = 100
        self.max_days = 180
        self.day = 0
        self.temperature = 33.0
        self.dissolved_oxygen = 0.6
        self.un_ionized_ammonia = 0.06
        self.feed_today = 0.0
        self.feed_yesterday = 0.0
        self.feed_rate_today = 0.0
        self.feed_rate_yesterday = 0.0
        self.initial_fish_count = 100
        self.max_days = 180

        self.growth_model = IndividualGrowthModel()
        self.temperature_model = TemperatureModel(region=region)
        self.uia_model = UIAModel(region=region)
        self.reward_model = RewardCost(region=region)
        self._initialize_population()
        self.prev_biomass = self._compute_total_biomass()

        self.renderer = Renderer(self)
        
    def _initialize_population(self):
        self.fishes = [
            Fish.generate_random(self.growth_model)
            for _ in range(self.initial_fish_count)
        ]

    def _compute_total_biomass(self):
        return sum(f.weight for f in self.fishes)

    def _compute_fish_count(self):
        return len(self.fishes)

    def _get_observation(self, biomass, fish_count, temp):
        raw = np.array([
            biomass,
            fish_count,
            temp,
            self.dissolved_oxygen,
            self.un_ionized_ammonia
        ], dtype=np.float32)
        norm = (raw - self.obs_low) / (self.obs_high - self.obs_low)
        return np.clip(norm, 0.0, 1.0)
    
    def denormalize(self, obs_norm: np.ndarray) -> np.ndarray:
        return obs_norm * (self.obs_high - self.obs_low) + self.obs_low

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        feed_rate, temp_setpoint, aeration_rate = action

        self.dissolved_oxygen = float(aeration_rate)
        self.temperature_model.set_day_of_year(self.day)
        ambient_temp = self.temperature_model.get_ambient_temperature()
        temp_heated  = max(temp_setpoint - ambient_temp, 0.0)
        self.temperature = self.temperature_model.set_temperature(temp_setpoint)

        for fish in self.fishes:
            fish.grow(feed_rate, self.temperature, self.dissolved_oxygen, self.un_ionized_ammonia)

        biomass = self._compute_total_biomass()
        fish_count = self._compute_fish_count()
        biomass_gain = biomass - self.prev_biomass

        feed_amount_total = feed_rate * 0.1 * biomass
        self.feed_yesterday = self.feed_today
        self.feed_today = feed_amount_total
        self.feed_rate_yesterday = self.feed_rate_today
        self.feed_rate_today = feed_rate
        self.un_ionized_ammonia = self.uia_model.get_uia(feed_amount_total, self.temperature)

        fish_value = self.reward_model.fish_value_gain(self.prev_biomass / 1000, biomass / 1000) * 2
        feed_cost = self.reward_model.feed_cost(feed_amount_total / 1000) * 0.9
        heat_cost = self.reward_model.heat_cost(delta_T=temp_heated) * 0.75
        oxy_cost = self.reward_model.oxygenation_cost(DO_level=self.dissolved_oxygen) * 0.75

        reward = fish_value - feed_cost - heat_cost - oxy_cost

        self.prev_biomass = biomass
        self.day += 1

        obs = self._get_observation(biomass, fish_count, self.temperature)
        terminated = bool(self.day >= self.max_days or biomass <= 100)
        truncated = False
        info = {
            "biomass_gain": biomass_gain,
            "uia": self.un_ionized_ammonia,
            "reward": reward,
            "feed_rate": feed_rate,
            "temperature": self.temperature,
            "dissolved_oxygen": self.dissolved_oxygen,
            "fish_value": fish_value,
            "feed_cost": feed_cost,
            "heat_cost": heat_cost,
            "oxygenation_cost": oxy_cost
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.day = 0
        self._initialize_population()
        self.prev_biomass = self._compute_total_biomass()

        self.uia_model = UIAModel(region=self.region)
        self.temperature_model = TemperatureModel(region=self.region)
        ambient_temp = self.temperature_model.get_ambient_temperature()
        self.temperature = self.temperature_model.set_temperature(ambient_temp)
        
        self.dissolved_oxygen = 0.6
        self.un_ionized_ammonia = 0.06
        self.feed_today = 0.0
        self.feed_yesterday = 0.0
        self.feed_rate_today = 0.0
        self.feed_rate_yesterday = 0.0
        self.uia_model.temperature = self.temperature

        obs = self._get_observation(self.prev_biomass, self._compute_fish_count(), self.temperature)
        return obs, {}

    def render(self, mode='human'):
        if mode not in self.metadata['render_modes']:
            raise ValueError(f"Unsupported render mode: {mode}")
        self.renderer.render()

    def close(self):
        if hasattr(self, "renderer"):
            try:
                self.renderer.close()
            except Exception as e:
                print(f"[Warning] renderer.close() threw: {e}")
        super().close()
