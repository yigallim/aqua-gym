import numpy as np
import math
from utils.config import Config

class TemperatureModel:

    def __init__(
        self,
        region: str = "guangdong",
        season_period: int = 365,
        alpha: float = 0.25,
        beta: float = 0.05
    ):
        Config.load()
        if not hasattr(Config.temp_model, region):
            raise ValueError(f"unknown site '{region}' in temp_model section")
        site = getattr(Config.temp_model, region)

        self.T_mean = site.T_mean
        self.T_amp = site.T_amp
        self.phase_shift = site.phase_shift
        self.season_period = season_period

        self.Tmin = Config.ind_growth_model.T_min
        self.Tmax = Config.ind_growth_model.T_max
        self.alpha = alpha
        self.beta  = beta

        self.day_of_year = 1
        self.current_T   = self.T_mean   

    def set_day_of_year(self, day):
        self.day_of_year = day

    def get_ambient_temperature(self):
        ambient = (
            self.T_mean
            + self.T_amp * math.sin(2 * math.pi * (self.day_of_year - self.phase_shift) / self.season_period)
            + np.random.normal(0, 1)
        )
        return ambient

    # https://ocw.mit.edu/courses/10-450-process-dynamics-operations-and-control-spring-2006/dc573f23401eeb5822818fbaa177eaac_5_heated_tank.pdf
    def set_temperature(self, set_temperature):
        T_set = np.clip(set_temperature, self.Tmin, self.Tmax)
        T_amb = self.get_ambient_temperature()

        heater_on = self.current_T < T_set
        alpha_eff = self.alpha if heater_on else 0.0

        T_next = (
            self.current_T
            + alpha_eff * (T_set - self.current_T)
            + self.beta * (T_amb - self.current_T)
        )

        T_next = np.clip(T_next, self.Tmin, self.Tmax)
        self.current_T = T_next
        self.day_of_year = 1 + (self.day_of_year % self.season_period)
        return T_next