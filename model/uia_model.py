from utils.config import Config
import numpy as np

class UIAModel:
    def __init__(self, region: str = "guangdong"):
        Config.load()

        try:
            self.params = getattr(Config.reward_cost_parameters, region)
        except AttributeError:
            raise ValueError(f"Unknown region '{region}' in reward_cost_parameters")

        region_params = getattr(Config.reward_cost_parameters, region)
        common_params = Config.reward_cost_parameters.common

        self.pH = 7
        self.oxygen_level = 1
        self.UIA = 0.06
        self.decay_rate = 0.8
        self.tank_volume = common_params.V

    def get_uia(self, feed_g, temperature):
        protein_fraction = 0.30
        N_fraction_in_protein = 0.16
        N_excreted_as_ammonia = 0.90
        g_to_mg = 1000

        log_feed = feed_g / (1 + Config.ind_growth_model.UIA_slowdown * feed_g)
        ammonia_nitrogen_mg = (
            log_feed * protein_fraction * N_fraction_in_protein * N_excreted_as_ammonia * g_to_mg
        )

        TAN = ammonia_nitrogen_mg / self.tank_volume

        pKa = 0.09018 + (2729.92 / (temperature + 273.15))
        UIA_fraction = 1 / (1 + 10 ** (pKa - self.pH))
        UIA_produced = (TAN * UIA_fraction)
        self.UIA = self.UIA * (1 - self.decay_rate) + UIA_produced
        return np.clip(self.UIA, 0.06, 1.8)