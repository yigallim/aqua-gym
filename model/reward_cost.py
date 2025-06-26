from utils.config import Config

class RewardCost:
    def __init__(self, region: str = "guangdong"):
        Config.load()

        if not hasattr(Config.reward_cost_parameters, region):
            raise ValueError(f"Unknown region '{region}' in reward_cost_parameters")

        region_params = getattr(Config.reward_cost_parameters, region)
        common_params = Config.reward_cost_parameters.common

        self.P_s = region_params.P_s     # Selling price
        self.P_f = region_params.P_f     # Feed price
        self.P_e = region_params.P_e     # Electricity price

        self.c_p = common_params.c_p     # Specific heat
        self.V = common_params.V         # Tank volume
        self.m = common_params.m         # Water mass
        self.P_max = common_params.P_max # Max power

    def fish_value_gain(self, biomass_prev, biomass_curr, alpha=1.0): # biomass in kg
        """
        α · [(ξ_k+1 - ξ_k) · P_s] = fish value gain
        """
        biomass_gain = biomass_curr - biomass_prev
        return alpha * biomass_gain * self.P_s

    def feed_cost(self, feed_weight, beta=1.0): # feed_weight in kg
        """
        β · (P_f · R · u1), u1 = feeding rate, R = feed ration
        """
        return beta * self.P_f * feed_weight

    def heat_cost(self, delta_T):
        """
        (P_e · c_p · V · m · ΔT) / 3600
        """
        return (self.P_e * self.c_p * self.V * self.m * delta_T) / 3600

    def oxygenation_cost(self, DO_level):
        """
        24 · P_e · P_max · u3, u3 = DO_level
        """
        return 24 * self.P_e * self.P_max * DO_level
