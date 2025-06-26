import numpy as np

class Calculation:

    @staticmethod
    def compute_fcr(feed_weight, final_weight, initial_weight):
        # Calculates the Feed Conversion Ratio (FCR) based on feed used and weight gain, in KG
        gain = final_weight - initial_weight
        if gain <= 0:
            return None
        return feed_weight / gain

    @staticmethod
    def compute_feed_weight(feed_rate, weight):
        # Estimates the amount of feed given based on feed rate and fish biomass, in KG
        return feed_rate * weight * 0.1

    @staticmethod
    def compute_sgr(initial_weight, final_weight, days=180):
        # Computes the Specific Growth Rate (SGR) as a percentage increase per day.
        if initial_weight <= 0 or final_weight <= 0:
            return None
        return (np.log(final_weight) - np.log(initial_weight)) / days * 100

    @staticmethod
    def compute_profit_margin(fish_value_list, cost_list):
        # Calculates the overall profit margin (%) based on fish value and associated costs.
        if len(fish_value_list) != len(cost_list):
            return None
        total_profit = sum(fv - c for fv, c in zip(fish_value_list, cost_list))
        total_revenue = sum(fish_value_list)
        if total_revenue == 0:
            return None
        return (total_profit / total_revenue) * 100

    @staticmethod
    def compute_energy_efficiency(fish_value_gain, heat_cost, oxygenation_cost):
        # Computes the Energy Efficiency based on Fish Value Gain and Electricity Cost (Heat + Oxygenation costs)
        electricity_cost = heat_cost + oxygenation_cost
        if electricity_cost == 0:
            return None  # Avoid division by zero
        return fish_value_gain / electricity_cost