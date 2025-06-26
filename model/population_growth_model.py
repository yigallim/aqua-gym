# extract saturation logic from individual, apply for population
# implement mortality

import math

class PopulationGrowthModel:
    def __init__(
        self,
        initial_population: int,
        stocking_rate: int = 0,
        individual_biomass: float = 0.0,
        mean_biomass: float = 0.0
    ):
        self.p = initial_population
        self.p_s = stocking_rate
        self.xi_i = individual_biomass
        self.xi_a = mean_biomass

        # Mortality‐fit parameters (from logistic regression on UIA)
        self._Delta = 99.41
        self._beta  = 10.36
        self._eta   = 0.80

    def mortality_coeff(self, UIA: float) -> float:
        """
        Compute the mortality coefficient k1(UIA) via
        logistic regression:
            k1 = Delta / (1 + exp[-beta * (UIA - eta)])
        :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
        """
        exponent = -self._beta * (UIA - self._eta)
        return self._Delta / (1 + math.exp(exponent))

    def apply_mortality(self, UIA: float) -> int:
        """
        Remove fish according to p * k1(UIA), rounded down.
        Returns the number of deaths this step.
        :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
        """
        k1 = self.mortality_coeff(UIA)
        deaths = int(self.p * k1)
        self.p = max(0, self.p - deaths)
        return deaths

    def step(self, UIA: float) -> dict:
        """
        Advance one time unit:
          1) Add stocked fish
          2) Apply mortality based on current UIA
        Returns a summary dict.
        """
        # Stock new fish
        self.p += self.p_s

        # Apply mortality
        deaths = self.apply_mortality(UIA)

        return {
            'population': self.p,
            'deaths': deaths
        }
