import numpy as np
import random

from model.individual_growth_model import IndividualGrowthModel

class FishStage:
    FINGERLING = "fingerling"
    JUVENILE = "juvenile"
    ADULT = "adult"

class Fish:
    def __init__(self, weight: float, growth_model: IndividualGrowthModel):
        if not growth_model:
            raise ValueError("growth_model must be provided")

        self.weight = max(weight if weight is not None else round(np.random.normal(5.25, 0.5), 2), 5)
        self.age_days = 0
        self.growth_model = growth_model

        self.to_juvenile_weight = max(np.random.normal(15, 3), 5)        # mean 15g ±3, min 5g
        self.to_juvenile_days = max(int(np.random.normal(30, 10)), 15)   # mean 30d ±10, min 15d
        self.to_adult_weight = max(np.random.normal(250, 30), 180)       # mean 250g ±30, min 180g
        self.to_adult_days = max(int(np.random.normal(180, 15)), 150)    # mean 180d ±15, min 150d

        self.to_juvenile_weight, self.to_adult_weight = sorted([self.to_juvenile_weight, self.to_adult_weight])
        self.to_juvenile_days, self.to_adult_days = map(int, sorted([self.to_juvenile_days, self.to_adult_days]))

    @staticmethod
    def generate_random(growth_model: IndividualGrowthModel):
        chosen_stage = np.random.choice([FishStage.FINGERLING, FishStage.JUVENILE])

        if chosen_stage == FishStage.FINGERLING:
            weight = round(np.random.normal(5.25, 0.5), 2)
        else:
            weight = round(np.random.normal(20, 4), 2)

        return Fish(weight=max(weight, 5), growth_model=growth_model)
        
    @property
    def stage(self) -> str:
        if self.weight >= self.to_adult_weight or self.age_days >= self.to_adult_days:
            return FishStage.ADULT
        elif self.weight >= self.to_juvenile_weight or self.age_days >= self.to_juvenile_days:
            return FishStage.JUVENILE
        else:
            return FishStage.FINGERLING


    def grow(self, feeding_rate: float, temperature: float, dissolved_oxygen: float, uia: float):
        growth = self.growth_model.compute_growth(
            feeding_rate, temperature, dissolved_oxygen, uia, self.weight
        )
        self.weight += growth

        if growth >= 0:
            self.age_days += 1
        else:
            if random.random() < 0.3:
                self.age_days += 1

    def __str__(self):
        return (
            f"Fish(stage={self.stage}, weight={self.weight:.2f}g, "
            f"age={self.age_days} days)"
        )
