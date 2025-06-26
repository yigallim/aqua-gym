import math
from datetime import datetime
from utils.config import Config

class IndividualGrowthModel:
    def __init__(self, latitude=None):
        Config.load()
        if latitude is None:
            latitude = Config.ind_growth_model.latitude.get("guangdong", 0.0)
        self.latitude = latitude

        self.day_of_year = datetime.now().timetuple().tm_yday
        self.rho = self._compute_photoperiod_factor(self.day_of_year, self.latitude)

    def set_day_of_year(self, day_of_year):
        self.day_of_year = day_of_year
        self.rho = self._compute_photoperiod_factor(day_of_year, self.latitude)

    def _compute_photoperiod_factor(self, day_of_year, latitude):
        tilt = math.radians(23.439)
        j = math.pi / 182.625
        lat_rad = math.radians(latitude)
        m = 1 - math.tan(lat_rad) * math.tan(tilt * math.cos(j * day_of_year))
        m = max(0.0, min(2.0, m))
        frac = math.acos(1 - m) / math.pi
        hours = frac * 24.0
        return hours / 12.0

    def tau(self, T):
        ig = Config.ind_growth_model
        if T >= ig.T_opt:
            return math.exp(-ig.kappa * ((T - ig.T_opt) / (ig.T_max - ig.T_opt))**4)
        else:
            return math.exp(-ig.kappa * ((ig.T_opt - T) / (ig.T_opt - ig.T_min))**4)

    def sigma(self, DO):
        ig = Config.ind_growth_model 
        if DO > ig.DO_crit:
            return 1.0
        elif ig.DO_min <= DO <= ig.DO_crit:
            return (DO - ig.DO_min) / (ig.DO_crit - ig.DO_min)
        else:
            return 0.0

    def nu(self, UIA):
        ig = Config.ind_growth_model
        if UIA < ig.UIA_crit:
            return 1.0
        elif ig.UIA_crit <= UIA <= ig.UIA_max:
            return (ig.UIA_max - UIA) / (ig.UIA_max - ig.UIA_crit)
        else:
            return 0.0


    def compute_anabolism(self, f, T, DO, UIA, w):
        if f == 0: return 0
        ig = Config.ind_growth_model
        bm = Config.biomass_model

        tau = self.tau(T)
        sigma = self.sigma(DO)
        v = self.nu(UIA)

        f_opt = 0.68
        left_width = 0.4
        right_width = 0.4

        if f < f_opt:
            feed_efficiency = math.exp(-(abs(f - f_opt) / left_width) ** 2.8)
        else:
            feed_efficiency = math.exp(-(abs(f - f_opt) / right_width) ** 2.8)

        return ig.h * self.rho * feed_efficiency * ig.b * (1 - ig.a) * tau * sigma * v * (w ** bm.m)

    # def compute_anabolism(self, f, T, DO, UIA, w):
    #     ig = Config.ind_growth_model
    #     bm = Config.biomass_model
    #     tau = self.tau(T)
    #     sigma = self.sigma(DO)
    #     v = self.nu(UIA)
    #     return ig.h * self.rho * f * ig.b * (1 - ig.a) * tau * sigma * v * (w ** bm.m)

    def compute_catabolism(self, T, w):
        ig = Config.ind_growth_model
        bm = Config.biomass_model
        return ig.k_min * math.exp(ig.j * (T - ig.T_min)) * (w ** bm.n)

    def compute_growth(self, f, T, DO, UIA, w):
        A = self.compute_anabolism(f, T, DO, UIA, w)
        C = self.compute_catabolism(T, w)
        base = A - C

        ig = Config.ind_growth_model
        w_mid = ig.w_threshold
        k = ig.slowdown_gamma

        slowdown = 1.0 / (1.0 + math.exp( k * (w - w_mid) ))
        return base * slowdown
