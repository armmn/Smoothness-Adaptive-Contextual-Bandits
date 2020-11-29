import numpy as np
import math

from Bins import BinLepski


class Lepski:
    def __init__(self, T, beta_min, beta_max, gamma, sample_mult_step=2, len_base_div=2, under_smooth_coef=1, l_coef=1):
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma = gamma
        self.sample_mult_step = sample_mult_step  # the multiplicative step in number of samples after each round
        self.len_base_div = len_base_div  # the base for dividing side-length of hyper cubes
        self.under_smooth_coef = under_smooth_coef  # the multiplier for decreasing the under-smoothing term
        self.l_coef = l_coef  # the multiplier for increasing l

        self.l = int(np.ceil(l_coef * beta_min * math.log(T, len_base_div) / (2 * beta_max + 1) ** 2))
        self.r_min = np.ceil(2 * self.l * beta_min)
        self.r_bar = 2 * self.l * self.beta_max + (2 / self.beta_min + 4) * math.log(np.log(self.T), sample_mult_step)
        self.bins = [BinLepski(T=T,
                               depth=self.l,
                               a1=k * len_base_div**(-self.l),
                               a2=(k + 1) * len_base_div**(-self.l),
                               r_bar=self.r_bar,
                               gamma=self.gamma,
                               beta_min=self.beta_min,
                               x_predict=[(k + j/100) * len_base_div**(-self.l) for j in range(101)],
                               bw_exp_2=self.l + 1 / beta_min * math.log(np.log(T), sample_mult_step),
                               sample_mult_step=sample_mult_step,
                               len_base_div=len_base_div,
                               r_min=self.r_min)
                     for k in range(int(len_base_div ** self.l))]
        self.bins_edges = dict()
        self.bins_edges['lower'] = [k * len_base_div ** (-self.l) for k in range(int(len_base_div ** self.l))]
        self.bins_edges['higher'] = [(k + 1) * len_base_div ** (-self.l) for k in range(int(len_base_div ** self.l))]

        self.prev_arm = np.nan
        self.prev_bin_idx = np.nan

        self.beta_hat = beta_min
        self.t = 0

    def _determine_bin(self, x):
        bin_idx = np.searchsorted(self.bins_edges['lower'], x, side='right') - 1
        if (bin_idx >= 0) and (x <= self.bins_edges['higher'][bin_idx]):
            return bin_idx
        else:
            bin_idx = None
            return bin_idx

    def get_arm(self, x):
        self.t += 1
        bin_idx = self._determine_bin(x)
        if bin_idx is not None:
            arm = self.bins[bin_idx].get_arm()
        else:
            arm = np.random.randint(0,2)
        self.prev_arm = arm
        self.prev_bin_idx = bin_idx
        return arm

    def collect_observation(self, x, y, arm):
        if self.prev_bin_idx is None:
            return self.beta_hat, False
        bin_ = self.bins[self.prev_bin_idx]
        bin_.collect_observation(x, y, arm)
        r_last_min = np.nanmin([bin_.r_last for bin_ in self.bins])

        if (not np.isnan(r_last_min)) and (r_last_min <= np.nanmax([bin_.r for bin_ in self.bins])):
            r_last_min_computed_flag = True
            self.beta_hat = (r_last_min - (2 / self.beta_min + 4) *
                             self.under_smooth_coef * math.log(np.log(self.T), self.sample_mult_step)) / 2 / self.l
            print(self.beta_hat, self.bins[self.prev_bin_idx].r, self.t)
        else:
            r_last_min_computed_flag = False

        return self.beta_hat, r_last_min_computed_flag
