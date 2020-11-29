from LEPSKI import Lepski
from ABSE import ABSE


class SACB:
    def __init__(self, T, beta_min, beta_max, gamma_Lepski, gamma_ABSE, L_ABSE, sigma=.5, sample_mult_step=2,
                 len_base_div=2, under_smooth_coef=1, l_coef=1):
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_Lepski = gamma_Lepski
        self.gamma_ABSE = gamma_ABSE
        self.L_ABSE = L_ABSE
        self.sigma = sigma
        self.sample_mult_step = sample_mult_step  # the multiplicative step in number of samples after each round
        self.len_base_div = len_base_div  # the base for dividing side-length of hyper cubes
        self.under_smooth_coef = under_smooth_coef  # the multiplier for decreasing the under-smoothing term
        self.l_coef = l_coef  # the multiplier for increasing l

        self.lepski = Lepski(T, beta_min, beta_max, gamma_Lepski, sample_mult_step=sample_mult_step,
                             len_base_div=len_base_div, under_smooth_coef=under_smooth_coef, l_coef=l_coef)
        self.abse = None

    def get_arm(self, x):
        if self.abse is None:
            arm = self.lepski.get_arm(x)
        else:
            arm = self.abse.get_arm(x)
        return arm

    def collect_observation(self, x, y, arm):
        if self.abse is None:
            beta_hat, r_last_min_computed_flag = self.lepski.collect_observation(x, y, arm)
            if r_last_min_computed_flag:
                beta_hat = min(max(self.beta_min, beta_hat), self.beta_max)
                self.abse = ABSE(self.T, beta_hat, self.L_ABSE, self.gamma_ABSE, sigma=self.sigma)
        else:
            self.abse.collect_observation(x, y, arm)
