import numpy as np
import matplotlib.pyplot as plt

from ABSE import ABSE
from SACB import SACB
from Simulator import Simulator


def phi(x, beta):
    if np.abs(x) > 1:
        return 0
    else:
        return (1 - np.abs(x)) ** beta


def shift_scale_phi(x, beta, M, q):
    return M ** (-beta) * phi(x=M * (x - q), beta=beta)


def f1_unwrapped_nss(x, beta, M, C, m):
    i = int(M * x)
    if i >= m:
        return 0
    sgn = (-1) ** i
    q = (i + 1 / 2) / M
    y = sgn * C * shift_scale_phi(x, beta, 2 * M, q)
    return y


def f1_unwrapped_ss(x, beta, M, C, m, L1):
    if x < 1 / 2:
        return 1 / 2 + L1 * (1 / 2) ** beta / 2 - L1 * x ** beta / 2
    else:
        return 1 / 2 + 2 ** (-beta) * f1_unwrapped_nss(2 - 2 * x, beta, M, C, m)


def f2_unwrapped_ss(x, beta, L1):
    if x < 1 / 2:
        return 1 / 2 + L1 * (1 / 2) ** beta / 2 - L1 * x ** beta / 2
    else:
        return 1/2


class SimulatorExp212:
    def __init__(self, T, beta, beta_hat, L, gamma_ABSE, L_ABSE, beta_min, beta_max, gamma_Lepski, sigma, alpha=1,
                 C=None, L1=None, sample_mult_step=2, len_base_div=2, under_smooth_coef=1, l_coef=1):
        """Based on depth"""
        self.T = T
        self.beta = beta
        self.beta_hat = beta_hat
        self.L = L
        self.gamma_ABSE = gamma_ABSE
        self.L_ABSE = L_ABSE
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_Lepski = gamma_Lepski
        self.sigma = sigma  # Normal noise standard deviation
        if C is None:
            self.C = L
        else:
            self.C = C
        if L1 is None:
            self.L1 = L
        else:
            self.L1 = L1
        self.alpha = alpha  # Margin parameter
        self.sample_mult_step = sample_mult_step
        self.len_base_div = len_base_div
        self.under_smooth_coef = under_smooth_coef
        self.l_coef = l_coef

        k_0_hat = int(np.ceil(np.log2(T / 2 / np.log(2)) / (1 + 2 * beta_hat)))
        M = 2 ** k_0_hat
        m = 2 * (int(np.ceil(M ** (1 - alpha * beta)/2)))
        self.reg_trajectory = []

        def f1(x): return f1_unwrapped_ss(x, beta, M, self.C, m, self.L1)

        def f2(x): return f2_unwrapped_ss(x, beta, self.L)

        self.payoff_funcs = [np.vectorize(f1, otypes=[np.float]), np.vectorize(f2, otypes=[np.float])]

        self.algs = [ABSE(T, beta, L_ABSE, gamma_ABSE, sigma=sigma),
                     ABSE(T, beta_hat, L_ABSE, gamma_ABSE, sigma=sigma),
                     SACB(T, beta_min, beta_max, gamma_Lepski, gamma_ABSE, L_ABSE, sigma=sigma,
                          sample_mult_step=sample_mult_step,
                          len_base_div=len_base_div, under_smooth_coef=under_smooth_coef, l_coef=l_coef)
                     ]

    def run(self, num_iters, parallel=False):
        simulator = Simulator(self.T, self.payoff_funcs, num_iters, self.algs, self.sigma, parallel=parallel)
        simulator.run()
        self.reg_trajectory = simulator.get_reg_trajectory()
        mean_cum_reg = np.sum(np.mean(self.reg_trajectory, axis=1), axis=1)
        np.save('mean_cum_reg/Exp212_T_{}_beta_{}_beta_hat_{}'
                '_L_{}_gamma_ABSE_{}'
                '_L_ABSE_{}_beta_min_{}_beta_max'
                '_{}_gamma_Lepski_{}_sigma_{}_alpha_{}_C_{}_L1_{}_sample_mult'
                '_step_{}_under_smooth_coef_{}'.format(self.T, self.beta, self.beta_hat, self.L,
                                                       self.gamma_ABSE, self.L_ABSE, self.beta_min, self.beta_max,
                                                       self.gamma_Lepski, self.sigma, self.alpha, self.C,
                                                       self.L1, self.sample_mult_step, self.under_smooth_coef),
                mean_cum_reg)

    def plot_results(self):
        fig, ax = plt.subplots(1)
        ax.plot(np.cumsum(np.mean(self.reg_trajectory[0, :, :], axis=0)), lw=2, label='ABSE_beta', color='blue')
        ax.plot(np.cumsum(np.mean(self.reg_trajectory[1, :, :], axis=0)), lw=2, label='ABSE_beta_hat', color='green')
        ax.plot(np.cumsum(np.mean(self.reg_trajectory[2, :, :], axis=0)), lw=2, label='SACB', color='red')
        ax.legend(loc='upper left')
        ax.set_xlabel('$t$')
        ax.set_ylabel('Regret Trajectory')
        ax.grid()
        plt.savefig('plots/Exp212_T_{}_beta_{}_beta_hat_{}'
                    '_L_{}_gamma_ABSE_{}'
                    '_L_ABSE_{}_beta_min_{}_beta_max'
                    '_{}_gamma_Lepski_{}_sigma_{}_alpha_{}_C_{}_L1_{}_sample_mult_step_{}'
                    '_under_smooth_coef_{}.png'.format(self.T, self.beta, self.beta_hat, self.L,self.gamma_ABSE,
                                                       self.L_ABSE, self.beta_min, self.beta_max,
                                                       self.gamma_Lepski, self.sigma, self.alpha, self.C,
                                                       self.L1, self.sample_mult_step, self.under_smooth_coef))
