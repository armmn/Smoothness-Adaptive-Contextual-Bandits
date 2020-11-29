import numpy as np

from Bins import BinABSE
from utilities import get_u


class ABSE:
    def __init__(self, T, beta, L, gamma, sigma=.5):
        # Problem parameters
        self.T = T
        self.beta = beta
        self.L = L

        # Constants used to tune the policy
        self.gamma = gamma

        # constants defined based on problem params
        self.c_0 = 2 * L
        self.k_min = 0
        self.k_0 = int(np.ceil(np.log2(T / 2 / np.log(2)) / (1 + 2 * beta)))

        # initializing the set of live bins
        l = self._get_l(self.k_min)
        self.live_bins = [BinABSE(T=T,
                                  depth=self.k_min,
                                  a1=k * 2 ** (-self.k_min),
                                  a2=(k + 1) * 2 ** (-self.k_min),
                                  l=l,
                                  gamma=self.gamma)
                          for k in range(2 ** self.k_min)]
        self.live_bins_edges = dict()
        self.live_bins_edges['lower'] = [k * 2 ** (-self.k_min) for k in range(2 ** self.k_min)]
        self.live_bins_edges['higher'] = [(k + 1) * 2 ** (-self.k_min) for k in range(2 ** self.k_min)]

        self.prev_arm = np.nan
        self.prev_live_bin_idx = np.nan

    def _get_l(self, depth):
        """get the maximum number of epochs in a bin at a certain depth before getting burst"""
        eps = 1e-6
        rhs = 2 * self.c_0 * 2 ** (-depth * self.beta)
        lower = 1
        upper = self.T

        if upper < lower:
            return lower
        if get_u(upper, self.T * 2 ** (-depth)) >= rhs:
            return upper
        if get_u(lower, self.T * 2 ** (-depth)) <= rhs:
            return lower
        mid = (upper + lower) / 2
        while abs(get_u(mid, self.T * 2 ** (-depth)) - rhs) > eps:
            if get_u(mid, self.T * 2 ** (-depth)) < rhs:
                upper = mid
            else:
                lower = mid
            mid = (upper + lower) / 2
        return int(np.floor(mid))

    def _determine_live_bin(self, x):
        live_bin_idx = np.searchsorted(self.live_bins_edges['lower'], x, side='right') - 1
        if (live_bin_idx >= 0) and (x <= self.live_bins_edges['higher'][live_bin_idx]):
            return live_bin_idx
        else:
            raise ValueError

    def get_arm(self, x):
        live_bin_idx = self._determine_live_bin(x)
        arm = self.live_bins[live_bin_idx].get_arm()
        self.prev_arm = arm
        self.prev_live_bin_idx = live_bin_idx
        return arm

    def collect_observation(self, x, y, arm):
        bin_ = self.live_bins[self.prev_live_bin_idx]
        bin_.collect_observation(x, y, arm)
        if (bin_.tau >= bin_.l) and (bin_.depth < self.k_0) and (len(bin_.active_arms) >= 2):
            self._burst()

    def _burst(self):
        bin_ = self.live_bins[self.prev_live_bin_idx]
        children = bin_.get_children()
        l = self._get_l(bin_.depth + 1)

        self.live_bins = self.live_bins[:self.prev_live_bin_idx] \
                         + [BinABSE(T=self.T,
                                    depth=bin_.depth + 1,
                                    a1=child[0],
                                    a2=child[1],
                                    l=l,
                                    gamma=self.gamma) for child in children] \
                         + self.live_bins[self.prev_live_bin_idx + 1:]

        self.live_bins_edges['lower'] = self.live_bins_edges['lower'][:self.prev_live_bin_idx] \
                                        + [child[0] for child in children] \
                                        + self.live_bins_edges['lower'][self.prev_live_bin_idx + 1:]

        self.live_bins_edges['higher'] = self.live_bins_edges['higher'][:self.prev_live_bin_idx] \
                                         + [child[1] for child in children] \
                                         + self.live_bins_edges['higher'][self.prev_live_bin_idx + 1:]
