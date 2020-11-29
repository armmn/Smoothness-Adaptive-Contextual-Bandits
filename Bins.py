import numpy as np

from SE import SE


class BinLepski:
    def __init__(self, T, depth, a1, a2, r_bar, gamma, beta_min, x_predict, bw_exp_2, sample_mult_step=2,
                 len_base_div=2, r_min=1):
        self.T = T
        self.depth = depth  # depth of the bin
        self.a1 = a1  # min interval range
        self.a2 = a2  # max interval range
        self.r_bar = r_bar  # maximum number of rounds allowed to collect samples
        self.gamma = gamma
        self.beta_min = beta_min
        self.x_predict = x_predict  # the points at which to compute local average
        self.bw_exp_2 = bw_exp_2  # second bandwidth exponent (j_2 in the paper)
        self.sample_mult_step = sample_mult_step  # the multiplicative step in number of samples after each round
        self.len_base_div = len_base_div  # the base for dividing side-length of hyper cubes

        # initialize
        self.n = [0, 0]  # observation counter
        self.tau = 0  # internal timer
        self.r = r_min  # initialize the first round of sampling
        self.observed_x = [[], []]
        self.observed_y = [[], []]
        self.r_last = np.nan
        self.bw = len_base_div ** (-bw_exp_2)  # the bandwidth for the second estimator

    def get_arm(self):
        if self.n[0] == self.n[1]:
            self.n[0] += 1
            return 0
        else:
            self.n[1] += 1
            self.tau += 1
            return 1

    def collect_observation(self, x, y, arm):
        self.observed_x[arm].append(x)
        self.observed_y[arm].append(y)
        if (self.tau >= self.sample_mult_step ** self.r) and (self.r <= self.r_bar) and np.isnan(self.r_last):
            # compute the local averages
            local_avg_diff = [[], []]
            for k in range(2):
                x = np.expand_dims(np.array(self.observed_x[k]), 0)
                x_pred = np.expand_dims(np.array(self.x_predict), 1)
                G = np.array((np.abs(x_pred - x) < self.bw)).astype(float)
                obs_conut = np.sum(G, 1)
                obs_conut_plus = np.expand_dims(np.maximum(1, obs_conut), 1)  # number of observations at each x_pred
                y = np.expand_dims(np.array(self.observed_y[k]), 1)
                local_avg_2 = np.matmul(G, y)
                local_avg_2 = local_avg_2 / obs_conut_plus
                local_avg_1 = np.mean(self.observed_y[k])
                local_avg_diff[k] = local_avg_2 - local_avg_1
                # correct local difference for the points at which second estimate has no observations
                for i in range(len(x_pred)):
                    if obs_conut[i] == 0:
                        local_avg_diff[k][i, 0] = 0
            local_avg_diff = np.abs(local_avg_diff)
            lhs = np.amax(local_avg_diff)
            threshold = self.gamma * np.log(self.T) ** (1 / 2 + 1 / 2 / self.beta_min) /\
                        self.sample_mult_step ** (self.r / 2)
            if np.isnan(self.r_last) and (lhs > threshold):
                self.r_last = int(self.r)
                print(self.a1)
            # reset
            self.n = [0, 0]  # observation counter
            self.tau = 0  # internal timer
            self.observed_x = [[], []]
            self.observed_y = [[], []]
            # go to the next round
            self.r += 1


class BinABSE:
    def __init__(self, T, depth, a1, a2, l, gamma):
        self.depth = depth  # depth of the bin
        self.a1 = a1  # min interval range
        self.a2 = a2  # max interval range
        self.l = l  # maximum number of samples bin is allowed to collect before getting burst

        # observed data pairs
        self.active_arms = [0, 1]
        self.tau = 0
        self.SE = SE(T=T * 2 ** (-depth), gamma=gamma)

    def get_arm(self):
        arm, self.tau, self.active_arms = self.SE.get_arm()
        return arm

    def collect_observation(self, _, y, arm):
        self.SE.collect_observation(y, arm)

    def get_children(self):
        mid = (self.a1 + self.a2) / 2
        children = [[self.a1, mid], [mid, self.a2]]
        return children
