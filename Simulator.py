import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import copy as copy
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()


class Simulator:
    def __init__(self, T, payoff_funcs, num_iters, algs, sigma, parallel=False):
        self.T = T
        self.payoff_funcs = payoff_funcs
        self.num_iters = num_iters
        self.algs = algs  # list of algorithms to simulate
        self.sigma = sigma  # Normal noise standard deviation
        self.parallel = parallel

        self.reg_trajectory = None

    def run(self):
        x = np.random.rand(self.num_iters, self.T)  # contexts
        mean_rewards = np.concatenate(tuple([np.expand_dims(self.payoff_funcs[arm](x), 0) for arm in range(2)]))
        # y = np.random.binomial(n=1, p=mean_rewards)  # outcomes
        y = np.random.normal(loc=mean_rewards, scale=self.sigma)  # outcomes
        max_reward = np.max(mean_rewards, 0)

        if self.parallel:
            reg_trajectories = Parallel(n_jobs=num_cores)(delayed(self.run_one_iter)(iter_, x[iter_, :],
                                                                                     y[:, iter_, :],
                                                                                     mean_rewards[:, iter_, :],
                                                                                     max_reward[iter_, :])
                                                          for iter_ in tqdm(range(self.num_iters)))
            self.reg_trajectory = np.concatenate(reg_trajectories, axis=1)
            # self.reg_trajectory = reg_trajectories
        else:
            self.reg_trajectory = np.zeros((len(self.algs), self.num_iters, self.T))
            for iter_ in tqdm(range(self.num_iters)):
                self.reg_trajectory[:, iter_, :] = np.squeeze(self.run_one_iter(iter_, x[iter_, :],
                                                                                y[:, iter_, :],
                                                                                mean_rewards[:, iter_, :],
                                                                                max_reward[iter_, :]), axis=1)

    def run_one_iter(self, iter_, x, y, mean_rewards, max_reward):
        print('iteration: {}'.format(iter_), end='\r', flush=True)
        # initiate variables for one iteration
        actions = [None for _ in self.algs]
        reg_trajectory_one_iter = np.zeros((len(self.algs), 1, self.T))
        algs_one_iter = copy.deepcopy(self.algs)
        for t in range(self.T):
            for i in range(len(self.algs)):
                # collect the actions
                actions[i] = algs_one_iter[i].get_arm(x[t])
                # compute the regrets
                reg_trajectory_one_iter[i, 0, t] = max_reward[t] - mean_rewards[actions[i], t]
                # return feedback to algorithms
                algs_one_iter[i].collect_observation(x[t], y[actions[i], t], actions[i])
        return reg_trajectory_one_iter

    def get_reg_trajectory(self):
        return self.reg_trajectory
