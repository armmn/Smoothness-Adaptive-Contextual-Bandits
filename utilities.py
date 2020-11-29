import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def periodically_continued(a, b):
    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)


def saw(x):
    # L=1; beta=1
    if x < .5:
        return x
    else:
        return 1 - x


def prabola(x):
    # L=1; beta=2
    if x <= 1 / 4:
        return x ** 2 / 2
    elif x <= 3 / 4:
        return -x ** 2 / 2 + x / 2 - 1 / 16
    else:
        return x ** 2 / 2 - x + 1 / 2


def saw_peridic(x):
    return periodically_continued(0, 1)(saw)


def parobla_peridic(x):
    return periodically_continued(0, 1)(prabola)


def saw_f(scale, period):
    def f(x):
        x_norm = (x / period) % 1
        if x_norm < .5:
            return scale * 1
        else:
            return scale * (1 - x_norm)

    return f


def prabola_f(scale, period, intercept):
    def f(x):
        x_norm = (x / period) % 1
        if x_norm <= 1 / 4:
            return scale * (x_norm ** 2 / 2) + intercept
        elif x_norm <= 3 / 4:
            return scale * (-x_norm ** 2 / 2 + x_norm / 2 - 1 / 16) + intercept
        else:
            return scale * (x_norm ** 2 / 2 - x_norm + 1 / 2) + intercept

    return f


def plot_cum_reg(reg_trajectory, labels, plot_conf_bd=True):
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['-', '--', ':', '-.', 'o']

    fig, ax = plt.subplots(1)
    num_algs = reg_trajectory.shape[0]
    T = reg_trajectory.shape[2]
    for k in range(num_algs):
        single_cum_trajectory = np.cumsum(reg_trajectory[k, :, :], axis=1)
        single_mean_trajectory = np.mean(single_cum_trajectory, axis=0)
        color_num = k % 5
        marker_num = (k + int(k / 5)) % 5
        ax.plot(single_mean_trajectory, lw=1, label=labels[k],
                color=colors[color_num], ls=markers[marker_num])
        if plot_conf_bd:
            trajectory_std = stats.sem(single_cum_trajectory, axis=0)
            ax.fill_between(np.arange(0, T), y1=single_mean_trajectory - 2 * trajectory_std,
                            y2=single_mean_trajectory + 2 * trajectory_std, facecolor=colors[color_num], alpha=0.2)

    ax.legend(loc='upper left')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Cumulative Regret')
    ax.grid()
    return fig, ax


def plot_inst_reg(reg_trajectory, labels, plot_conf_bd=True):
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['-', '--', ':', '-.', 'o']

    fig, ax = plt.subplots(1)
    num_algs = reg_trajectory.shape[0]
    T = reg_trajectory.shape[2]
    for k in range(num_algs):
        single_inst_trajectory = reg_trajectory[k, :, :]
        single_mean_trajectory = np.mean(single_inst_trajectory, axis=0)
        color_num = k % 5
        marker_num = (k + int(k / 5)) % 5
        ax.plot(single_mean_trajectory, lw=1, label=labels[k],
                color=colors[color_num], ls=markers[marker_num])
        if plot_conf_bd:
            trajectory_std = stats.sem(single_inst_trajectory, axis=0)
            ax.fill_between(np.arange(0, T), y1=single_mean_trajectory - 2 * trajectory_std,
                            y2=single_mean_trajectory + 2 * trajectory_std, facecolor=colors[color_num], alpha=0.2)

    ax.legend(loc='upper left')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Instantaneous Regret')
    ax.grid()
    return fig, ax


def plot_reg_rate(cum_regs, labels, T_list, plot_conf_bd=True):
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['-', '--', ':', '-.', 'o']

    fig, ax = plt.subplots(1)
    num_algs = cum_regs.shape[0]
    for k in range(num_algs):
        single_cum_reg = cum_regs[k, :, :]
        single_mean_reg = np.mean(single_cum_reg, axis=0)
        color_num = k % 5
        marker_num = (k + int(k / 5)) % 5
        ax.plot(np.log(T_list), single_mean_reg, lw=1, label=labels[k],
                color=colors[color_num], ls=markers[marker_num])
        if plot_conf_bd:
            reg_std = stats.sem(single_cum_reg, axis=0)
            ax.fill_between(np.log(T_list), y1=single_mean_reg - 2 * reg_std,
                            y2=single_mean_reg + 2 * reg_std, facecolor=colors[color_num], alpha=0.2)

    ax.legend(loc='upper left')
    ax.set_xlabel('$\log(T)$')
    ax.set_ylabel('$\log(R)$')
    ax.grid()
    return fig, ax


def get_u(tau, T):
    if np.log(T / tau) < 1:
        return 2 * np.sqrt(2 / tau)
    else:
        return 2 * np.sqrt(2 * np.log(T / tau) / tau)
