from Exp211 import SimulatorExp211

T = int(2*10**6)
sigma = 0.05
beta_min = .4
beta_max = 1
base = 1.1
under_smooth_coef = 0.325
l_coef = 1

beta = .9

L = 1
beta_hat = .85

gamma_ABSE = 2
L_ABSE = L

gamma_Lepski = .145

alpha = .01
C = L
L1 = L

num_iters = 1 * 10

simulator = SimulatorExp211(T, beta, beta_hat, L, gamma_ABSE, L_ABSE, beta_min, beta_max, gamma_Lepski, sigma,
                            alpha=alpha, C=C, L1=L1, sample_mult_step=base, len_base_div=base,
                            under_smooth_coef=under_smooth_coef, l_coef=l_coef)
simulator.run(num_iters, parallel=False)
simulator.plot_results()

input("Press Enter to continue...")
