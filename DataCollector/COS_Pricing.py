from PolynomialPricingMethod.COSMethod import PolyByCosMethod
from PolynomialPricingMethod.utils.CharacteristicFunc import *
from PolynomialPricingMethod.utils.DensityTools import DensityRecover
from math import inf
from PolynomialPricingMethod.utils.Tools import timeit

###################### Process Setting ######################
# Basic
r = 0.06
# d = 0.06
T = 0.5
sigma = 0.2
# Stochastic volatility
std_of_var_process = 0.1
mean_reversion_speed = 3
long_term_var_mean = 0.04
corr = -0.1
# Jump process
jump_intensity = 140
jump_mean = 0.01
jump_var = 0.02 ** 2
# KDJ
jump_intensity_kdj = 1
p = 0.4
eta1 = 10
eta2 = 5
# Gamma
gamma_mean = -0.14
gamma_var = 0.2
# NIG
delta = 1.326
alpha = 15.624
beta = 4.025
# SVJJ
SVJJ_corr = -0.82
SVJJ_corr_J = -0.38
SVJJ_v_bar = 0.008
SVJJ_k_v = 3.46
SVJJ_sigma_v = 0.14
SVJJ_intensity = 0.47
# SVJJ_mu_bar = -0.10
SVJJ_sigma_y = 0.0001
SVJJ_mu_v = 0.05
SVJJ_v_0 = 0.087 ** 2
SVJJ_mu_y = -0.03
# SVCDJ
SVCDJ_Y_bar = 0.49
SVCDJ_Y_0 = 0.0968
SVCDJ_corr = - 0.1
SVCDJ_intensity = 1.64
SVCDJ_mu_0 = -0.03
SVCDJ_mu_xy = -7.78
SVCDJ_sigma_xy = 0.22
SVCDJ_theta_y = 0.0036
SVCDJ_sigma_y = 0.61
SVCDJ_k_y = 5.06


processes = {
            #"GBM": GBM(r=r, sigma=sigma, T=T),
           #"Heston": Heston(r=r, sigma=sigma, T=T, mean_reversion_speed=mean_reversion_speed, long_term_var_mean=long_term_var_mean, corr=corr, std_of_var_process=std_of_var_process),
           #"MJD": MJD(r=r, sigma=sigma, T=T, jump_intensity=jump_intensity, jump_mean=jump_mean, jump_var=jump_var),
           #"KJD": KJD(r=r, sigma=sigma, T=T, jump_intensity=jump_intensity_kdj, p=p, eat1=eta1, eat2=eta2),
           #"SVJ": SVJ(r=r, sigma=sigma, T=T, mean_reversion_speed=mean_reversion_speed, long_term_var_mean=long_term_var_mean, corr=corr, std_of_var_process=std_of_var_process, jump_intensity=jump_intensity, jump_mean=jump_mean, jump_var=jump_var),
        #"SVJJ": SVJJ(r=r, d=d, T=T, sigma=sqrt(SVJJ_v_0), intensity=SVJJ_intensity, sigma_v=SVJJ_sigma_v, corr=SVJJ_corr, k_v=SVJJ_k_v, v_bar=SVJJ_v_bar, mu_v=SVJJ_mu_v, mu_y=SVJJ_mu_y, sigma_y=SVJJ_sigma_y, corr_J=SVJJ_corr_J),
            #"SVCDJ": SVCDJ(r=r, d=d, T=T, sigma=sqrt(SVCDJ_Y_0), intensity=SVCDJ_intensity, corr=SVCDJ_corr, k_y=SVCDJ_k_y, sigma_y=SVCDJ_sigma_y, theta_y=SVCDJ_theta_y, mu_xy=SVCDJ_mu_xy, Y_bar=SVCDJ_Y_bar, mu_0=SVCDJ_mu_0, sigma_xy=SVCDJ_sigma_xy)
           #"VG": VG(r=r, sigma=sigma, T=T, gamma_mean=gamma_mean,gamma_var=gamma_var),
           "NIG": NIG(r=r, T=T, delta=delta, alpha=alpha, beta=beta)
}

N_list = [32, 64, 128, 256, 512, 1024, 10000]

def Call():
    # Polynomial Setting
    S0 = 100
    poly_coeff = [-100, 1]
    positive_interval = [100, inf]

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(S0=S0, process_cf=processes[process_name], poly_coeff=poly_coeff,
                                        positive_interval=positive_interval, error_acceptance=1e-15)
        lower_limit, upper_limit, best_positive_interval[0], best_positive_interval[-1] = densityRecover.getIntegralRangeAndInterval()

        for N in N_list:
            res = PolyByCosMethod(S0=S0, T=T, r=r, sigma=sigma, process_cf=processes[process_name],
                                  poly_coeff=poly_coeff, positive_interval=best_positive_interval,
                                  N=N, lower_limit=lower_limit, upper_limit=upper_limit)
            ans, time_consuming = timeit(res.getValue)
            print(f"N={N}: value: {ans:.53f}")


def RightUp():
    # Polynomial Setting
    S0 = 90
    poly_coeff = [-20, -5, 0.05]
    positive_interval = [10 * (5 + sqrt(29)), inf]

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(S0=S0, process_cf=processes[process_name], poly_coeff=poly_coeff,
                                        positive_interval=positive_interval, error_acceptance=1e-8)
        lower_limit, upper_limit, best_positive_interval[0], best_positive_interval[
            -1] = densityRecover.getIntegralRangeAndInterval()

        for N in N_list:
            res = PolyByCosMethod(S0=S0, T=T, r=r, sigma=sigma, process_cf=processes[process_name],
                                  poly_coeff=poly_coeff, positive_interval=best_positive_interval, N=N,
                                  lower_limit=lower_limit, upper_limit=upper_limit)
            ans, time_consuming = timeit(res.getValue)
            print(f"N={N}: value: {ans:.53f}")

def LeftUp():
    # Polynomial Setting
    S0 = 110
    poly_coeff = [947.1, -30.164, 0.309, -0.001]
    positive_interval = [0, 77, 82, 150]

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(S0=S0, process_cf=processes[process_name], poly_coeff=poly_coeff,
                                        positive_interval=positive_interval, error_acceptance=1e-15)
        lower_limit, upper_limit, best_positive_interval[0], best_positive_interval[
            -1] = densityRecover.getIntegralRangeAndInterval()

        for N in N_list:
            res = PolyByCosMethod(S0=S0, T=T, r=r, sigma=sigma, process_cf=processes[process_name],
                                  poly_coeff=poly_coeff, positive_interval=best_positive_interval, N=N,
                                  lower_limit=lower_limit, upper_limit=upper_limit)
            ans, time_consuming = timeit(res.getValue)
            print(f"N={N}: value: {ans:.53f}")

def BothUp():
    # Polynomial Setting
    S0 = 15
    poly_coeff = [44.235, -39.474, 5.4793, -0.2358, 0.0031]
    positive_interval = [0, 1.363962, 10.620047, 25.599102, 38.481405, inf]

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(S0=S0, process_cf=processes[process_name], poly_coeff=poly_coeff,
                                        positive_interval=positive_interval, error_acceptance=1e-8)
        lower_limit, upper_limit, best_positive_interval[0], best_positive_interval[
            -1] = densityRecover.getIntegralRangeAndInterval()

        for N in N_list:
            res = PolyByCosMethod(S0=S0, T=T, r=r, sigma=sigma, process_cf=processes[process_name],
                                  poly_coeff=poly_coeff, positive_interval=best_positive_interval, N=N,
                                  lower_limit=lower_limit, upper_limit=upper_limit)
            ans, time_consuming = timeit(res.getValue)
            print(f"N={N}: value: {ans:.53f}")

def BothDown():
    # Polynomial Setting
    S0 = 30
    poly_coeff = [-44.235, 39.474, -5.4793, 0.2358, -0.0031]
    positive_interval = [1.363962, 10.620047, 25.599102, 38.481405]

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(S0=S0, process_cf=processes[process_name], poly_coeff=poly_coeff,
                                        positive_interval=positive_interval, error_acceptance=1e-8)
        lower_limit, upper_limit, best_positive_interval[0], best_positive_interval[
            -1] = densityRecover.getIntegralRangeAndInterval()

        for N in N_list:
            res = PolyByCosMethod(S0=S0, T=T, r=r, sigma=sigma, process_cf=processes[process_name],
                                  poly_coeff=poly_coeff, positive_interval=best_positive_interval, N=N,
                                  lower_limit=lower_limit, upper_limit=upper_limit)
            ans, time_consuming = timeit(res.getValue)
            print(f"N={N}: value: {ans:.53f}")

if __name__ == "__main__":
    print("Call")
    Call()
    # print("Right Up")
    # RightUp()
    # print("Left Up")
    # LeftUp()
    # print("Both Up")
    # BothUp()
    # print("Both Down")
    # BothDown()