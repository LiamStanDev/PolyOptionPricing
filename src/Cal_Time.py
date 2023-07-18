from math import inf
from numpy import sqrt
from PolynomialPricingMethod.COSMethod import PolyByCosMethod
from PolynomialPricingMethod.utils.CharacteristicFunc import (
    GBM,
    Heston,
    MJD,
    KJD,
    SVJ,
    VG,
    NIG,
)
from PolynomialPricingMethod.utils.DensityTools import DensityRecover
from PolynomialPricingMethod.utils.Tools import timeit

# ###################### Process Setting ######################
# Basic
r = 0.05
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
jump_var = 0.02**2
jump_intensity_kdj = 1
# KDJ
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

processes = {
    "GBM": GBM(r=r, sigma=sigma, T=T),
    "Heston": Heston(
        r=r,
        sigma=sigma,
        T=T,
        mean_reversion_speed=mean_reversion_speed,
        long_term_var_mean=long_term_var_mean,
        corr=corr,
        std_of_var_process=std_of_var_process,
    ),
    "MJD": MJD(
        r=r,
        sigma=sigma,
        T=T,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_var=jump_var,
    ),
    "KJD": KJD(
        r=r,
        sigma=sigma,
        T=T,
        jump_intensity=jump_intensity_kdj,
        p=p,
        eat1=eta1,
        eat2=eta2,
    ),
    "SVJ": SVJ(
        r=r,
        sigma=sigma,
        T=T,
        mean_reversion_speed=mean_reversion_speed,
        long_term_var_mean=long_term_var_mean,
        corr=corr,
        std_of_var_process=std_of_var_process,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_var=jump_var,
    ),
    "NIG": NIG(r=r, T=T, delta=delta, alpha=alpha, beta=beta),
    "VG": VG(r=r, sigma=sigma, T=T, gamma_mean=gamma_mean, gamma_var=gamma_var),
}


def Call():
    # Polynomial Setting
    S0 = 100
    poly_coeff = [-100, 1]
    positive_interval = [100, inf]

    for process_name in processes.keys():
        print("========", process_name, "=============")
        best_positive_interval = positive_interval.copy()
        # VG 會爆掉
        if process_name == "VG":
            densityRecover = DensityRecover(
                S0=S0,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=positive_interval,
                error_acceptance=1e-12,
            )
        else:
            densityRecover = DensityRecover(
                S0=S0,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=positive_interval,
                error_acceptance=1e-15,
            )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        N_dict = {
            "GBM": 187,
            "Heston": 193,
            "MJD": 168,
            "KJD": 252,
            "SVJ": 179,
            "VG": 188,
            "NIG": 222,
        }

        res = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=N_dict[process_name],
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        ans, time_consuming = timeit(res.getValue)
        print(process_name, "time: ", time_consuming)


def RightUp():
    # Polynomial Setting
    S0 = 90
    poly_coeff = [-20, -5, 0.05]
    positive_interval = [10 * (5 + sqrt(29)), inf]

    for process_name in processes.keys():
        print("========", process_name, "=============")
        best_positive_interval = positive_interval.copy()
        # VG 會爆掉
        if process_name == "VG":
            densityRecover = DensityRecover(
                S0=S0,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=positive_interval,
                error_acceptance=1e-12,
            )
        else:
            densityRecover = DensityRecover(
                S0=S0,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=positive_interval,
                error_acceptance=1e-15,
            )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        N_dict = {
            "GBM": 278,
            "Heston": 768,
            "MJD": 165,
            "KJD": 307,
            "SVJ": 168,
            "VG": 446,
            "NIG": 254,
        }

        res = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=N_dict[process_name],
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        ans, time_consuming = timeit(res.getValue)
        print(process_name, "time: ", time_consuming)


def BothDown():
    # Polynomial Setting
    S0 = 30
    poly_coeff = [-44.235, 39.474, -5.4793, 0.2358, -0.0031]
    positive_interval = [1.363962, 10.620047, 25.599102, 38.481405]

    for process_name in processes.keys():
        print("========", process_name, "=============")
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-15,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        N_dict = {
            "GBM": 161,
            "Heston": 173,
            "MJD": 37,
            "KJD": 191,
            "SVJ": 36,
            "VG": 1156,
            "NIG": 115,
        }
        res = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=N_dict[process_name],
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        ans, time_consuming = timeit(res.getValue)
        print(process_name, "time: ", time_consuming)


if __name__ == "__main__":
    Call()
    RightUp()
    BothDown()
