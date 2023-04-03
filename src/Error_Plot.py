from math import inf
from deprecated import deprecated
import numpy as np
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
from PolynomialPricingMethod.utils.plot_utils import plotErrorRegression
from PricingMethod.CallCloseForm import BSMCloseForm, MertonCloseForm
from PolynomialPricingMethod.utils.Tools import timeit

save_dir = (
    "/Users/lindazhong/Documents/Code/Projects/PolyOptionPricing/Data/Error/Error_Plot"
)
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
# SVCDJ
SVCDJ_Y = 0.0968
SVCDJ_intensity = 1.64
SVCDJ_mu0 = -0.03
SVCDJ_mu_xy = -7.87
SVCDJ_sigma_xy = 0.22
SVCDJ_sigma_y = 0.61
SVCDJ_corr = -0.1
SVCDJ_Y_bar = 0.49
SVCDJ_theta_y = 0.0036
SVCDJ_k_y = 5.06

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
    # "SVCDJ": SVCDJ(r=r, T=T, Y=SVCDJ_Y, intensity=SVCDJ_intensity,mu0=SVCDJ_mu0, mu_xy=SVCDJ_mu_xy, sigma_xy=SVCDJ_sigma_xy, sigma_y=SVCDJ_sigma_y, corr=SVCDJ_corr, Y_bar=SVCDJ_Y_bar, theta_y=SVCDJ_theta_y, k_y=SVCDJ_k_y),
    "VG": VG(r=r, sigma=sigma, T=T, gamma_mean=gamma_mean, gamma_var=gamma_var),
    "NIG": NIG(r=r, T=T, delta=delta, alpha=alpha, beta=beta),
}


def Call():
    # Polynomial Setting
    S0 = 100
    poly_coeff = [-100, 1]
    positive_interval = [100, inf]

    ref_val_close_form = {
        "GBM": BSMCloseForm(S0=S0, r=r, sigma=sigma, T=T, K=100).getValue(),
        "Heston": 6.8816576853411586256470400257967412471771240234375,
        "MJD": MertonCloseForm(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            K=100,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
        ).getValue(5000),
        "KJD": None,
        "SVJ": None,
        "VG": None,
        "NIG": None,
        # "SVCDJ" : 12.609768306568486906371617806144058704376220703125
    }

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

        ref_val = ref_val_close_form[process_name]
        if ref_val is None:
            ref_val = PolyByCosMethod(
                S0=S0,
                T=T,
                r=r,
                sigma=sigma,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=best_positive_interval,
                N=10000,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            ).getValue()
        ref_val_close_form[process_name] = ref_val

        plot_N_Config = {
            "GBM": list(range(0, 1000, 1))[1:],
            "Heston": list(range(0, 1000, 1))[1:],
            "MJD": list(range(0, 1000, 1))[1:],
            "KJD": list(range(0, 1000, 1))[1:],
            "SVJ": list(range(0, 1000, 1))[1:],
            "SVCDJ": list(range(0, 5000, 1))[1:],
            "VG": list(range(0, 5000, 1))[1:],
            "NIG": list(range(0, 1000, 1))[1:],
        }

        N_list = plot_N_Config[process_name]
        val_list = []
        for N in N_list:
            res = PolyByCosMethod(
                S0=S0,
                T=T,
                r=r,
                sigma=sigma,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=best_positive_interval,
                N=N,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
            ans, time_consuming = timeit(res.getValue)
            val_list.append(ans)
        plotErrorRegression(
            N_list,
            np.abs(np.array(val_list) - ref_val),
            save_dir,
            "error-plot-" + process_name + "-call",
        )
    print(ref_val_close_form)


def RightUp():
    # Polynomial Setting
    S0 = 90
    poly_coeff = [-20, -5, 0.05]
    positive_interval = [10 * (5 + sqrt(29)), inf]

    ref_val_close_form = {
        "GBM": None,
        "Heston": None,
        "MJD": None,
        "KJD": None,
        "SVJ": None,
        "VG": None,
        "NIG": None,
    }
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
        ref_val = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=10000,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ).getValue()
        ref_val_close_form[process_name] = ref_val
        plot_N_Config = {
            "GBM": list(range(0, 1000, 1))[1:],
            "Heston": list(range(0, 1000, 1))[1:],
            "MJD": list(range(0, 1000, 1))[1:],
            "KJD": list(range(0, 1000, 1))[1:],
            "SVJ": list(range(0, 1000, 1))[1:],
            "SVCDJ": list(range(0, 5000, 1))[1:],
            "VG": list(range(0, 5000, 1))[1:],
            "NIG": list(range(0, 1000, 1))[1:],
        }

        val_list = []
        N_list = plot_N_Config[process_name]
        for N in N_list:
            res = PolyByCosMethod(
                S0=S0,
                T=T,
                r=r,
                sigma=sigma,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=best_positive_interval,
                N=N,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
            ans, time_consuming = timeit(res.getValue)
            val_list.append(ans)
        plotErrorRegression(
            N_list,
            np.abs(np.array(val_list) - ref_val),
            save_dir,
            "error-plot-" + process_name + "-rightup",
        )
    print(f"{ref_val_close_form['VG']:.15f}")


@deprecated()
def LeftUp():
    # Polynomial Setting
    S0 = 110
    poly_coeff = [947.1, -30.164, 0.309, -0.001]
    positive_interval = [0, 77, 82, 150]
    ref_val_close_form = {
        "GBM": None,
        "Heston": None,
        "MJD": None,
        "KJD": None,
        "SVJ": None,
        "VG": None,
        "NIG": None,
    }
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
        ref_val = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=10000,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ).getValue()
        ref_val_close_form[process_name] = ref_val
        plot_N_Config = {
            "GBM": list(range(0, 1000, 1))[1:],
            "Heston": list(range(0, 1000, 1))[1:],
            "MJD": list(range(0, 1000, 1))[1:],
            "KJD": list(range(0, 1000, 1))[1:],
            "SVJ": list(range(0, 1000, 1))[1:],
            "SVCDJ": list(range(0, 2000, 1))[1:],
            "VG": list(range(0, 5000, 1))[1:],
            "NIG": list(range(0, 1000, 1))[1:],
        }
        val_list = []
        N_list = plot_N_Config[process_name]
        for N in N_list:
            res = PolyByCosMethod(
                S0=S0,
                T=T,
                r=r,
                sigma=sigma,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=best_positive_interval,
                N=N,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
            ans, time_consuming = timeit(res.getValue)
            val_list.append(ans)
        plotErrorRegression(
            N_list,
            np.abs(np.array(val_list) - ref_val),
            save_dir,
            "error-plot-" + process_name + "-leftup",
        )
    print(ref_val_close_form)


@deprecated()
def BothUp():
    # Polynomial Setting
    S0 = 15
    poly_coeff = [44.235, -39.474, 5.4793, -0.2358, 0.0031]
    positive_interval = [0, 1.363962, 10.620047, 25.599102, 38.481405, inf]
    ref_val_close_form = {
        "GBM": None,
        "Heston": None,
        "MJD": None,
        "KJD": None,
        "SVJ": None,
        "VG": None,
        "NIG": None,
    }
    for process_name in processes.keys():
        print("========", process_name, "=============")
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-12,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()
        ref_val = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=10000,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ).getValue()
        ref_val_close_form[process_name] = ref_val
        plot_N_Config = {
            "GBM": list(range(0, 1000, 1))[1:],
            "Heston": list(range(0, 1000, 1))[1:],
            "MJD": list(range(0, 1000, 1))[1:],
            "KJD": list(range(0, 1000, 1))[1:],
            "SVJ": list(range(0, 1000, 1))[1:],
            "SVCDJ": list(range(0, 2000, 1))[1:],
            "VG": list(range(0, 10000, 1))[1:],
            "NIG": list(range(0, 1000, 1))[1:],
        }
        val_list = []
        N_list = plot_N_Config[process_name]
        for N in N_list:
            res = PolyByCosMethod(
                S0=S0,
                T=T,
                r=r,
                sigma=sigma,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=best_positive_interval,
                N=N,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
            ans, time_consuming = timeit(res.getValue)
            val_list.append(ans)
        plotErrorRegression(
            N_list,
            np.abs(np.array(val_list) - ref_val),
            save_dir,
            "error-plot-" + process_name + "-bothup",
        )
    print(ref_val_close_form)


def BothDown():
    # Polynomial Setting
    S0 = 30
    poly_coeff = [-44.235, 39.474, -5.4793, 0.2358, -0.0031]
    positive_interval = [1.363962, 10.620047, 25.599102, 38.481405]
    ref_val_close_form = {
        "GBM": None,
        "Heston": None,
        "MJD": None,
        "KJD": None,
        "SVJ": None,
        "VG": None,
        "NIG": None,
    }
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
        ref_val = PolyByCosMethod(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=best_positive_interval,
            N=10000,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ).getValue()
        ref_val_close_form[process_name] = ref_val
        plot_N_Config = {
            "GBM": list(range(0, 1000, 1))[1:],
            "Heston": list(range(0, 1000, 1))[1:],
            "MJD": list(range(0, 1000, 1))[1:],
            "KJD": list(range(0, 1000, 1))[1:],
            "SVJ": list(range(0, 1000, 1))[1:],
            "SVCDJ": list(range(0, 1000, 1))[1:],
            "VG": list(range(0, 10000, 1))[1:],
            "NIG": list(range(0, 1000, 1))[1:],
        }
        val_list = []
        N_list = plot_N_Config[process_name]
        for N in N_list:
            res = PolyByCosMethod(
                S0=S0,
                T=T,
                r=r,
                sigma=sigma,
                process_cf=processes[process_name],
                poly_coeff=poly_coeff,
                positive_interval=best_positive_interval,
                N=N,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
            ans, time_consuming = timeit(res.getValue)
            val_list.append(ans)
        plotErrorRegression(
            N_list,
            np.abs(np.array(val_list) - ref_val),
            save_dir,
            "error-plot-" + process_name + "-bothdown",
        )
    print(ref_val_close_form)


if __name__ == "__main__":
    print("===========Call==============")
    Call()
    print("===========Right Up===========")
    RightUp()
    # print("Left Up")
    # LeftUp()
    # print("Both Up")
    # BothUp()
    print("==========Both Down==========")
    BothDown()
