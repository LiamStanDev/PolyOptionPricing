from math import inf
import sys

sys.path.append("/Users/lindazhong/Documents/Code/Projects/PolyOptionPricing")

from PolynomialPricingMethod.COSMethod import PolyByCosMethod
from PolynomialPricingMethod.utils.CharacteristicFunc import *
from PolynomialPricingMethod.utils.DensityTools import DensityRecover
from PolynomialPricingMethod.utils.plot_utils import plotError, plotValueWithCV
from PolynomialPricingMethod.utils.Tools import timeit

save_dir = "/Users/lindazhong/Documents/Code/Projects/PolyOptionPricing/DataCollector/Data/Error/Test"
###################### Process Setting ######################
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

    simulation_res = {
        "GBM": [(6.88532431, 6.89239491), 0.001767647865768763],
        "Heston": [(6.87612865, 6.88737754), 0.002812220820704026],
        "MJD": [(10.51947754, 10.53822222), 0.004686169577841212],
        "KJD": [(8.82124487, 8.83694563), 0.003925190664386232],
        "SVJ": [(10.51695755, 10.53436631), 0.004352189976865404],
        "SVCDJ": [(7.90055046, 8.32599804), 0.10636189376159098],
        "VG": [(6.88351487, 6.88696561), 0.0008626847804007355],
        "NIG": [(9.78201388, 9.80134857), 0.004833670440717833],
    }

    plot_N_Config = {
        "GBM": np.array(list(range(10, 300, 2))),
        "Heston": np.array(list(range(10, 300, 2))),
        "MJD": np.array(list(range(10, 150, 2))),
        "KJD": np.array(list(range(30, 250, 2))),
        "SVJ": np.array(list(range(10, 200, 2))),
        "SVCDJ": np.array(list(range(10, 400, 2))),
        "VG": np.array(list(range(10, 1600, 2))),
        "NIG": np.array(list(range(10, 250, 2))),
    }

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-6,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        # 取得資料
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

        # 畫圖
        plotValueWithCV(
            N_list,
            val_list,
            simulation_res[process_name][0],
            simulation_res[process_name][1],
            save_dir,
            "value-plot-" + process_name + "-call",
        )


def RightUp():
    # Polynomial Setting
    S0 = 90
    poly_coeff = [-20, -5, 0.05]
    positive_interval = [10 * (5 + sqrt(29)), inf]

    simulation_res = {
        "GBM": [(9.34389677, 9.38341305), 0.009879068455075636],
        "Heston": [(9.20373024, 9.23362933), 0.007474771448620839],
        "MJD": [(31.12915675, 31.25459844), 0.03136042368798125],
        "KJD": [(18.08147521, 18.16307429), 0.020399771149051726],
        "SVJ": [(31.06848677, 31.17168225), 0.025798871295148285],
        "SVCDJ": [(17.01361972, 20.44675647), 0.858284188134255],
        "VG": [(8.20882788, 8.22975426), 0.005231595821767995],
        "NIG": [(28.26422876, 28.37015655), 0.026481947424156264],
    }

    plot_N_Config = {
        "GBM": np.array(list(range(10, 300, 2))),
        "Heston": np.array(list(range(10, 300, 2))),
        "MJD": np.array(list(range(10, 150, 2))),
        "KJD": np.array(list(range(30, 250, 2))),
        "SVJ": np.array(list(range(10, 200, 2))),
        "SVCDJ": np.array(list(range(10, 400, 2))),
        "VG": np.array(list(range(10, 1600, 2))),
        "NIG": np.array(list(range(10, 250, 2))),
    }

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-6,
            assume_true_b=1e15,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        # 取得資料
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

        # 畫圖
        plotValueWithCV(
            N_list,
            val_list,
            simulation_res[process_name][0],
            simulation_res[process_name][1],
            save_dir,
            "value-plot-" + process_name + "-rightup",
        )


def LeftUp():
    # Polynomial Setting
    S0 = 110
    poly_coeff = [947.1, -30.164, 0.309, -0.001]
    positive_interval = [0, 77, 82, 150]

    simulation_res = {
        "GBM": [(31.98613631, 31.99895833), 0.0032055048228150666],
        "Heston": [(32.09652904, 32.11001631), 0.0033718155184589414],
        "MJD": [(23.98404402, 24.00864153), 0.006149378313814785],
        "KJD": [(31.45671736, 31.48970445), 0.008246771915001961],
        "SVJ": [(24.02250625, 24.04707388), 0.006141907401386668],
        "SVCDJ": [(17.01361972, 20.44675647), 0.858284188134255],
        "VG": [(33.29810714, 33.30723862), 0.0022828693506454765],
        "NIG": [(24.94119354, 24.96012273), 0.004732296527171264],
    }

    plot_N_Config = {
        "GBM": np.array(list(range(80, 300, 1))),
        "Heston": np.array(list(range(80, 300, 1))),
        "MJD": np.array(list(range(65, 150, 1))),
        "KJD": np.array(list(range(65, 250, 1))),
        "SVJ": np.array(list(range(50, 200, 1))),
        "SVCDJ": np.array(list(range(180, 400, 1))),
        "VG": np.array(list(range(85, 600, 1))),
        "NIG": np.array(list(range(60, 250, 1))),
    }

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-6,
            assume_true_b=1e15,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        # 取得資料
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

        # 畫圖
        plotValueWithCV(
            N_list,
            val_list,
            simulation_res[process_name][0],
            simulation_res[process_name][1],
            save_dir,
            "value-plot-" + process_name + "-leftup",
        )


def BothUp():
    # Polynomial Setting
    S0 = 15
    poly_coeff = [44.235, -39.474, 5.4793, -0.2358, 0.0031]
    positive_interval = [0, 1.363962, 10.620047, 25.599102, 38.481405, inf]

    simulation_res = {
        "GBM": [(43.01140318, 43.02118617), 0.0024457484026965],
        "Heston": [(43.04640828, 43.05537684), 0.00224214037816186],
        "MJD": [(36.04558741, 36.07268964), 0.0067755575164168405],
        "KJD": [(40.75094345, 40.79677287), 0.011457355398660385],
        "SVJ": [(36.06885455, 36.10053754), 0.007920746888845797],
        "SVCDJ": [(40.31902031, 41.53222199), 0.3033004190494669],
        "VG": [(43.27119119, 43.28096831), 0.002444280146383726],
        "NIG": [(37.16044676, 37.20863028), 0.012045881107484908],
    }

    plot_N_Config = {
        "GBM": np.array(list(range(100, 200, 1))),
        "Heston": np.array(list(range(100, 200, 1))),
        "MJD": np.array(list(range(40, 150, 1))),
        "KJD": np.array(list(range(180, 250, 1))),
        "SVJ": np.array(list(range(40, 200, 1))),
        "SVCDJ": np.array(list(range(180, 400, 1))),
        "VG": np.array(list(range(2000, 4100, 20))),
        "NIG": np.array(list(range(90, 250, 1))),
    }

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-6,
            assume_true_b=1e15,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        # 取得資料
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
        print(val_list[-1])
        # 畫圖
        plotValueWithCV(
            N_list,
            val_list,
            simulation_res[process_name][0],
            simulation_res[process_name][1],
            save_dir,
            "value-plot-" + process_name + "-bothup",
        )


def BothDown():
    # Polynomial Setting
    S0 = 30
    poly_coeff = [-44.235, 39.474, -5.4793, 0.2358, -0.0031]
    positive_interval = [1.363962, 10.620047, 25.599102, 38.481405]

    simulation_res = {
        "GBM": [(48.73786033, 48.77377146), 0.008977782334233924],
        "Heston": [(48.98373131, 49.02334141), 0.009902524789808803],
        "MJD": [(33.13781867, 33.17092059), 0.00827548160153181],
        "KJD": [(43.79356064, 43.84128478), 0.011931036614157658],
        "SVJ": [(33.17827943, 33.21633361), 0.00951354309609421],
        "SVCDJ": [(40.31902031, 41.53222199), 0.3033004190494669],
        "VG": [(51.98617472, 52.01223229), 0.006514392737709473],
        "NIG": [(35.32451354, 35.35572022), 0.007801668935689434],
    }

    plot_N_Config = {
        "GBM": np.array(list(range(10, 300, 2))),
        "Heston": np.array(list(range(10, 300, 2))),
        "MJD": np.array(list(range(10, 150, 2))),
        "KJD": np.array(list(range(30, 250, 2))),
        "SVJ": np.array(list(range(10, 200, 2))),
        "SVCDJ": np.array(list(range(10, 400, 2))),
        "VG": np.array(list(range(10, 1600, 2))),
        "NIG": np.array(list(range(10, 250, 2))),
    }

    for process_name in processes.keys():
        print(process_name)
        best_positive_interval = positive_interval.copy()
        densityRecover = DensityRecover(
            S0=S0,
            process_cf=processes[process_name],
            poly_coeff=poly_coeff,
            positive_interval=positive_interval,
            error_acceptance=1e-6,
            assume_true_b=1e15,
        )
        (
            lower_limit,
            upper_limit,
            best_positive_interval[0],
            best_positive_interval[-1],
        ) = densityRecover.getIntegralRangeAndInterval()

        # 取得資料
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

        # 畫圖
        plotValueWithCV(
            N_list,
            val_list,
            simulation_res[process_name][0],
            simulation_res[process_name][1],
            save_dir,
            "value-plot-" + process_name + "-bothdown",
        )


if __name__ == "__main__":
    print("Call")
    Call()
    print("Right Up")
    RightUp()
    # print("Left Up")
    # LeftUp()
    # print("Both Up")
    # BothUp()
    print("Both Down")
    BothDown()
