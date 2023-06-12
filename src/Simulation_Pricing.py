import threading

# 導入自己寫的蒙地卡羅
from PolynomialPricingMethod.PathSimulationMethod import (
    GBMByMC,
    HestonByMC,
    MJDByMC,
    SVJByMC,
    # SVJJByMC,
    VGByMC,
    NIGByMC,
    KJDByMC,
)

# 設定蒙地卡羅次數
n = 252  # for setting dt
N = int(1e5)
N_repeat = 20
save_directory = (
    "/Users/lindazhong/Documents/Code/Projects/PolyOptionPricing/Data/Simulation"
)

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


def Call():
    # Polynomial Setting
    S0 = 100
    poly_coeff = [-100, 1]
    processes = {
        "GBM": GBMByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "Heston": HestonByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "MJD": MJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "KJD": KJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity_kdj,
            p=p,
            eta1=eta1,
            eta2=eta2,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "SVJ": SVJByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "VG": VGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            gamma_mean=gamma_mean,
            gamma_var=gamma_var,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "NIG": NIGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            delta=delta,
            alpha=alpha,
            beta=beta,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
    }

    for process_name in processes.keys():
        print(
            processes[process_name].getStatistic(
                save_data=True, save_dir=save_directory, file_name="Call.txt"
            )
        )


def RightUp():
    # Polynomial Setting
    S0 = 90
    poly_coeff = [-20, -5, 0.05]
    processes = {
        "GBM": GBMByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "Heston": HestonByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "MJD": MJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "KJD": KJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity_kdj,
            p=p,
            eta1=eta1,
            eta2=eta2,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "SVJ": SVJByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        # "SVCDJ": SVJJByMC(
        #     S0=S0,
        #     r=r,
        #     T=T,
        #     Y=SVCDJ_Y,
        #     Y_bar=SVCDJ_Y_bar,
        #     jump_intensity=SVCDJ_intensity,
        #     mu0=SVCDJ_mu0,
        #     mu_xy=SVCDJ_mu_xy,
        #     sigma_y=SVCDJ_sigma_y,
        #     sigma_xy=SVCDJ_sigma_xy,
        #     corr=SVCDJ_corr,
        #     theta_y=SVCDJ_theta_y,
        #     k_y=SVCDJ_k_y,
        #     poly_coeff=poly_coeff,
        #     N_line=N,
        #     n=n,
        #     N_repeat=N_repeat,
        # ),
        "VG": VGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            gamma_mean=gamma_mean,
            gamma_var=gamma_var,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "NIG": NIGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            delta=delta,
            alpha=alpha,
            beta=beta,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
    }
    for process_name in processes.keys():
        print(
            processes[process_name].getStatistic(
                save_data=True, save_dir=save_directory, file_name="RightUp.txt"
            )
        )


def LeftUp():
    # Polynomial Setting
    S0 = 110
    poly_coeff = [947.1, -30.164, 0.309, -0.001]
    processes = {
        "GBM": GBMByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "Heston": HestonByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "MJD": MJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "KJD": KJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity_kdj,
            p=p,
            eta1=eta1,
            eta2=eta2,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "SVJ": SVJByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        # "SVCDJ": SVJJByMC(
        #     S0=S0,
        #     r=r,
        #     T=T,
        #     Y=SVCDJ_Y,
        #     Y_bar=SVCDJ_Y_bar,
        #     jump_intensity=SVCDJ_intensity,
        #     mu0=SVCDJ_mu0,
        #     mu_xy=SVCDJ_mu_xy,
        #     sigma_y=SVCDJ_sigma_y,
        #     sigma_xy=SVCDJ_sigma_xy,
        #     corr=SVCDJ_corr,
        #     theta_y=SVCDJ_theta_y,
        #     k_y=SVCDJ_k_y,
        #     poly_coeff=poly_coeff,
        #     N_line=N,
        #     n=n,
        #     N_repeat=N_repeat,
        # ),
        "VG": VGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            gamma_mean=gamma_mean,
            gamma_var=gamma_var,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "NIG": NIGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            delta=delta,
            alpha=alpha,
            beta=beta,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
    }

    for process_name in processes.keys():
        print(
            processes[process_name].getStatistic(
                save_data=True, save_dir=save_directory, file_name="LeftUp.txt"
            )
        )


def BothUp():
    # Polynomial Setting
    S0 = 15
    poly_coeff = [44.235, -39.474, 5.4793, -0.2358, 0.0031]
    processes = {
        "GBM": GBMByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "Heston": HestonByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "MJD": MJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "KJD": KJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity_kdj,
            p=p,
            eta1=eta1,
            eta2=eta2,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "SVJ": SVJByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        # "SVCDJ": SVJJByMC(
        #     S0=S0,
        #     r=r,
        #     T=T,
        #     Y=SVCDJ_Y,
        #     Y_bar=SVCDJ_Y_bar,
        #     jump_intensity=SVCDJ_intensity,
        #     mu0=SVCDJ_mu0,
        #     mu_xy=SVCDJ_mu_xy,
        #     sigma_y=SVCDJ_sigma_y,
        #     sigma_xy=SVCDJ_sigma_xy,
        #     corr=SVCDJ_corr,
        #     theta_y=SVCDJ_theta_y,
        #     k_y=SVCDJ_k_y,
        #     poly_coeff=poly_coeff,
        #     N_line=N,
        #     n=n,
        #     N_repeat=N_repeat,
        # ),
        "VG": VGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            gamma_mean=gamma_mean,
            gamma_var=gamma_var,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "NIG": NIGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            delta=delta,
            alpha=alpha,
            beta=beta,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
    }

    for process_name in processes.keys():
        print(
            processes[process_name].getStatistic(
                save_data=True, save_dir=save_directory, file_name="BothUp.txt"
            )
        )


def BothDown():
    # Polynomial Setting
    S0 = 30
    poly_coeff = [-44.235, 39.474, -5.4793, 0.2358, -0.0031]
    processes = {
        "GBM": GBMByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "Heston": HestonByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "MJD": MJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "KJD": KJDByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity_kdj,
            p=p,
            eta1=eta1,
            eta2=eta2,
            N_line=N,
            N_repeat=N_repeat,
            n=n,
        ),
        "SVJ": SVJByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            mean_reversion_speed=mean_reversion_speed,
            long_term_var_mean=long_term_var_mean,
            corr=corr,
            std_of_var_process=std_of_var_process,
            poly_coeff=poly_coeff,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_var=jump_var,
            n=n,
            N_line=N,
            N_repeat=N_repeat,
        ),
        # "SVCDJ": SVJJByMC(
        #     S0=S0,
        #     r=r,
        #     T=T,
        #     Y=SVCDJ_Y,
        #     Y_bar=SVCDJ_Y_bar,
        #     jump_intensity=SVCDJ_intensity,
        #     mu0=SVCDJ_mu0,
        #     mu_xy=SVCDJ_mu_xy,
        #     sigma_y=SVCDJ_sigma_y,
        #     sigma_xy=SVCDJ_sigma_xy,
        #     corr=SVCDJ_corr,
        #     theta_y=SVCDJ_theta_y,
        #     k_y=SVCDJ_k_y,
        #     poly_coeff=poly_coeff,
        #     N_line=N,
        #     n=n,
        #     N_repeat=N_repeat,
        # ),
        "VG": VGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            gamma_mean=gamma_mean,
            gamma_var=gamma_var,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
        "NIG": NIGByMC(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            delta=delta,
            alpha=alpha,
            beta=beta,
            poly_coeff=poly_coeff,
            N_line=N,
            N_repeat=N_repeat,
        ),
    }

    for process_name in processes.keys():
        print(
            processes[process_name].getStatistic(
                save_data=True, save_dir=save_directory, file_name="BothDown.txt"
            )
        )


if __name__ == "__main__":
    thread_Call = threading.Thread(target=Call)
    thread_RightUp = threading.Thread(target=RightUp)
    # thread_LeftUp = threading.Thread(target=LeftUp)
    # thread_BothUp = threading.Thread(target=BothUp)
    thread_BothDown = threading.Thread(target=BothDown)

    thread_Call.start()
    time.sleep(1)
    thread_RightUp.start()
    time.sleep(1)
    # thread_LeftUp.start()
    # thread_Call.join()
    # thread_RightUp.join()
    # thread_LeftUp.join()
    #
    # thread_BothUp.start()
    # time.sleep(1)
    thread_BothDown.start()
