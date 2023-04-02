import sys

sys.path.append("/Users/lindazhong/Documents/Code/Projects/PolyOptionPricing")
from PolynomialPricingMethod.PathSimulationMethod import GBMByMC, HestonByMC

# Simulation Setting
n = 252  # for setting dt
N = int(1e5)
N_repeat = 20
save_directory = "/Data/Saving/Path"

###################### Process Setting ######################
# Parameter Setting (Basic)
S0 = 15
r = 0.05
T = 0.5
sigma = 0.2
# Parameter Setting (Specific Process)
std_of_var_process = 0.1
mean_reversion_speed = 3
long_term_var_mean = 0.04
corr = -0.1

# Payoff Function Setting
poly_coeff = [44.235, -39.474, 5.4793, -0.2358, 0.0031]

# Create Characteristic Functions
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
}

for process_name in processes.keys():
    print(
        processes[process_name].getStatistic(
            save_data=True, save_dir=save_directory + process_name + ".txt"
        )
    )
