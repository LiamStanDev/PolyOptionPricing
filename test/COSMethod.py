import sys

sys.path.append("/Users/lindazhong/Documents/Code/Projects/PolyOptionPricing")
from math import inf
from PolynomialPricingMethod.COSMethod import PolyByCosMethod
from PolynomialPricingMethod.utils.CharacteristicFunc import Heston
from PolynomialPricingMethod.utils.DensityTools import DensityRecover
from PolynomialPricingMethod.utils.Tools import timeit

# Parameter Setting (Basic).
S0 = 15
T = 1
r = 0.1
sigma = 0.3
# Parameter Setting (Specific Process).
std_of_var_process = 0.1
mean_reversion_speed = 3
long_term_var_mean = 0.06
corr = -0.1
# Payoff Function Setting.
poly_coeff = [44.235, -39.474, 5.4793, -0.2358, 0.0031]
positive_interval = [
    0,
    1.363962,
    10.620047,
    25.599102,
    38.481405,
    inf,
]  # positive region between roots and boundaries
# Create Characteristic Function.
process_cf = Heston(
    r=r,
    sigma=sigma,
    T=T,
    mean_reversion_speed=mean_reversion_speed,
    long_term_var_mean=long_term_var_mean,
    corr=corr,
    std_of_var_process=std_of_var_process,
)
# Using density tools to automatically find the best integration region
best_positive_interval = positive_interval.copy()
(
    lower_limit,
    upper_limit,
    best_positive_interval[0],
    best_positive_interval[-1],
) = DensityRecover(
    S0=S0,
    process_cf=process_cf,
    poly_coeff=poly_coeff,
    positive_interval=positive_interval,
).getIntegralRangeAndInterval()

for N in [64, 128, 256, 512, 1024]:
    polynomial = PolyByCosMethod(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        process_cf=process_cf,
        poly_coeff=poly_coeff,
        positive_interval=best_positive_interval,
        N=N,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    # print the total spending time
    result, spending_time = timeit(polynomial.getValue)
    print(f"N={N}: value: {result:.6f}, CUP Time: {spending_time * 1000:.1f}ms.")


# r = 0.0
# T = 0.25
# sigma = 0.2
#
# SVCDJ_Y = 0.0968
# SVCDJ_intensity = 1.64
# SVCDJ_mu0 = -0.03
# SVCDJ_mu_xy = -7.87
# SVCDJ_sigma_xy = 0.22
# SVCDJ_sigma_y = 0.61
# SVCDJ_corr = -0.1
# SVCDJ_Y_bar = 0.49
# SVCDJ_theta_y = 0.0036
# SVCDJ_k_y = 5.06
#
# S0 = 100
# poly_coeff = [-100, 1]
# positive_interval = [100, inf]
# best_positive_interval = positive_interval.copy()
# cf = SVCDJ(r=r, T=T, Y=SVCDJ_Y, intensity=SVCDJ_intensity,mu0=SVCDJ_mu0, mu_xy=SVCDJ_mu_xy, sigma_xy=SVCDJ_sigma_xy, sigma_y=SVCDJ_sigma_y, corr=SVCDJ_corr, Y_bar=SVCDJ_Y_bar, theta_y=SVCDJ_theta_y, k_y=SVCDJ_k_y)
# densityRecover = DensityRecover(S0=S0, process_cf=cf, poly_coeff=poly_coeff, positive_interval=positive_interval, error_acceptance=1e-15)
# lower_limit, upper_limit, best_positive_interval[0], best_positive_interval[-1] = densityRecover.getIntegralRangeAndInterval()
#
# ref_val = PolyByCosMethod(S0=S0, T=T, r=r, sigma=sigma, process_cf=cf, poly_coeff=poly_coeff,
#                                   positive_interval=best_positive_interval, N=10000, lower_limit=lower_limit,
#                                   upper_limit=upper_limit).getValue()
# print(ref_val)
