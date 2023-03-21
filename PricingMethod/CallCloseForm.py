import numpy as np
from numpy import log, sqrt, exp
from scipy import stats
import warnings
import math
import scipy
warnings.filterwarnings('ignore')

class BSMCloseForm:
    def __init__(self, S0, T, r, sigma, K):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K = K
    def getValue(self):
        d1 = (log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) \
            / (self.sigma * sqrt(self.T))
        d2 = (log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) \
            / (self.sigma * sqrt(self.T))
        BS_C = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0) -
                self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return BS_C


def factorial(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    return n * factorial(n - 1)


class MertonCloseForm:
    def __init__(self, S0, T, r, sigma, K, jump_intensity, jump_mean, jump_var):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K = K
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_var = jump_var

    def getValue(self, iterator_time):
        k = exp(self.jump_mean + 0.5 * self.jump_var) - 1
        var_n = lambda n: self.sigma ** 2 + n * self.jump_var / self.T
        r_n = lambda n: self.r + n * (self.jump_mean + 0.5 * self.jump_var) / self.T - self.jump_intensity * k
        lambda_start = self.jump_intensity * (k + 1)
        d1_n = lambda n: (log(self.S0 / self.K) + (r_n(n) + var_n(n) / 2) * self.T) / (sqrt(var_n(n) * self.T))
        d2_n = lambda n: (log(self.S0 / self.K) + (r_n(n) - var_n(n) / 2) * self.T) / (sqrt(var_n(n) * self.T))
        Merton_C = 0
        for i in range(iterator_time):
            Merton_C += (self.S0 * stats.norm.cdf(d1_n(i), 0.0, 1.0)) * stats.poisson.pmf(i, lambda_start * self.T) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2_n(i), 0.0, 1.0) * stats.poisson.pmf(i, self.jump_intensity * self.T)

        return Merton_C


if __name__ == "__main__":
    BS_call = BSMCloseForm(S0=100, r=0.05, sigma=0.2, T=0.5, K=100)
    print(f"{BS_call.getValue():.50f}")

    Merton_call = MertonCloseForm(S0=100, T=0.5, r=0.05, sigma=0.2, K=100, jump_intensity=140, jump_mean=0.01, jump_var=0.02 ** 2)
    print(f"{Merton_call.getValue(5000):.50f}")
    # polynomialCosMethod = PolynomialCosMethod(S0=100, r=0.1, sigma=0.0175, T=1, process=process_GBM,
    #                          N=100000, lower_limit=0.001, upper_limit=2000,
    #                          poly_coef=[-90, 1], positive_interval=[90, 2000])
    # r = 0.05
    # T = 0.5
    # sigma = 0.2
    # # Jump process
    # jump_intensity = 140
    # jump_mean = 0.01
    # jump_var = 0.02 ** 2
