import numpy as np
from numpy import pi, sin, cos, log, exp
i = 1j

# 排查流程
# 1. 檢查cf是否有問題
# 2. 檢查積分區間的影響
# 3. 是否有必要進行St/ K 最後手段

class PolynomialByBinomialTree:
    def __init__(self, S0, T, r, sigma, poly_coef, N):
        self.S0 = S0
        self.x0 = log(S0)
        self.T = T
        self.r = r
        self.sigma = sigma
        self.poly_coef = np.array(poly_coef)  # store the coefficients of the polynomial


        # ============== Hyperparameter ==============
        self.N = N  # number of frequency for fitting density function

    def StockPrice(self, i, j, u, d):
        return self.S0 * (u ** (i - j) * d ** j)

    def Payoff(self, price):
        payoff = 0
        for power, coef in enumerate(self.poly_coef):
            payoff += coef * price ** power
        return max(payoff, 0)

    def getValue(self):
        delta_t = self.T / self.N
        u = exp(self.sigma * (delta_t ** 0.5))
        d = 1 / u
        p = (exp(self.r * delta_t) - d) / (u - d)
        payoff_arr = np.zeros(self.N + 1).T
        for j in range(self.N + 1):
            payoff_arr[j] = self.Payoff(self.StockPrice(self.N, j, u, d))

        for i in reversed(list(range(self.N))):
            for j in range(i + 1):
                payoff_arr[j] = (p * payoff_arr[j] + (1 - p) * payoff_arr[j + 1]) * exp(-self.r * delta_t)
        return payoff_arr[0]

if __name__ == "__main__":
    polynomialCosMethod = PolynomialByBinomialTree(S0=100, r=0.1, sigma=0.12, T=2, N=1000,
                                                   poly_coef=[-90, 1])

    print(polynomialCosMethod.getValue())
