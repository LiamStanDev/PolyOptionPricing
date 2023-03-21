import numpy as np
from numpy import log, exp

i = 1j


class PolyByTree:
    def __init__(self, S0, T, r, sigma, poly_coeff, N):
        self.S0 = S0
        self.x0 = log(S0)
        self.T = T
        self.r = r
        self.sigma = sigma
        self.poly_coeff = np.array(poly_coeff)  # store the coefficients of the polynomial

        # ============== Hyperparameter ==============
        self.N = N  # number of frequency for fitting density function

    def _StockPrice(self, i, j, u, d):
        return self.S0 * (u ** (i - j) * d ** j)

    def _Payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price ** power
        return max(payoff, 0)

    def getValue(self):
        delta_t = self.T / self.N
        u = exp(self.sigma * (delta_t ** 0.5))
        d = 1 / u
        p = (exp(self.r * delta_t) - d) / (u - d)
        payoff_arr = np.zeros(self.N + 1).T
        for j in range(self.N + 1):
            payoff_arr[j] = self._Payoff(self._StockPrice(self.N, j, u, d))

        for k in reversed(list(range(self.N))):
            for j in range(k + 1):
                payoff_arr[j] = (p * payoff_arr[j] + (1 - p) * payoff_arr[j + 1]) * exp(-self.r * delta_t)
        return payoff_arr[0]
