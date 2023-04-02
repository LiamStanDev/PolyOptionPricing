from numpy import pi, sin, cos
from PolynomialPricingMethod.utils.CharacteristicFunc import *

i = 1j


class PolyByCosMethod:
    def __init__(
        self,
        S0,
        T,
        r,
        sigma,
        process_cf,
        poly_coeff,
        positive_interval,
        N,
        lower_limit,
        upper_limit,
    ):
        self.x0 = log(S0)
        self.T = T
        self.r = r
        # self.sigma = sigma
        self.process_cf = process_cf
        self.poly_coeff = np.array(poly_coeff)
        if positive_interval[0] == 0:
            positive_interval[0] = 1e-4
        self.positive_interval = np.log(positive_interval)

        # ============== Hyperparameter ==============
        self.N = N  # number of frequency for fitting density function
        self.lower_limit = log(lower_limit)
        self.upper_limit = log(upper_limit)

    # COS model
    def getValue(self):
        val = 0.0
        for k in range(self.N):
            if k == 0:
                val += (
                    0.5
                    * np.real(
                        self.process_cf.getCFValue(
                            k * pi / (self.upper_limit - self.lower_limit)
                        )
                        * exp(
                            i
                            * k
                            * pi
                            * (self.x0 - self.lower_limit)
                            / (self.upper_limit - self.lower_limit)
                        )
                    )
                    * self._V(k)
                )
            else:
                val += np.real(
                    self.process_cf.getCFValue(
                        k * pi / (self.upper_limit - self.lower_limit)
                    )
                    * exp(
                        i
                        * k
                        * pi
                        * (self.x0 - self.lower_limit)
                        / (self.upper_limit - self.lower_limit)
                    )
                ) * self._V(k)
        return exp(-self.r * self.T) * val

    def _Chi(self, k, kd_down, kd_up):
        n = len(self.poly_coeff)

        def sin_integral(x, v):
            return sin(
                k * pi * (x - self.lower_limit) / (self.upper_limit - self.lower_limit)
            ) * exp(v * x)

        def cos_integral(x, v):
            return cos(
                k * pi * (x - self.lower_limit) / (self.upper_limit - self.lower_limit)
            ) * exp(v * x)

        # 計算polynomial的每一項
        chi = 0
        for j in range(1, n):
            aj = self.poly_coeff[j]
            chi += (
                aj
                * (
                    1
                    / (1 + (k * pi / (j * (self.upper_limit - self.lower_limit))) ** 2)
                )
                * (
                    1 / j * (cos_integral(kd_up, j) - cos_integral(kd_down, j))
                    + k
                    * pi
                    / (j**2 * (self.upper_limit - self.lower_limit))
                    * (sin_integral(kd_up, j) - sin_integral(kd_down, j))
                )
            )
        return chi

    # K的影響
    def _Posi(self, k, kd_down, kd_up):
        a0 = self.poly_coeff[0]

        def sin_integral(x):
            return sin(
                k * pi * (x - self.lower_limit) / (self.upper_limit - self.lower_limit)
            )

        if k == 0:
            return a0 * (kd_up - kd_down)
        else:
            return (
                a0
                * (self.upper_limit - self.lower_limit)
                / (k * pi)
                * (sin_integral(kd_up) - sin_integral(kd_down))
            )

    # 計算不同頻率下的值
    def _V(self, k):
        v = 0.0
        for d in range(int(len(self.positive_interval) / 2)):
            kd_down = self.positive_interval[2 * d]
            kd_up = self.positive_interval[2 * d + 1]
            v += self._Chi(k, kd_down, kd_up) + self._Posi(k, kd_down, kd_up)
        return 2 / (self.upper_limit - self.lower_limit) * v
