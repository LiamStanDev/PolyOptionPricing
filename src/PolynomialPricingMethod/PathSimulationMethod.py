import numpy as np
from numpy import log, exp, sqrt, power
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


class GBMByMC:
    """
    Geometric Brownian motion
    """

    def __init__(
        self, S0, r, sigma, T, poly_coeff, N_line=int(1e6), n=252, N_repeat=20
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

        self.poly_coeff = poly_coeff

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    # 選擇權的payoff計算

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    # 計算一次蒙地卡羅的價格
    def getValue(self, repeat):
        dt = self.T / self.n
        # 生成S0的array
        lnSt = np.full(self.N_line, log(self.S0))
        # 每次dt的變化
        for i in range(self.n):
            if i % 20 == 0:
                print(f"GBM: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv = np.random.randn(int(self.N_line / 2))
            norm_rv = np.append(norm_rv, -norm_rv)
            norm_rv = norm_rv / np.std(norm_rv)

            dW = norm_rv * pow(dt, 0.5)
            # 計算dS
            dlnSt = (self.r - pow(self.sigma, 2) / 2) * dt + self.sigma * dW

            lnSt = lnSt + dlnSt

        ST = exp(lnSt)

        # 將所有股價利用_payoff方法mapping成payoff，利用mapping會更快
        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    # 進行多次蒙地卡羅:為使用者調用的方法，回傳平均值與標準差
    # save_data如果為True則會建立文件存放資料
    def getStatistic(self, file_name="GBM", save_data=False, save_dir=""):
        values = []
        print("GBM simulation starting...")
        for i in range(self.N_repeat):
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: GBM\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )

                file.write(s)
        return mean, std


class HestonByMC:
    """
    Heston model
    """

    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        mean_reversion_speed,
        long_term_var_mean,
        corr,
        std_of_var_process,
        poly_coeff,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.mean_reversion_speed = mean_reversion_speed
        self.long_term_var_mean = long_term_var_mean
        self.corr = corr
        self.std_of_var_process = std_of_var_process

        self.poly_coeff = poly_coeff

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def getValue(self, repeat):
        dt = self.T / self.n

        lnSt = np.full(self.N_line, log(self.S0))
        Vt = np.full(self.N_line, pow(self.sigma, 2))

        for i in range(self.n):
            if i % 20 == 0:
                print(f"Heston: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv1 = np.random.randn(int(self.N_line / 2))
            norm_rv1 = np.append(norm_rv1, -norm_rv1)
            norm_rv1 = norm_rv1 / np.std(norm_rv1)

            norm_rv2 = np.random.randn(int(self.N_line / 2))
            norm_rv2 = np.append(norm_rv2, -norm_rv2)
            norm_rv2 = norm_rv2 / np.std(norm_rv2)

            dW1t = norm_rv1 * pow(dt, 0.5)
            dW2t = pow(dt, 0.5) * (
                norm_rv1 * self.corr + norm_rv2 * pow(1 - pow(self.corr, 2), 0.5)
            )

            dlnSt = (self.r - 0.5 * Vt) * dt + sqrt(Vt) * dW1t
            lnSt = lnSt + dlnSt
            Vt = Vt + (
                self.mean_reversion_speed * (self.long_term_var_mean - Vt) * dt
                + self.std_of_var_process * sqrt(Vt) * dW2t
            )
            Vt = np.where(Vt > 0, Vt, 0)

        ST = exp(lnSt)

        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="Heston"):
        print("Heston simulation starting...")
        values = []
        for i in range(self.N_repeat):
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: Heston\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"\tmean_reversion: {self.mean_reversion_speed}, long_term_var_mean: {self.long_term_var_mean}, corr: {self.corr}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )

                file.write(s)
        return mean, std


class MJDByMC:
    """
    Merton Jump-Diffusion model
    """

    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        poly_coeff,
        jump_intensity,
        jump_mean,
        jump_var,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.poly_coeff = poly_coeff
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_var = jump_var

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def getValue(self, repeat):
        dt = self.T / self.n

        lnSt = np.full(self.N_line, log(self.S0))

        k = exp(self.jump_mean + 0.5 * self.jump_var) - 1  # k = E(Y - 1)
        mu = self.r - self.jump_intensity * k  # risk neutral adjust rate

        for i in range(self.n):
            if i % 20 == 0:
                print(f"MDJ: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv1 = np.random.randn(int(self.N_line / 2))
            norm_rv1 = np.append(norm_rv1, -norm_rv1)
            norm_rv1 = norm_rv1 / np.std(norm_rv1)

            norm_rv2 = np.random.randn(int(self.N_line / 2))
            norm_rv2 = np.append(norm_rv2, -norm_rv2)
            norm_rv2 = norm_rv2 / np.std(norm_rv2)

            dWt = norm_rv1 * pow(dt, 0.5)
            lnYt = norm_rv2 * sqrt(self.jump_var) + self.jump_mean

            jumps = np.random.poisson(self.jump_intensity * dt, self.N_line)
            # deal with jump > 1
            over_jump_ind = np.squeeze(np.argwhere(jumps > 1), axis=1)
            J = lnYt * jumps
            if over_jump_ind.size > 0:
                for j in over_jump_ind:
                    J[j] = np.sum(
                        np.random.normal(
                            self.jump_mean, sqrt(self.jump_var), size=jumps[j]
                        )
                    )

            dlnSt = (mu - 0.5 * pow(self.sigma, 2)) * dt + self.sigma * dWt + J
            lnSt = lnSt + dlnSt

        ST = exp(lnSt)

        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="MJD"):
        values = []
        print("MJD simulation starting...")
        for i in range(self.N_repeat):
            values.append(self.getValue(i))
        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: MJD\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"\tjump_intensity: {self.jump_intensity}, jump_mean: {self.jump_mean}, jump_var: {self.jump_var}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )

                file.write(s)
        return mean, std


class KJDByMC:
    """
    Kou's Double Exponential model
    """

    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        poly_coeff,
        jump_intensity,
        p,
        eta1,
        eta2,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.poly_coeff = poly_coeff
        self.jump_intensity = jump_intensity
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def getValue(self, repeat):
        dt = self.T / self.n

        lnSt = np.full(self.N_line, log(self.S0))

        k = (
            self.p * self.eta1 / (self.eta1 - 1)
            + (1 - self.p) * self.eta2 / (self.eta2 + 1)
            - 1
        )  # k = E(Y - 1)
        mu = self.r - self.jump_intensity * k  # risk neutral adjust rate

        for i in range(self.n):
            if i % 20 == 0:
                print(f"KJD: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv = np.random.randn(int(self.N_line / 2))
            norm_rv = np.append(norm_rv, -norm_rv)
            norm_rv = norm_rv / np.std(norm_rv)
            dWt = norm_rv * pow(dt, 0.5)

            exp_rv1 = np.random.exponential(1 / self.eta1, size=self.N_line)
            exp_rv2 = np.random.exponential(1 / self.eta2, size=self.N_line)
            up_rv = np.random.binomial(n=1, p=self.p, size=self.N_line)
            lnYt = up_rv * exp_rv1 + (1 - up_rv) * (-exp_rv2)

            jumps = np.random.poisson(self.jump_intensity * dt, size=self.N_line)
            # deal with jump > 1
            over_jump_ind = np.squeeze(np.argwhere(jumps > 1), axis=1)
            J = lnYt * jumps
            if over_jump_ind.size > 0:
                for j in over_jump_ind:
                    J[j] = 0
                    for k in range(jumps[j]):
                        exp_rv_over_jump = np.random.exponential(1 / self.eta1, size=2)
                        up_rv_over_jump = np.random.binomial(n=1, p=self.p, size=1)
                        lnYt_over_jump = up_rv_over_jump * exp_rv_over_jump[0] + (
                            1 - up_rv_over_jump
                        ) * (-exp_rv_over_jump[1])
                        J[j] += lnYt_over_jump

            dlnSt = (mu - 0.5 * pow(self.sigma, 2)) * dt + self.sigma * dWt + J
            lnSt = lnSt + dlnSt

        ST = exp(lnSt)

        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="KJD"):
        values = []
        print("KJD simulation starting...")
        for i in range(self.N_repeat):
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: KJD\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"\tjump_intensity: {self.jump_intensity}, p: {self.p}, eta1: {self.eta1}, eta2: {self.eta2}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )

                file.write(s)
        return mean, std


class SVJByMC:
    """
    Bate's Stochastic Volatility Jump model
    """

    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        mean_reversion_speed,
        long_term_var_mean,
        corr,
        std_of_var_process,
        poly_coeff,
        jump_intensity,
        jump_mean,
        jump_var,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.mean_reversion_speed = mean_reversion_speed
        self.long_term_var_mean = long_term_var_mean
        self.corr = corr
        self.std_of_var_process = std_of_var_process
        self.poly_coeff = poly_coeff
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_var = jump_var

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def getValue(self, repeat):
        dt = self.T / self.n

        lnSt = np.full(self.N_line, log(self.S0))
        Vt = np.full(self.N_line, pow(self.sigma, 2))

        k = exp(self.jump_mean + 0.5 * self.jump_var) - 1  # k = E(Y - 1)
        mu = self.r - self.jump_intensity * k  # risk neutral adjust rate

        for i in range(0, self.n):
            if i % 20 == 0:
                print(f"SVJ: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv1 = np.random.randn(int(self.N_line / 2))
            norm_rv1 = np.append(norm_rv1, -norm_rv1)
            norm_rv1 = norm_rv1 / np.std(norm_rv1)

            norm_rv2 = np.random.randn(int(self.N_line / 2))
            norm_rv2 = np.append(norm_rv2, -norm_rv2)
            norm_rv2 = norm_rv2 / np.std(norm_rv2)

            norm_rv3 = np.random.randn(int(self.N_line / 2))
            norm_rv3 = np.append(norm_rv3, -norm_rv3)
            norm_rv3 = norm_rv3 / np.std(norm_rv3)

            dW1t = norm_rv1 * pow(dt, 0.5)
            dW2t = pow(dt, 0.5) * (
                norm_rv1 * self.corr + norm_rv2 * pow(1 - pow(self.corr, 2), 0.5)
            )
            lnYt = norm_rv3 * sqrt(self.jump_var) + self.jump_mean

            jumps = np.random.poisson(self.jump_intensity * dt, self.N_line)
            # deal with jump > 1
            over_jump_ind = np.squeeze(np.argwhere(jumps > 1), axis=1)
            J = lnYt * jumps
            if over_jump_ind.size > 0:
                for j in over_jump_ind:
                    J[j] = np.sum(
                        np.random.normal(
                            self.jump_mean, sqrt(self.jump_var), size=jumps[j]
                        )
                    )

            dlnSt = (mu - 0.5 * Vt) * dt + sqrt(Vt) * dW1t + J
            lnSt = lnSt + dlnSt

            Vt = Vt + (
                self.mean_reversion_speed * (self.long_term_var_mean - Vt) * dt
                + self.std_of_var_process * sqrt(Vt) * dW2t
            )
            Vt = np.where(Vt > 0, Vt, 0)

        ST = exp(lnSt)

        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="SVJ"):
        values = []
        print("SVJ simulation starting...")
        for i in range(self.N_repeat):
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: SVJ\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"\tmean_reversion: {self.mean_reversion_speed}, long_term_var_mean: {self.long_term_var_mean}, corr: {self.corr}\n"
                    f"\tjump_intensity: {self.jump_intensity}, jump_mean: {self.jump_mean}, jump_var: {self.jump_var}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )
                file.write(s)
        return mean, std


class SVJJByMC:
    """
    Stochastic Volatility with Correlated Double Jump
    """

    def __init__(
        self,
        S0,
        r,
        d,
        T,
        sigma,
        intensity,
        sigma_v,
        corr,
        k_v,
        v_bar,
        mu_v,
        mu_y,
        sigma_y,
        corr_J,
        poly_coeff,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.d = d
        self.T = T
        self.v0 = sigma**2
        self.intensity = intensity
        self.sigma_v = sigma_v
        self.corr = corr
        self.k_v = k_v
        self.v_bar = v_bar
        self.mu_v = mu_v
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        self.corr_J = corr_J

        self.poly_coeff = poly_coeff

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def getValue(self, repeat):
        dt = self.T / self.n

        Yt = np.full(self.N_line, log(self.S0))
        Vt = np.full(self.N_line, self.v0)

        def theta(c1, c2):
            return exp(self.mu_y * c1 + 0.5 * power(self.sigma_y * c1, 2)) / (
                1 - self.mu_v * c2 - self.corr_J * self.mu_v * c1
            )

        mu_bar = theta(1, 0) - 1

        for i in range(0, self.n):
            if i % 20 == 0:
                print(f"SVJJ: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv1 = np.random.randn(int(self.N_line / 2))
            norm_rv1 = np.append(norm_rv1, -norm_rv1)
            norm_rv1 = norm_rv1 / np.std(norm_rv1)

            norm_rv2 = np.random.randn(int(self.N_line / 2))
            norm_rv2 = np.append(norm_rv2, -norm_rv2)
            norm_rv2 = norm_rv2 / np.std(norm_rv2)

            norm_rv3 = np.random.randn(int(self.N_line / 2))
            norm_rv3 = np.append(norm_rv3, -norm_rv3)
            norm_rv3 = norm_rv3 / np.std(norm_rv3)

            # Cov(dW1t, dW2t)
            dW1t = norm_rv1 * pow(dt, 0.5)
            dW2t = norm_rv2 * pow(dt, 0.5)

            # variance jump level
            z_v = np.random.exponential(self.mu_v, self.N_line)
            # return jump level
            z_y = norm_rv3 * self.sigma_y + self.mu_y + self.corr_J * z_v

            jumps = np.random.poisson(self.intensity * dt, self.N_line)

            dJ_Yt = z_y * jumps
            dJ_Vt = z_v * jumps
            over_jump_ind = np.squeeze(np.argwhere(jumps > 1), axis=1)
            # Deal with over-jump situation
            if over_jump_ind.size > 0:
                for j in over_jump_ind:
                    over_jump_z_v = np.random.exponential(self.mu_v, jumps[j])
                    dJ_Vt[j] = np.sum(over_jump_z_v)
                    dJ_Yt[j] = np.sum(
                        list(
                            map(
                                lambda z: np.random.normal(
                                    self.mu_y + self.corr_J * z, self.sigma_y
                                ),
                                over_jump_z_v,
                            )
                        )
                    )

            Yt = (
                Yt
                + (self.r - self.d - self.intensity * mu_bar - 0.5 * Vt) * dt
                + sqrt(Vt) * dW1t
                + dJ_Yt
            )
            Vt = (
                Vt
                + self.k_v * (self.v_bar - Vt) * dt
                + sqrt(Vt)
                * (
                    self.corr * self.sigma_v * dW1t
                    + sqrt(1 - self.corr**2) * self.sigma_v * dW2t
                )
                + dJ_Vt
            )
            Vt = np.where(Vt > 0, Vt, 0)

        ST = exp(Yt)
        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="SVJJ"):
        values = []
        print("SVJJ simulation starting...")
        for i in range(self.N_repeat):
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: SVJJ\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )

                file.write(s)
        return mean, std


class SVCDJByMC:
    """
    Stochastic Volatility with Correlated Double Jump(Guo 2009)
    """

    def __init__(
        self,
        S0,
        r,
        d,
        T,
        sigma,
        intensity,
        corr,
        sigma_y,
        poly_coeff,
        k_y,
        theta_y,
        mu_xy,
        Y_bar,
        mu_0,
        sigma_xy,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.d = d
        self.T = T
        self.Y_0 = sigma**2
        self.sigma_y = sigma_y
        self.corr = corr
        self.k_y = k_y
        self.theta_y = theta_y
        self.mu_xy = mu_xy
        self.intensity = intensity
        self.Y_bar = Y_bar
        self.mu_0 = mu_0
        self.sigma_xy = sigma_xy

        self.poly_coeff = poly_coeff

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def getValue(self, repeat):
        dt = self.T / self.n

        Yt = np.full(self.N_line, log(self.S0))
        Vt = np.full(self.N_line, self.Y_0)

        k = (
            exp(self.mu_0 + 0.5 * self.sigma_xy**2) / (1 - self.theta_y * self.mu_xy)
            - 1
        )

        for i in range(0, self.n):
            if i % 20 == 0:
                print(f"SVCDJ: {repeat + 1} round, {((i + 1) / self.n) * 100:.1f}%")

            # antithetic + moment match
            norm_rv1 = np.random.randn(int(self.N_line / 2))
            norm_rv1 = np.append(norm_rv1, -norm_rv1)
            norm_rv1 = norm_rv1 / np.std(norm_rv1)

            norm_rv2 = np.random.randn(int(self.N_line / 2))
            norm_rv2 = np.append(norm_rv2, -norm_rv2)
            norm_rv2 = norm_rv2 / np.std(norm_rv2)

            norm_rv3 = np.random.randn(int(self.N_line / 2))
            norm_rv3 = np.append(norm_rv3, -norm_rv3)
            norm_rv3 = norm_rv3 / np.std(norm_rv3)

            # Cov(dW1t, dW2t)
            dW1t = norm_rv1 * pow(dt, 0.5)
            dW2t = norm_rv2 * pow(dt, 0.5)

            # variance jump level
            z_v = np.random.exponential(self.theta_y, self.N_line)
            # return jump level
            z_y = norm_rv3 * self.sigma_xy + self.mu_0 + self.mu_xy * z_v

            jumps = np.random.poisson(self.intensity * dt, self.N_line)

            dJ_Yt = z_y * jumps
            dJ_Vt = z_v * jumps
            over_jump_ind = np.squeeze(np.argwhere(jumps > 1), axis=1)
            # Deal with over-jump situation
            if over_jump_ind.size > 0:
                for j in over_jump_ind:
                    over_jump_z_v = np.random.exponential(self.theta_y, jumps[j])
                    dJ_Vt[j] = np.sum(over_jump_z_v)
                    dJ_Yt[j] = np.sum(
                        list(
                            map(
                                lambda z: np.random.normal(
                                    self.mu_0 + self.mu_xy * z, self.sigma_xy
                                ),
                                over_jump_z_v,
                            )
                        )
                    )

            Yt = (
                Yt
                + (self.r - self.d - self.intensity * k - 0.5 * Vt) * dt
                + sqrt(Vt) * dW1t
                + dJ_Yt
            )
            Vt = (
                Vt
                + (self.Y_bar - self.k_y * Vt) * dt
                + sqrt(Vt)
                * (
                    self.corr * self.sigma_y * dW1t
                    + sqrt(1 - self.corr**2) * self.sigma_y * dW2t
                )
                + dJ_Vt
            )
            Vt = np.where(Vt > 0, Vt, 0)

        ST = exp(Yt)
        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="SVCDJ"):
        values = []
        print("SVCDJ simulation starting...")
        for i in range(self.N_repeat):
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: SVCDJ\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )

                file.write(s)
        return mean, std


class VGByMC:
    """
    Variance Gamma model
    """

    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        gamma_mean,
        gamma_var,
        poly_coeff,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

        self.gamma_mean = gamma_mean
        self.gamma_var = gamma_var

        self.poly_coeff = poly_coeff

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def Xt_cf(self, u):
        return pow(
            1
            - 1j * u * self.gamma_mean * self.gamma_var
            + 0.5 * self.sigma**2 * self.gamma_var * u**2,
            -self.T / self.gamma_var,
        )

    def getValue(self, repeat):
        # this method from Wikipwdia simulation 1
        # https://en.wikipedia.org/wiki/Variance_gamma_process

        mu = self.r - log(self.Xt_cf(-1j)) / self.T
        Xt = np.zeros(self.N_line)

        # Gamma process rv.
        G = np.random.gamma(
            shape=self.T / self.gamma_var, scale=self.gamma_var, size=self.N_line
        )
        # antithetic + moment match
        Z = np.random.randn(int(self.N_line / 2))
        Z = np.append(Z, -Z)
        Z = Z / np.std(Z)
        # random time normal increment
        W = Z * np.sqrt(G)
        # Brownian motion with random time
        Xt = self.gamma_mean * G + self.sigma * W

        ST = np.real(self.S0 * exp(mu * self.T + Xt))

        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="VG"):
        print("Variance gamma simulation starting...")
        values = []
        for i in range(self.N_repeat):
            print(f"VG: {i + 1} round.")
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: VG\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"\tgamma_mean: {self.gamma_mean}, gamma_var: {self.gamma_var}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )
                file.write(s)
        return mean, std


class NIGByMC:
    """
    Normal Inverse Gaussian
    """

    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        delta,
        alpha,
        beta,
        poly_coeff,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

        self.delta = delta
        self.alpha = alpha
        self.beta = beta

        self.poly_coeff = poly_coeff

        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

    def _payoff(self, price):
        payoff = 0
        for power, coeff in enumerate(self.poly_coeff):
            payoff += coeff * price**power
        return max(payoff, 0)

    def Xt_cf(self, u):
        return exp(
            self.delta
            * self.T
            * (
                pow(self.alpha**2 - self.beta**2, 0.5)
                - pow(self.alpha**2 - (self.beta + 1j * u) ** 2, 0.5)
            )
        )

    def getValue(self, repeat):
        mu = self.r - log(self.Xt_cf(-1j)) / self.T
        Xt = stats.norminvgauss.rvs(
            a=self.alpha * self.delta * self.T,
            b=self.beta * self.delta * self.T,
            scale=self.delta * self.T,
            size=self.N_line,
        )

        ST = np.sort(np.real(self.S0 * exp(mu * self.T + Xt)))

        payoff = list(map(self._payoff, ST))
        value = np.mean(payoff) * exp(-self.r * self.T)
        return value

    def getStatistic(self, save_data=False, save_dir="", file_name="NIG"):
        print("Normal Inverse Gaussian simulation starting...")
        values = []
        for i in range(self.N_repeat):
            print(f"NIG: {i + 1} round.")
            values.append(self.getValue(i))

        mean = np.mean(values).item()
        std = np.std(values).item()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        if save_data:
            with open(save_dir + f"/{file_name}", "a+") as file:
                s = (
                    f"\nProcess: NIG\n"
                    f"Polynomial Coefficient: {self.poly_coeff}\n"
                    f"Parameters:\n"
                    f"\tS0: {self.S0}, T: {self.T}, sigma: {self.sigma}, r: {self.r}\n"
                    f"\tdelta: {self.delta}, alpha: {self.alpha}, beta: {self.beta}\n"
                    f"Result:\n"
                    f"\tC.V.: ({lower_bound:.8f}, {upper_bound:.8f})\n"
                    f"\tmean: {mean:.52f}\n"
                    f"\tstd: {std}\n"
                )
                file.write(s)
        return mean, std


if __name__ == "__main__":
    # Basic
    r = 0.06
    d = 0.06
    T = 0.25
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
    # SVJJ
    SVJJ_corr = -0.82
    SVJJ_corr_J = -0.38
    SVJJ_v_bar = 0.008
    SVJJ_k_v = 3.46
    SVJJ_sigma_v = 0.14
    SVJJ_intensity = 0.47
    # SVJJ_mu_bar = -0.10
    SVJJ_sigma_y = 0.0001
    SVJJ_mu_v = 0.05
    SVJJ_v_0 = 0.087**2
    SVJJ_mu_y = -0.03
    # SVCDJ
    SVCDJ_Y_bar = 0.49
    SVCDJ_Y_0 = 0.0968
    SVCDJ_corr = -0.1
    SVCDJ_intensity = 1.64
    SVCDJ_mu_0 = -0.03
    SVCDJ_mu_xy = -7.78
    SVCDJ_sigma_xy = 0.22
    SVCDJ_theta_y = 0.0036
    SVCDJ_sigma_y = 0.61
    SVCDJ_k_y = 5.06

    S0 = 100
    poly_coeff = [-100, 1]
    #
    # SVJJByMC(S0=S0, r=r, d=d, T=T, sigma=sqrt(SVJJ_v_0), intensity=SVJJ_intensity, sigma_v=SVJJ_sigma_v, corr=SVJJ_corr,
    #          k_v=SVJJ_k_v, v_bar=SVJJ_v_bar, mu_v=SVJJ_mu_v, mu_y=SVJJ_mu_y, sigma_y=SVJJ_sigma_y, corr_J=SVJJ_corr_J, poly_coeff=poly_coeff,
    #              N_line=int(1e5), n=252, N_repeat=20)\
    #     .getStatistic(save_data=True, save_dir="./", file_name="SVJJ.txt")
    SVCDJByMC(
        S0=S0,
        r=r,
        d=d,
        T=T,
        sigma=sqrt(SVCDJ_Y_0),
        intensity=SVCDJ_intensity,
        corr=SVCDJ_corr,
        k_y=SVCDJ_k_y,
        sigma_y=SVCDJ_sigma_y,
        theta_y=SVCDJ_theta_y,
        mu_xy=SVCDJ_mu_xy,
        Y_bar=SVCDJ_Y_bar,
        mu_0=SVCDJ_mu_0,
        sigma_xy=SVCDJ_sigma_xy,
        poly_coeff=poly_coeff,
        N_line=int(1e6),
        n=252,
        N_repeat=20,
    ).getStatistic(save_data=True, save_dir="./", file_name="SVCDJ.txt")
