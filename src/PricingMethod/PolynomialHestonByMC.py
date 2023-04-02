import numpy as np
from numpy import log, exp
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class PolynomialHestonByMC:
    def __init__(
        self,
        S0,
        r,
        sigma,
        T,
        mean_reversion_speed,
        long_term_var_mean,
        corr,
        var_variance_process,
        poly_coef,
        N_line=10000,
        n=252,
        N_repeat=10,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.mean_reversion_speed = mean_reversion_speed
        self.long_term_var_mean = long_term_var_mean
        self.corr = corr
        self.var_variance_process = var_variance_process
        self.poly_coef = poly_coef
        self.N_line = N_line
        self.n = n
        self.N_repeat = N_repeat

        self.delta_T = self.T / self.n

        self.lnSt_df = np.zeros((self.n, self.N_line))
        self.maxlnST = []
        self.minlnST = []

    def Payoff(self, price):
        payoff = 0
        for power, coef in enumerate(self.poly_coef):
            payoff += coef * price**power
        return max(payoff, 0)

    def getValue(self):
        lnST = []
        for i in range(self.N_line):
            if (i + 1) % 1000 == 0:
                print("i = ", i + 1)
            prev_lnSt = log(self.S0)
            prev_sigma = self.sigma
            for j in range(self.n):
                z1 = np.random.randn(1)
                z2 = np.random.randn(1)
                dW1t = z1 * pow(self.delta_T, 0.5)
                dW2t = pow(self.delta_T, 0.5) * (
                    z1 * self.corr + z2 * pow(1 - pow(self.corr, 2), 0.5)
                )
                delta_lnSt = (
                    self.r - 0.5 * prev_sigma**2
                ) * self.delta_T + prev_sigma * dW1t
                delta_sigma_squre = (
                    self.mean_reversion_speed
                    * (self.long_term_var_mean - prev_sigma**2)
                    * self.delta_T
                    + self.var_variance_process * prev_sigma * dW2t
                )
                lnSt = prev_lnSt + delta_lnSt
                sigma_squre = prev_sigma**2 + delta_sigma_squre
                # 若sigma為負，直接用零取代
                sigma_squre = sigma_squre if sigma_squre > 0 else 0
                prev_lnSt = lnSt
                prev_sigma = pow(sigma_squre, 0.5)

                self.lnSt_df[j, i] = prev_lnSt
            lnST.append(prev_lnSt)
        self.maxlnST.append(np.max(lnST))
        self.minlnST.append(np.min(lnST))

        payoff_ls = []
        for S in np.exp(lnST):
            payoff_ls.append(self.Payoff(S))

        mean = np.mean(payoff_ls)
        return exp(-self.r * self.T) * mean

    def getStastic(self, plot=False, plot_save=False, file_name=""):
        values = []
        for i in range(self.N_repeat):
            print("N =", i)
            values.append(self.getValue())
            if plot:
                self.plotMC(i, plot_save, file_name)
        mean = np.mean(values)
        std = np.std(values)
        return (
            mean - 2 * std,
            mean + 2 * std,
            np.exp(np.min(self.minlnST)),
            np.exp(np.max(self.maxlnST)),
        )

    def plotMC(self, i, plot_save=False, file_name=""):
        data = np.exp(self.lnSt_df)
        plt.plot(data)
        plt.title("Heston Simulation")
        if plot_save:
            plt.savefig(file_name + f"_{i + 1}" + ".jpg")
        plt.show()


if __name__ == "__main__":
    polynomialHestonMC = PolynomialHestonByMC(
        S0=100,
        r=0,
        sigma=0.0175,
        T=1,
        mean_reversion_speed=1.5768,
        long_term_var_mean=0.0398,
        corr=-0.5711,
        var_variance_process=0.0751,
        poly_coef=[-100, 1],
        N_line=10000,
        n=252,
        N_repeat=20,
    )
    print(polynomialHestonMC.getStastic())
