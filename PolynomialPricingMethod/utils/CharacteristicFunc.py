import numpy as np
from numpy import exp, sqrt, log, power
from abc import abstractmethod, ABC

i = 1j


class CharacteristicFunction(ABC):
    @abstractmethod
    def getCFValue(self, u):
        pass


class GBM(CharacteristicFunction):
    """
    Black-Scholes Model
    """

    def __init__(self, r, sigma, T):
        self.r = r
        self.sigma = sigma
        self.T = T

    def getCFValue(self, u):
        cf_value = exp((self.r - 0.5 * self.sigma ** 2) * self.T * i * u -
                       0.5 * self.sigma ** 2 * u ** 2 * self.T)

        return cf_value


class Heston(CharacteristicFunction):
    """
    Heston's SV Model
    """

    def __init__(self, r, sigma, T, mean_reversion_speed, long_term_var_mean, corr, std_of_var_process):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.mean_reversion_speed = mean_reversion_speed
        self.long_term_var_mean = long_term_var_mean
        self.corr = corr
        self.std_of_var_process = std_of_var_process

    def getCFValue(self, u):
        alpha = (self.mean_reversion_speed - i * self.corr * self.std_of_var_process * u) ** 2
        beta = (u ** 2 + i * u) * self.std_of_var_process ** 2
        b = sqrt(alpha + beta)
        a_minus = self.mean_reversion_speed - i * self.corr * self.std_of_var_process * u - b
        a_plus = self.mean_reversion_speed - i * self.corr * self.std_of_var_process * u + b
        a = a_minus / a_plus
        eta = 1 - a * exp(-b * self.T)
        C = i * u * self.r * self.T + self.mean_reversion_speed * self.long_term_var_mean / pow(self.std_of_var_process,
                                                                                                2) * (
                    self.T * a_minus - 2 * log(eta / (1 - a)))
        D = a_minus / pow(self.std_of_var_process, 2) * ((1 - exp(-b * self.T)) / eta)
        cf_value = exp(C + D * pow(self.sigma, 2))

        return cf_value


class SVJ(CharacteristicFunction):
    """
    Bates's Stochastic Volatility Jump Model
    """

    def __init__(self, r, sigma, T, mean_reversion_speed, long_term_var_mean, corr, std_of_var_process, jump_intensity,
                 jump_mean, jump_var):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.mean_reversion_speed = mean_reversion_speed
        self.long_term_var_mean = long_term_var_mean
        self.corr = corr
        self.std_of_var_process = std_of_var_process
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_var = jump_var

    def getCFValue(self, u):
        k = exp(self.jump_mean + 0.5 * self.jump_var) - 1
        mu = self.r - self.jump_intensity * k

        alpha = (self.mean_reversion_speed - i * self.corr * self.std_of_var_process * u) ** 2
        beta = (u ** 2 + i * u) * self.std_of_var_process ** 2
        b = sqrt(alpha + beta)
        a_minus = self.mean_reversion_speed - i * self.corr * self.std_of_var_process * u - b
        a_plus = self.mean_reversion_speed - i * self.corr * self.std_of_var_process * u + b
        a = a_minus / a_plus
        eta = 1 - a * exp(-b * self.T)
        C = i * u * mu * self.T + self.mean_reversion_speed * self.long_term_var_mean / pow(self.std_of_var_process,
                                                                                            2) * (
                    a_minus * self.T - 2 * log(eta / (1 - a)))
        D = a_minus / pow(self.std_of_var_process, 2) * ((1 - exp(-b * self.T)) / eta)

        cf_value = exp(C + D * pow(self.sigma, 2) - self.jump_intensity * self.T * (
                1 - exp(i * u * self.jump_mean - 0.5 * pow(u, 2) * self.jump_var)))
        return cf_value


class MJD(CharacteristicFunction):
    """
    Merton's Normal Jump Diffusion Model
    """

    def __init__(self, r, sigma, T, jump_intensity, jump_mean, jump_var):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_var = jump_var

    def getCFValue(self, u):
        k = exp(self.jump_mean + 0.5 * self.jump_var) - 1
        mu = self.r - self.jump_intensity * k
        lnY_cf = exp(i * u * self.jump_mean - 0.5 * u ** 2 * self.jump_var)
        cf_value = exp(i * u * (mu - pow(self.sigma, 2) / 2) * self.T - 0.5 * pow(u * self.sigma,
                                                                                  2) * self.T - self.jump_intensity * self.T * (
                               1 - lnY_cf))
        return cf_value


# class BJD(CharacteristicFunction):
#     """
#     Amin's Bivariate Jump Diffusion Model
#     """
#
#     def __init__(self, p, delta):
#         self.p = p
#         self.delta = delta
#
#     def getCFValue(self, u):
#         cf_value = self.p * exp(i * u * self.delta) + (1 - self.p) * exp(-i * u * self.delta)
#         return cf_value
#
#
class KJD(CharacteristicFunction):
    """
    Kou's Double Exponential Jump Diffusion Model
    """

    def __init__(self, r, sigma, T, jump_intensity, p, eat1, eat2):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.jump_intensity = jump_intensity
        self.p = p
        self.eta1 = eat1
        self.eta2 = eat2

    def getCFValue(self, u):
        k = self.p * self.eta1 / (self.eta1 - 1) + (1 - self.p) * self.eta2 / (self.eta2 + 1) - 1
        mu = self.r - self.jump_intensity * k
        lnY_cf = self.p * self.eta1 / (self.eta1 - i * u) + (1 - self.p) * self.eta2 / (self.eta2 + i * u)
        cf_value = exp(i * u * (mu - pow(self.sigma, 2) / 2) * self.T - 0.5 * pow(u * self.sigma,
                                                                                  2) * self.T - self.jump_intensity * self.T * (
                               1 - lnY_cf))

        return cf_value


class VG(CharacteristicFunction):
    """
    Madan, D.B., and Seneta, E. (1990). The Variance Gamma Model
    """

    def __init__(self, gamma_mean, gamma_var, sigma, T, r):
        self.gamma_mean = gamma_mean
        self.gamma_var = gamma_var
        self.sigma = sigma
        self.r = r
        self.T = T

    def Xt_cf(self, u):
        return pow(1 - i * u * self.gamma_mean * self.gamma_var + 0.5 * self.sigma ** 2 * self.gamma_var * u ** 2,
                   -self.T / self.gamma_var)

    def getCFValue(self, u):
        mu = self.r - log(self.Xt_cf(-i)) / self.T
        return_cf = exp(i * u * mu * self.T) * self.Xt_cf(u)

        return return_cf


class NIG(CharacteristicFunction):
    """
    Barndorff-Nielsen, O.L. (1997) Normal Inverse Gaussian Model
    """

    def __init__(self, r, T, delta, alpha, beta):
        self.r = r
        self.T = T
        self.delta = delta
        self.alpha = alpha
        self.beta = beta

    def Xt_cf(self, u):
        return exp(self.delta * self.T * (
                    pow(self.alpha ** 2 - self.beta ** 2, 0.5) - pow(self.alpha ** 2 - (self.beta + i * u) ** 2, 0.5)))

    def getCFValue(self, u):
        mu = self.r - log(self.Xt_cf(-i)) / self.T
        return_cf = exp(i * u * mu * self.T) * self.Xt_cf(u)

        return return_cf


# class CGMY(CharacteristicFunction):
#     """
#     Carr, P., Geman, H., Madan, D., and Yor, M. (2002) CGMY Model
#     """
#
#     def __init__(self, r, T, sigma, C, G, M, Y):
#         self.r = r
#         self.T = T
#         self.sigma = sigma
#         self.C = C
#         self.G = G
#         self.M = M
#         self.Y = Y
#
#     def getCFValue(self, u):
#         A = i * u * self.r * self.T - 0.5 * u ** 2 * self.sigma ** 2 * self.T
#         B = self.T * self.C * gamma(-self.Y)
#         C = (self.M - i * u) ** self.Y - self.M ** self.Y + (self.G + i * u) ** self.Y - self.G ** self.Y
#         cf_value = exp(A + B * C)
#         return cf_value


class SVJJ(CharacteristicFunction):
    """
    Stochastic Volatility Model With Correlated Double Jumps(Duffie)
    """

    def __init__(self, r, d, T, sigma, intensity, sigma_v, corr, k_v, v_bar, mu_v, mu_y, sigma_y, corr_J):
        self.r = r
        self.d = d
        self.T = T
        self.v0 = sigma ** 2
        self.intensity = intensity
        self.sigma_v = sigma_v
        self.corr = corr
        self.k_v = k_v
        self.v_bar = v_bar
        self.mu_v = mu_v
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        self.corr_J = corr_J

    def getCFValue(self, u):
        if u == 0:
            return self._getCFValue(0.0001)
        return self._getCFValue(u)
    def _getCFValue(self, u):
        b = self.sigma_v * self.corr * i * u - self.k_v
        a = i * u * (1 - i * u)
        gamma = sqrt(b ** 2 + a * self.sigma_v ** 2)
        beta = - a * (1 - exp(-gamma * self.T)) / (2 * gamma - (gamma + b) * (1 - exp(- gamma * self.T)))

        alpha_0 = -self.r * self.T + (self.r - self.d) * i * u * self.T - self.k_v * self.v_bar * (
                (gamma + b) / power(self.sigma_v, 2) * self.T + 2 / power(self.sigma_v, 2) * log(
            1 - (gamma + b) / (2 * gamma) * (1 - exp(-gamma * self.T)))
        )
        c = 1 - self.corr_J * self.mu_v * i * u

        d = (gamma - b) / ((gamma - b) * c + self.mu_v * a) * self.T \
            - 2 * self.mu_v * a / (power(gamma * c, 2) - power(b * c - self.mu_v * a, 2)) * log(
            1 - ((gamma + b) * c - self.mu_v * a) / (2 * gamma * c) * (1 - exp(-gamma * self.T)))
        f = exp(self.mu_y * i * u - 0.5 * self.sigma_y ** 2 * u ** 2) * d

        theta = lambda c1, c2: exp(self.mu_y * c1 + 0.5 * power(self.sigma_y * c1, 2)) / (
                1 - self.mu_v * c2 - self.corr_J * self.mu_v * c1)
        mu_bar = theta(1, 0) - 1
        alpha = alpha_0 - self.intensity * self.T * (1 + mu_bar * i * u) + self.intensity * f

        return exp(alpha + beta * self.v0)

class SVCDJ(CharacteristicFunction):
    """
    Stochastic Volatility Model With Correlated Double Jumps(Guo 2009)
    """

    def __init__(self, r, d, T, sigma, intensity, corr, k_y, sigma_y, theta_y, mu_xy, Y_bar, mu_0, sigma_xy):
        self.r = r
        self.d = d
        self.T = T
        self.Y_0 = sigma ** 2
        self.sigma_y = sigma_y
        self.corr = corr
        self.k_y = k_y
        self.theta_y = theta_y
        self.mu_xy = mu_xy
        self.intensity = intensity
        self.Y_bar = Y_bar
        self.mu_0 = mu_0
        self.sigma_xy = sigma_xy

    def getCFValue(self, u):
        if u == 0:
            return self._getCFValue(0.0001)
        return self._getCFValue(u)

    def _getCFValue(self, u):
        e = sqrt((i * u * self.sigma_y * self.corr - self.k_y) ** 2 - i * u * (i * u - 1) * self.sigma_y ** 2)
        b = e + i * u * self.sigma_y * self.corr - self.k_y
        q = i * u * (i * u - 1) * self.theta_y + b * (1 - self.theta_y * i * u * self.mu_xy)
        p = 2 * e * (1 - self.theta_y * i * u * self.mu_xy) - q
        B = i * u * (i * u - 1) * (1 - exp(-e * self.T)) / (
                    2 * e - (e + i * u * self.sigma_y * self.corr - self.k_y) * (1 - exp(-e * self.T)))
        A = (i * u * (self.r - self.d) - self.r) * self.T
        A -= self.Y_bar / power(self.sigma_y, 2) * (
                    (e + i * u * self.sigma_y * self.corr - self.k_y) * self.T + 2 * log(
                1 - (e + i * u * self.sigma_y * self.corr - self.k_y) * (1 - exp(-e * self.T)) / (2 * e)))
        A -= i * u * self.intensity * (exp(self.mu_0 + 0.5 * self.sigma_xy ** 2) / (
                    1 - self.theta_y * self.mu_xy) - 1) * self.T + self.intensity * self.T
        A += self.intensity * (2 * e - b) * exp(i * u * self.mu_0 - 0.5 * u ** 2 * self.sigma_xy ** 2) * self.T / p
        A += 2 * self.intensity * self.theta_y * i * u * (i * u - 1) * exp(
            i * u * self.mu_0 - 0.5 * u ** 2 * self.sigma_xy ** 2) / (p * q) * log((p + q * exp(-e * self.T)) / (p + q))

        return exp(A + B * self.Y_0)