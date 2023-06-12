from PolynomialPricingMethod.TreeMethod import PolyByTree
from PolynomialPricingMethod.COSMethod import PolyByCosMethod
from PolynomialPricingMethod.utils.CharacteristicFunc import GBM
from PolynomialPricingMethod.utils.DensityTools import DensityRecover
from PolynomialPricingMethod.utils.Tools import timeit
from PricingMethod.CallCloseForm import BSMCloseForm
from math import inf
import numpy as np
from numpy import sqrt

##################### Process Setting ######################
# Basic
r = 0.05
T = 0.5
sigma = 0.2
process_cf = GBM(r, sigma, T)


# Call
def cal_call():
    S0 = 100
    poly_coef = [-100, 1]
    positive_interval = [100, inf]
    best_positive_interval = positive_interval.copy()

    densityRecover = DensityRecover(
        S0=S0,
        process_cf=process_cf,
        poly_coeff=poly_coef,
        positive_interval=positive_interval,
        error_acceptance=1e-6,
    )
    (
        lower_limit,
        upper_limit,
        best_positive_interval[0],
        best_positive_interval[-1],
    ) = densityRecover.getIntegralRangeAndInterval()

    cos_time_arr = []
    cos_res_arr = []
    bin_time_arr = []
    bin_res_arr = []
    for i, N in enumerate([10, 100, 1000, 10000, 100000]):
        print("N:", N)
        bin = PolyByTree(S0, T, r, sigma, poly_coef, N)

        COS = PolyByCosMethod(
            S0,
            T,
            r,
            sigma,
            process_cf,
            poly_coef,
            best_positive_interval,
            N,
            lower_limit,
            upper_limit,
        )

        bin_result, bin_time = timeit(bin.getValue)
        bin_time_arr.append(bin_time)
        bin_res_arr.append(bin_result)

        cos_result, cos_time = timeit(COS.getValue)
        cos_time_arr.append(cos_time)
        cos_res_arr.append(cos_result)
    true_val = BSMCloseForm(S0, T, r, sigma, 100).getValue()
    cos_err = np.log10(np.abs(np.array(cos_res_arr) - true_val))
    bin_err = np.log10(np.abs(np.array(bin_res_arr) - true_val))
    # plotCOSvsBin(
    #     cos_err=cos_err,
    #     cos_time=np.array(cos_time_arr),
    #     bin_err=bin_err,
    #     bin_time=np.array(bin_time_arr),
    #     save_dir=None,
    #     file_name=None,
    #     enableShow=True,
    # )
    print("COS:")
    print(cos_time_arr)
    print(cos_err)

    print("Bin:")
    print(bin_time_arr)
    print(bin_err)


def cal_rightup():
    S0 = 90
    poly_coef = [-20, -5, 0.05]
    positive_interval = [10 * (5 + sqrt(29)), inf]
    best_positive_interval = positive_interval.copy()

    densityRecover = DensityRecover(
        S0=S0,
        process_cf=process_cf,
        poly_coeff=poly_coef,
        positive_interval=positive_interval,
        error_acceptance=1e-6,
    )
    (
        lower_limit,
        upper_limit,
        best_positive_interval[0],
        best_positive_interval[-1],
    ) = densityRecover.getIntegralRangeAndInterval()

    cos_time_arr = []
    cos_res_arr = []
    bin_time_arr = []
    bin_res_arr = []
    for i, N in enumerate([10, 100, 1000, 10000, 100000]):
        print("N:", N)
        bin = PolyByTree(S0, T, r, sigma, poly_coef, N)

        COS = PolyByCosMethod(
            S0,
            T,
            r,
            sigma,
            process_cf,
            poly_coef,
            best_positive_interval,
            N,
            lower_limit,
            upper_limit,
        )

        bin_result, bin_time = timeit(bin.getValue)
        bin_time_arr.append(bin_time)
        bin_res_arr.append(bin_result)

        cos_result, cos_time = timeit(COS.getValue)
        cos_time_arr.append(cos_time)
        cos_res_arr.append(cos_result)
    true_val = BSMCloseForm(S0, T, r, sigma, 100).getValue()
    cos_err = np.log10(np.abs(np.array(cos_res_arr) - true_val))
    bin_err = np.log10(np.abs(np.array(bin_res_arr) - true_val))
    # plotCOSvsBin(
    #     cos_err=cos_err,
    #     cos_time=np.array(cos_time_arr),
    #     bin_err=bin_err,
    #     bin_time=np.array(bin_time_arr),
    #     save_dir=None,
    #     file_name=None,
    #     enableShow=True,
    # )
    print("COS:")
    print(cos_time_arr)
    print(cos_err)

    print("Bin:")
    print(bin_time_arr)
    print(bin_err)


# Call
def cal_rightdown():
    S0 = 30
    poly_coef = [-44.235, 39.474, -5.4793, 0.2358, -0.0031]
    positive_interval = [1.363962, 10.620047, 25.599102, 38.481405]
    best_positive_interval = positive_interval.copy()

    densityRecover = DensityRecover(
        S0=S0,
        process_cf=process_cf,
        poly_coeff=poly_coef,
        positive_interval=positive_interval,
        error_acceptance=1e-6,
    )
    (
        lower_limit,
        upper_limit,
        best_positive_interval[0],
        best_positive_interval[-1],
    ) = densityRecover.getIntegralRangeAndInterval()

    cos_time_arr = []
    cos_res_arr = []
    bin_time_arr = []
    bin_res_arr = []
    for i, N in enumerate([10, 100, 1000, 10000, 100000]):
        print("N:", N)
        bin = PolyByTree(S0, T, r, sigma, poly_coef, N)

        COS = PolyByCosMethod(
            S0,
            T,
            r,
            sigma,
            process_cf,
            poly_coef,
            best_positive_interval,
            N,
            lower_limit,
            upper_limit,
        )

        bin_result, bin_time = timeit(bin.getValue)
        bin_time_arr.append(bin_time)
        bin_res_arr.append(bin_result)

        cos_result, cos_time = timeit(COS.getValue)
        cos_time_arr.append(cos_time)
        cos_res_arr.append(cos_result)
    true_val = BSMCloseForm(S0, T, r, sigma, 100).getValue()
    cos_err = np.log10(np.abs(np.array(cos_res_arr) - true_val))
    bin_err = np.log10(np.abs(np.array(bin_res_arr) - true_val))
    # plotCOSvsBin(
    #     cos_err=cos_err,
    #     cos_time=np.array(cos_time_arr),
    #     bin_err=bin_err,
    #     bin_time=np.array(bin_time_arr),
    #     save_dir=None,
    #     file_name=None,
    #     enableShow=True,
    # )
    print("COS:")
    print(cos_time_arr)
    print(cos_err)

    print("Bin:")
    print(bin_time_arr)
    print(bin_err)


cal_call()
