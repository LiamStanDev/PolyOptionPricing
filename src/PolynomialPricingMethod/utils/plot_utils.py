import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from deprecated import deprecated
import statsmodels.api as sm

plt.rcParams["font.family"] = "Times New Roman"


@deprecated()
def plotError(x, y, process_name, save_path, enableShow=False, error_filter=-9):
    # 讀取輸入的 x 與 y 列表
    y = np.array(y)
    x = np.array(x, dtype=int)
    mask = np.full(len(y), True)
    for i in range(1, len(y)):
        if log10(y[i]) < error_filter and isSmallestFromKBefore(y, i, 5):
            for j in range(i, len(y)):
                mask[j] = False

    Y = log10(y[mask])
    X = (x[mask]).tolist()

    # 繪製帶有圓形點標記的紅色虛線折線圖
    plt.plot(X, Y, "ro--")
    # sns.scatterplot(y=Y, x=X, markers="x", color="b")

    # plt.title(f'{process_name}: Error Convergence with {benchmark_name} as Benchmark')
    plt.xlabel("N")
    plt.ylabel(r"$log_{10}\left(|Error|\right)$")

    Y_up_bound = (
        int(np.max(Y)) + 1
        if int(np.max(Y)) + 1 - np.max(Y) >= 1
        else int(np.max(Y)) + 2
    )
    Y_low_bound = (
        int(np.min(Y)) - 1
        if np.min(Y) - (int(np.min(Y)) - 1) >= 1
        else int(np.min(Y)) - 2
    )
    plt.ylim((Y_low_bound, Y_up_bound))
    # plt.xticks(X)
    sns.despine(top=True, right=True)

    if save_path is not None:
        plt.savefig(save_path + "/" + "error-plot-" + process_name + ".svg")
    if enableShow:
        plt.show()


def isSmallestFromKBefore(arr, i, lastK):
    res = True
    if i <= lastK:
        return res
    for j in range(i - 1, i - lastK, -1):
        if arr[i] > arr[j]:
            res = False
    return res


def plotValueWithCV(N_list, val_list, ci, std, save_dir, file_name, enableShow=False):
    val_list = np.array(val_list)
    plt.figure()
    x = 1 / N_list
    mean_simulation = (ci[1] + ci[0]) / 2
    plt.plot(x, val_list - mean_simulation, color="r", marker=",", label="Error")
    cvRange = ci[1] - ci[0]
    plt.axhline(ci[1] - mean_simulation, color="black", linestyle="--", label="C.I")
    plt.axhline(ci[0] - mean_simulation, color="black", linestyle="--")
    plt.text(
        np.min(x), ci[1] - mean_simulation + cvRange / 2, f"se = {std:.6f}", fontsize=12
    )
    plt.ylim(-8 * cvRange, 8 * cvRange)
    plt.xlabel("1/N")
    plt.ylabel("Error")
    plt.legend(loc="best")
    sns.despine(top=True, right=True)
    if save_dir is not None:
        plt.savefig(save_dir + "/" + file_name + ".svg")
    if enableShow:
        plt.show()


def plotErrorRegression(x, y, save_dir, file_name, error_filter=-8, enableShow=False):
    # 讀取輸入的 x 與 y 列表
    y = np.array(y)
    x = np.array(x, dtype=int)
    mask = np.full(len(y), True)
    for i in range(1, len(y)):
        if log10(y[i]) < error_filter and isSmallestFromKBefore(y, i, 5):
            for j in range(i, len(y)):
                mask[j] = False

    Y = log10(y[mask])
    X = x[mask]

    # regression
    # preprocessing
    polynomial_features = PolynomialFeatures(degree=3)
    Xp = polynomial_features.fit_transform(X.reshape(-1, 1))
    model = sm.OLS(Y, Xp)
    results = model.fit()
    Y_fit = results.fittedvalues

    print(results.summary())
    if len(X) > 500:
        plt.scatter(X[::15], Y[::15], marker="x", color="black")
    elif len(X) > 200:
        plt.scatter(X[::5], Y[::5], marker="x", color="black")
    elif len(X) > 100:
        plt.scatter(X[::2], Y[::2], marker="x", color="black")
    else:
        plt.scatter(X, Y, marker="x", color="black")
    # 畫fitting線
    plt.plot(X, Y_fit, color="black")

    # plt.title(f'{process_name}: Error Convergence with {benchmark_name} as Benchmark')
    plt.xlabel("N")
    plt.ylabel(r"$log_{10}\left(|Error|\right)$")

    Y_up_bound = (
        int(np.max(Y)) + 1
        if int(np.max(Y)) + 1 - np.max(Y) >= 1
        else int(np.max(Y)) + 2
    )
    Y_low_bound = (
        int(np.min(Y)) - 1
        if np.min(Y) - (int(np.min(Y)) - 1) >= 1
        else int(np.min(Y)) - 2
    )
    plt.ylim((Y_low_bound, Y_up_bound))
    # plt.xticks(X)
    sns.despine(top=True, right=True)

    if save_dir is not None:
        plt.savefig(save_dir + "/" + file_name + ".svg")
        with open(save_dir + f"/{file_name}" + "-summary.txt", "a+") as file:
            s = results.summary2().as_text()
            file.write(s)
    if enableShow:
        plt.show()
