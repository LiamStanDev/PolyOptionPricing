import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
import seaborn as sns
# import scienceplots
plt.rcParams["font.family"] = "Times New Roman"

def plotError(x, y, process_name, benchmark_name, save_path,error_filter=-10):
    # plt.style.use(['science','ieee'])
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
    plt.plot(X, Y, 'ro--')

    # 設定標題、座標軸標籤

    # plt.title(f'{process_name}: Error Convergence with {benchmark_name} as Benchmark')
    plt.xlabel('N')
    plt.ylabel(r'$log_{10}\left(|Error|\right)$')

    Y_up_bound = int(np.max(Y)) + 1 if int(np.max(Y)) + 1 - np.max(Y) >= 1 else int(np.max(Y)) + 2
    Y_low_bound = int(np.min(Y)) - 1 if np.min(Y) - (int(np.min(Y)) - 1) >= 1 else int(np.min(Y)) - 2
    plt.ylim((Y_low_bound, Y_up_bound))
    plt.xticks(X)
    sns.despine(top=True, right=True)

    if save_path is not None:
        plt.savefig(save_path + "/" + "error-plot-" + process_name + ".jpg", dpi=300)

    plt.show()
def isSmallestFromKBefore(arr, i, lastK):
    res = True
    if i <= lastK:
        return res
    for j in range(i - 1, i - lastK, -1):
        if arr[i] > arr[j]:
            res = False
    return res


def plotValueWithCV(N_list, val_list, ci, std, save_dir, file_name):
    val_list = np.array(val_list)
    plt.figure(dpi=600)
    x = 1 / N_list
    mean_simulation = (ci[1] + ci[0]) / 2
    plt.plot(x, val_list - mean_simulation, color='r', marker=',', label="Error")
    rangeOfValue = np.max(val_list) - np.min(val_list)
    cvRange = ci[1] - ci[0]
    plt.axhline(ci[1] - mean_simulation, color="black", linestyle='--', label="C.I")
    plt.axhline(ci[0] - mean_simulation, color="black", linestyle='--')
    plt.text(np.min(x), ci[1] - mean_simulation + cvRange / 2, f"se = {std:.6f}", fontsize=12)
    plt.ylim(-8 * cvRange, 8 * cvRange)
    plt.xlabel("1/N")
    plt.ylabel("Error")
    plt.legend(loc="best")
    sns.despine(top=True, right=True)
    if save_dir is not None:
        plt.savefig(save_dir + "/" + file_name + ".jpg", dpi=300)
    plt.show()