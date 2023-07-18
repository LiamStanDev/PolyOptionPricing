import pandas as pd

pd.set_option("display.precision", 20)


def save_to_excel(N_arr, value_arr, save_dir, file_name):
    # 將NumPy數組轉換為Pandas DataFrame
    df = pd.DataFrame({"N": N_arr, "value": value_arr})

    # 保存DataFrame為Excel文件
    df.to_excel(save_dir + "/" + file_name + ".xlsx", index=False, float_format="%.20f")
