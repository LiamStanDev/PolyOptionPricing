# Introduction

這是一個指數誤差收斂率(O(e(n)))與線性定價速率(O(n))，且可以使用多種隨機過程(e.g. Heston, SVJ, VG 等)
與多項式報酬函數(一次多項式(call)，高次多項式)的選擇權定價模型。

# Requirements

我使用 miniconda 作為包管理工具，所有使用的 packages 請參照 requirement.txt

# Usages

在 src 目錄下的 python 文件，均是可直接運行的程式入口，區別如下

- COS_Pricing.py: 為定價模板文件，可直接對其修改，或者複製一份對於你感興趣的選擇權進行定價
- CI_Plot.py: 生成論文中所有 Error accuracy 圖片，結果存放在 Data/Error/CI_Plot 中
- Error_Ploy.py: 生成論文中所有 Error convergence speed 圖片，結果存放在 Data/Error/Error_Plot 中

運行方式如下

```bash
python src/COS_Pricing.py
```
