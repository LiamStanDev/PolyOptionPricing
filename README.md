# Introduction

這是一個接近指數誤差收斂率與線性時間複雜度，且可以使用多種隨機過程(e.g. Heston, SVJ, VG 等)
與多項式報酬函數的選擇權定價模型。

# Requirements

我使用 miniconda 作為包管理工具，所有使用的 packages 請參照 requirement.txt

# Usages

在 src 目錄下的 python 文件，均是可直接運行的程式入口，區別如下

- COS_Pricing.py: 為定價模板文件，可直接對其修改，或者複製一份對於你感興趣的選擇權進行定價
- CI_Plot.py: 生成論文中所有 Error accuracy 圖片與數據，結果存放在 Data/Error/CI_Plot 中
- Error_Ploy.py: 生成論文中所有 Error convergence speed 圖片，結果存放在 Data/Error/Error_Plot 中
- Cal_Time.py: 生成論文中所有 Table 中誤差小於 10 的-6 次方的計算時間
- Simulation_pricing.py: 計算論文中所有的 CI 的上界與下界，結果存放在 Data/Simulation 中

其他目錄

1. PolynomialPricingMethod 目錄：是用來放所有的與多項式選擇權定價的工具函數類等
   1. COSMethod.py: 本論文使用的定價模型
   2. PathSimulationMethod.py: 多項式選擇權定價使用蒙地卡羅模擬
   3. TreeMethod.py: 多項式選擇權定價使用樹模型
   4. utils 目錄:
      1. CharacteristicFunc.py: 共有 8 個隨機過程 e.g. Heston, GBM, VG, etc.
      2. DensityTools.py: 包含 DensityRecover 工具類，用於尋找 COSMehtod 的端點與畫出機率密度函數的圖
      3. Tools.py: 裡面有計算運算時間的工具函數
      4. ploy_utils: 有畫 Error Convergence 圖與 Confidence interval 圖的工具函數
      5. save_file_utils.py: 有將數據存放至 excel 的工具函數
2. ProcingMethod 目錄：存放計算 Call 定價公式
   1. CallCloseForm.py: 存放 GBM 與 Merton 兩個模型的計算方式
   2. 其他均為測試文件

運行方式如下

```bash
python src/COS_Pricing.py
```

# Note

- 所有註解只有 Call 有，其他的 payoff function 使用方式與 Call 相似
