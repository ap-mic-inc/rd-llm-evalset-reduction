# rd-llm-evalset-reduction

**為大型語言模型基準測試建立具代表性的縮小評測集，並結合分層抽樣與統計驗證方法。**  
*Build representative reduced eval sets for LLM benchmarks with stratified sampling and statistical validation.*

---

## 📖 Overview 專案總覽

大型評測集通常能提供較穩定的模型比較結果，但完整執行一次 benchmark 往往成本高、耗時久。  
本專案的目標是從完整評測集中自動挑選出一份較小、但仍具代表性的子集，並盡可能保留：

- 各模型的整體分數
- 模型彼此的相對排名
- 評測結果的穩定性

### 核心運作流程

本專案先根據各題在模型群上的平均答對率估計題目難度，並將題目劃分為 `hard`、`medium`、`easy` 三個層級。接著，依據母體中各難度層的實際比例進行分層抽樣，從完整題庫中搜尋具代表性的 300 題候選子集。最後，透過 Bootstrap 重抽分析與統計檢定，驗證該子集是否能在整體分數、模型排名與結果穩定性等面向上，高度近似完整評測集的表現。

### 評測題數設定-為何設定為 300 題

由於目前取得的資料中，各模型實際作答的題數並不一致，為確保模型比較與後續統計分析的有效性，所使用的評測矩陣必須不含缺失值。也就是說，僅能保留所有模型皆有作答紀錄的共同題目，作為可直接分析的完整矩陣。在此條件下，最終以原始 1538 題約五分之一的 300 題作為縮小版評測集規模。

若使用者擁有其他更完整或規模不同的評測資料，亦可依相同邏輯自行建立無缺失值的評測矩陣，並套用本專案流程進行代表性子集搜尋與驗證。資料格式可參考 `tmmlu_model_results_merged_complete_rows_only.csv`。

---

## ✨ Features 主要特色

- **自動化資料處理**：從 Hugging Face `twinkle-ai/datasets` 抓取多模型評測紀錄（eval-logs-and-scores），並自動整理成「題目 × 模型」的統一評分矩陣。
- **精準難度分層**：根據題目平均答對率自動估算難度，並依母體真實比例進行分層抽樣。
- **最佳化搜尋**：透過多次抽樣比對，搜尋出最能還原完整題庫分數與排名的最佳子集。
- **統計驗證**：內建 Bootstrap 穩定性分析與模型層級的統計檢定，確保抽樣品質。
- **結構化輸出**：產生分類清晰、便於追蹤的數據與報表檔案。

---

## 📁 Repository Structure 專案結構
```text
.
├── convert_csv_to_jsonl/            # CSV 轉 JSONL 的相關程式與資料夾
├── README.md                        # 專案說明文件
├── build_eval_results_table.py      # 抓取並整理評測結果表格
├── generate_evalsubset.py           # 執行分層抽樣、搜尋代表性子集與統計驗證
└── tmmlu_model_results_merged_*.csv # 模型評測結果整合資料
```
---

## 🛠 Installation & Setup 安裝指引

建議使用 Python 3.10 或以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scipy statsmodels datasets huggingface_hub pyarrow
```

---

## 📦 Data Source 資料來源

本專案的評測資料來自 Hugging Face `twinkle-ai/datasets`。  
這些資料集包含不同模型針對 TMMLU+ 題目的作答紀錄與詳細評分結果，可作為建構完整評測矩陣與後續子集分析的基礎。

---

## 🚀 Quick Start 快速開始

### Step 1. 建立完整評測矩陣（Build merged table）

先從 Hugging Face 抓取多模型紀錄，並整理成統一的 CSV 表格：

```bash
python build_eval_results_table.py
```

執行後將產出：

```text
tmlu_model_results_merged.csv
```

### Step 2. 生成縮小版評測集（Generate reduced eval subset）

使用主程式進行抽樣與驗證：

```bash
python generate_evalsubset.py
```

主程式流程如下：

1. **Detect model columns**  
   自動辨識模型作答結果欄位（0/1）。

2. **Compute item statistics**  
   計算每一題的平均答對率（`p`）與變異數（`p(1-p)`），並將題目分為：
   - `hard`：`p < 0.3`
   - `medium`：`0.3 <= p < 0.7`
   - `easy`：`p >= 0.7`

3. **Stratified sampling**  
   根據母體難度比例，分配 300 題子集中各層應抽取的題數。

4. **Search best subset**  
   重複抽樣多次，找出最能保留完整題庫分數與排名的題組。

5. **Bootstrap analysis**  
   對選出的子集進行重抽樣，檢查結果穩定性。

6. **Statistical validation**  
   進行模型層級的統計檢定，作為輔助分析。

---

## 📊 Output Files 輸出檔案說明

程式執行完畢後，會生成以下目錄結構與檔案：

```text
.
├── reduced_eval_set_300_population_ratio.csv
├── item_analysis/
│   └── item_statistics_with_difficulty.csv
├── model_comparison/
│   └── model_accuracy_comparison_population_ratio.csv
├── search_trials/
│   └── subset_search_trials_population_ratio.csv
├── bootstrap_results/
│   ├── bootstrap_per_model_stability.csv
│   ├── bootstrap_overall_summary.csv
│   └── ...
└── significance_tests/
    ├── significance_tests_across_models.csv
    └── significance_tests_per_model.csv
```

### 各檔案用途

- `reduced_eval_set_300_population_ratio.csv`  
  最終選出的 300 題縮小版評測集CSV檔（核心產出）。

- `item_analysis/item_statistics_with_difficulty.csv`  
  完整題庫各題統計資訊（答對率、變異數、難度分組）。

- `model_comparison/model_accuracy_comparison_population_ratio.csv`  
  各模型在完整題庫與子集上的表現差異比較。

- `search_trials/subset_search_trials_population_ratio.csv`  
  歷次候選子集的搜尋過程與表現紀錄。

- `bootstrap_results/`  
  Bootstrap 穩定性分析報表。

- `significance_tests/`  
  統計檢定結果報表。

### Step 3. 轉換回TMMLU+的jsonl格式（format transformation）

先檢視篩選到的評測集csv檔，如有必要可在此手動調整檔案內容(增刪題目)
`reduced_eval_set_300_population_ratio.csv`

執行 `convert_csv_to_jsonl` 程式，生成 `reduced_eval_set_300_tmmluplus.jsonl`,即可用於一般評測程式

---

## 📈 Interpretation Guidelines 數據解讀指南

### 1. 如何判斷縮小版評測集的好壞？

請優先檢查以下五個關鍵指標：

| 指標名稱 | 理想狀態 | 意義說明 |
|---|---|---|
| **MAE**（平均絕對誤差） | 越小越好 | 子集與完整題庫相比，平均每個模型分數的誤差（例：0.0078 代表 0.78%）。 |
| **Max abs diff** | 越小越好 | 所有模型中，最大偏差值為多少。 |
| **Spearman rank corr** | 接近 1 | 子集與完整題庫的模型排名相關性。接近 1 代表相對排名幾乎完整保留。 |
| **n_outside_tolerance** | 越小越好（最好為 0） | 誤差超過預設容忍值（±3%）的模型數量。 |
| **Full accuracy in bootstrap CI** | 越接近總模型數越好 | 完整題庫的分數中，有多少落在子集 Bootstrap 所估計的 95% 信賴區間內。 |

### 2. 如何解讀 Bootstrap 穩定性分析？

Bootstrap 的目的**不是**產生新考卷，而是檢查「這 300 題子集若有微小的抽樣波動，整體結果是否依然穩定」。

- `bootstrap_per_model_stability.csv`
  - `bootstrap_mean`：越接近完整題庫分數（`full_accuracy`）越好
  - `bootstrap_std`：數值越小代表越穩定
  - `within_tolerance_rate`：越接近 1 越好
  - `full_in_bootstrap_ci`：若為 `True`，表示與母體結果相容

- `bootstrap_overall_summary.csv`- `bootstrap_overall_summary.csv`
  - 重點觀察整體 `MAE`、`Max abs diff`（越小越好）以及 `Spearman rank corr`（接近 1）

### 3. 如何解讀統計檢定（Significance Tests）？

- `significance_tests_across_models.csv`（整體系統性偏差）  
  若 `p-value > 0.05`，代表未觀察到顯著的整體偏移；若 `p-value < 0.05`，則可能存在系統性偏差。

- `significance_tests_per_model.csv`（單一模型檢定）  
  `significant_at_0.05 = False` 代表該模型在子集與完整題庫間未有顯著差異。  

---

## 💡 Practical Rule of Thumb 實用準則

如果您的最終輸出大致符合以下條件，恭喜，這是一份品質極佳的縮小版評測集：

- [x] MAE 很小且 Max abs diff 不大
- [x] Spearman rank corr 非常接近 1
- [x] `n_outside_tolerance = 0`
- [x] 多數模型的 `full_in_bootstrap_ci = True`
- [x] Bootstrap 下的 `within_tolerance_rate` 高
- [x] Across-model 檢定未顯示顯著系統性偏差

---

## 📌 Notes 提醒

本專案的核心目標是在「測試成本 / 執行速度」與「代表性」之間取得平衡。  
統計檢定結果應與誤差指標、排名相關性一併綜合解讀。

若未來 benchmark 題庫或參測模型集合發生變動，建議重新執行本流程，以確保子集品質與代表性。


## 🔗 Additional Resources 相關補充資源

- 其他說明請見 [Notion 文件](https://notion.so/ap-mic/TMMLU-25b9a1ceaa6380f3a358e65cd19b1b41)
- 結果產出檔案影片請見 [Google Drive 影片連結](https://drive.google.com/file/d/12Mo8sQaMk6YQVVeYUxVTeqndsbFdgP9s/view?usp=sharing)

