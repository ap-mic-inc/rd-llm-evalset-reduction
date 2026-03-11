# rd-llm-evalset-reduction

Build representative reduced eval sets for LLM benchmarks with stratified sampling and statistical validation.  
為大型語言模型基準測試建立具代表性的縮小評測集，並結合分層抽樣與統計驗證方法。

---

## Overview

大型評測集通常能提供較穩定的模型比較結果，但完整跑完一次 benchmark 往往成本高、速度慢。  
本專案的目的，是從完整評測集中自動挑選出一份較小、但仍具代表性的子集，盡可能保留：

- 各模型的整體分數
- 模型彼此的相對排名
- 評測結果的穩定性

目前流程會先根據各題在模型群上的平均答對率估計題目難度，將題目分為 `hard / medium / easy` 三層，再依母體難度比例進行分層抽樣，從完整題庫中搜尋最具代表性的 300 題候選子集。之後再以 Bootstrap 重抽分析與統計檢定，檢查這份子集是否能近似重現完整評測集的結果。

---

## Features

- 從 Hugging Face `twinkle-ai/datasets` 抓取多模型 `eval-logs-and-scores`
- 自動整理成題目 × 模型的統一評測表格
- 根據題目平均答對率估計題目難度
- 依母體難度比例進行分層抽樣
- 多次抽樣搜尋最佳縮小版評測集
- 提供 Bootstrap 穩定性分析
- 提供模型層級統計檢定結果
- 輸出清楚分資料夾整理的結果檔案

---

## Repository Structure

```text
.
├── build_eval_results_table.py
├── generate_evalsubset.py
├── README.md
└── ...
Main Scripts

build_eval_results_table.py
從 Hugging Face twinkle-ai/datasets 抓取各模型的 eval-logs-and-scores，整理成統一 CSV 表格。

generate_evalsubset.py
讀取整理好的模型答題表格，進行題目難度分析、分層抽樣、最佳子集搜尋、Bootstrap 穩定性分析與統計檢定。

Installation

建議使用 Python 3.10 以上版本。

Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scipy statsmodels datasets huggingface_hub pyarrow
Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scipy statsmodels datasets huggingface_hub pyarrow
Data Source

本專案使用的模型評測結果資料來源：

Hugging Face: https://huggingface.co/twinkle-ai/datasets

這些 dataset 包含不同模型對 TMMLU+ 題目的作答紀錄與評分結果。

Quick Start
Step 1. Build the merged evaluation table

先從 twinkle-ai/datasets 抓取多模型的 eval-logs-and-scores，並整理成統一表格。

python build_eval_results_table.py

執行後通常會產出類似以下檔案：

tmmlu_model_results_merged.csv

若你另外有做過完整列過濾，也可能接著使用：

tmmlu_model_results_merged_complete_rows_only.csv
Step 2. Generate the reduced eval subset

使用主程式根據完整題庫建立縮小版評測集。

python generate_evalsubset.py
What generate_evalsubset.py Does

這支主程式會依序完成以下流程：

1. Detect model columns

自動辨識哪些欄位是模型作答結果欄位（0/1）。

2. Compute item statistics

對每一題計算：

p_value：平均答對率

q_value = 1 - p_value

item_variance = p(1-p)

並依 p_value 將題目分成：

hard：p < 0.3

medium：0.3 <= p < 0.7

easy：p >= 0.7

3. Stratified sampling by population ratio

根據母體中 hard / medium / easy 的實際比例，分配 300 題子集中各層應抽幾題。

4. Search for the best subset

重複抽樣多次，從大量候選子集中找出最能保留完整題庫模型分數與排名的題組。

5. Bootstrap stability analysis

對最佳 300 題子集進行 Bootstrap 重抽，檢查評測結果是否穩定。

6. Statistical validation

對子集與完整題庫之間的模型分數差異進行模型層級的統計檢定，作為輔助分析。

Output Structure

執行完成後，輸出通常會整理成以下結構：

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
│   ├── bootstrap_raw_model_accuracies.csv
│   └── bootstrap_raw_overall_metrics.csv
└── significance_tests/
    ├── significance_tests_across_models.csv
    └── significance_tests_per_model.csv
Output Files Explained
reduced_eval_set_300_population_ratio.csv

最終選出的 300 題縮小版評測集。
這是後續實際拿來快速評測的主要檔案。

item_analysis/item_statistics_with_difficulty.csv

完整題庫中每一題的統計資訊，包含：

p_value

q_value

item_variance

difficulty_group

model_comparison/model_accuracy_comparison_population_ratio.csv

比較各模型在完整題庫與 300 題子集上的表現差異。

search_trials/subset_search_trials_population_ratio.csv

記錄每次候選子集的搜尋結果，方便檢查不同候選集的表現。

bootstrap_results/

存放 Bootstrap 穩定性分析結果。

significance_tests/

存放統計檢定結果。

How to Judge Whether the Reduced Eval Set Is Good

建議優先看以下幾個指標。

1. MAE

平均絕對誤差，表示子集與完整題庫相比，平均每個模型分數差多少。

越小越好

例如 0.0078 表示平均只差 0.78 個百分點

2. Max abs diff

所有模型中，偏差最大的模型與完整題庫相比差多少。

越小越好

3. Spearman rank corr

子集與完整題庫的模型排名相關。

越接近 1 越好

若接近 1，表示模型相對排名幾乎完整保留

4. n_outside_tolerance

有多少模型的誤差超過容忍值。
目前預設容忍值為 ±3%。

0 代表所有模型都在可接受誤差內

5. full accuracy 落在 subset 95% CI 內的模型數

有多少模型在完整題庫上的分數，落在子集估計的 95% 信賴區間中。

越接近全部模型數越好

How to Read Bootstrap Results

Bootstrap 的目的不是重新產生一份新考卷，而是檢查：

這份 300 題子集本身是否穩定？

也就是說，若題目組成有些微抽樣波動，模型分數與整體結果是否仍維持相近。

建議重點看兩份檔案
bootstrap_results/bootstrap_per_model_stability.csv

重點欄位：

bootstrap_mean

bootstrap_std

bootstrap_ci_lower_95

bootstrap_ci_upper_95

full_in_bootstrap_ci

within_tolerance_rate

解讀方式：

bootstrap_mean 越接近 full_accuracy 越好

bootstrap_std 越小越穩定

full_in_bootstrap_ci = True 表示與母體相容

within_tolerance_rate 越接近 1 越好

bootstrap_results/bootstrap_overall_summary.csv

重點欄位：

mae

max_abs_diff

spearman_rank_corr

解讀方式：

mae 越小越好

max_abs_diff 越小越好

spearman_rank_corr 越接近 1 越好

How to Read Statistical Tests
significance_tests/significance_tests_across_models.csv

包含：

one_sample_t_test

wilcoxon_signed_rank

用途是檢查：

across 所有模型來看，300 題子集是否存在顯著的整體系統性偏差

解讀方式：

p_value > 0.05：未觀察到顯著的整體偏移

p_value < 0.05：可能存在系統性偏差

significance_tests/significance_tests_per_model.csv

對每個模型 individually 進行近似比例檢定。

重點欄位：

subset_accuracy

full_accuracy

p_value

significant_at_0.05

解讀方式：

significant_at_0.05 = False：未觀察到顯著差異

significant_at_0.05 = True：該模型可能與完整題庫存在較大差異

注意：由於 300 題子集本身是完整題庫的子集，因此 per-model 比例檢定屬於近似參考，建議作為輔助分析使用。

Practical Rule of Thumb

若最後結果大致符合以下條件，通常可視為品質不錯的縮小版評測集：

MAE 很小

Max abs diff 不大

Spearman rank corr 非常接近 1

n_outside_tolerance = 0

多數模型的 full_in_subset_ci = True

Bootstrap 下的 within_tolerance_rate 高

Across-model 檢定未顯示顯著系統性偏差

若同時滿足以上條件，通常可合理認為這份子集具備作為快速評測集的潛力。
