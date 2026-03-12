rd-llm-evalset-reduction為大型語言模型基準測試建立具代表性的縮小評測集，並結合分層抽樣與統計驗證方法。Build representative reduced eval sets for LLM benchmarks with stratified sampling and statistical validation.📖 Overview 專案總覽大型評測集通常能提供較穩定的模型比較結果，但完整執行一次 benchmark 往往成本高、耗時久。本專案的目標是從完整評測集中自動挑選出一份較小、但仍具代表性的子集，並盡可能保留：各模型的整體分數模型彼此的相對排名評測結果的穩定性核心運作流程：系統會先根據各題在模型群上的平均答對率估計題目難度，將題目分為 hard、medium、easy 三層。接著，依據母體難度的真實比例進行分層抽樣，從完整題庫中搜尋最具代表性的 300 題候選子集。最後，透過 Bootstrap 重抽分析與統計檢定，確保這份子集能高度近似完整評測集的結果。✨ Features 主要特色自動化資料處理：從 Hugging Face twinkle-ai/datasets 抓取多模型評測紀錄（eval-logs-and-scores），並自動整理成「題目 × 模型」的統一評分矩陣。精準難度分層：根據題目平均答對率自動估算難度，並依母體真實比例進行分層抽樣。最佳化搜尋：透過多次抽樣比對，搜尋出最能還原完整題庫分數與排名的最佳子集。嚴謹統計驗證：內建 Bootstrap 穩定性分析與模型層級的統計檢定，確保抽樣品質。結構化輸出：產生分類清晰、便於追蹤的數據與報表檔案。📂 Repository Structure 專案結構Plaintext.
├── build_eval_results_table.py  # 抓取並整併評測結果表格
├── generate_evalsubset.py       # 執行分析、抽樣、搜尋與驗證的主程式
├── README.md
└── ...
🛠 Installation & Setup 安裝指引建議使用 Python 3.10 或以上版本。Linux / macOSBashpython -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scipy statsmodels datasets huggingface_hub pyarrow
Windows PowerShellPowerShellpython -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scipy statsmodels datasets huggingface_hub pyarrow
Data Source 資料來源本專案的評測資料來自 Hugging Face: twinkle-ai/datasets。這些資料集包含了不同模型針對 TMMLU+ 題目的作答紀錄與詳細評分結果。🚀 Quick Start 快速開始Step 1. 建立完整評測矩陣 (Build merged table)先從 Hugging Face 抓取多模型的紀錄，並整理成統一的 CSV 表格：Bashpython build_eval_results_table.py
執行後產出：tmmlu_model_results_merged.csv (若有過濾完整列，則可能產出 tmmlu_model_results_merged_complete_rows_only.csv)Step 2. 生成縮小版評測集 (Generate reduced eval subset)使用主程式進行抽樣與驗證：Bashpython generate_evalsubset.py
主程式背景執行流程：Detect model columns：自動辨識模型作答結果欄位 (0/1)。Compute item statistics：計算每一題的平均答對率 (p) 與變異數 (p(1-p))，並將題目分為 hard (p < 0.3)、medium (0.3 <= p < 0.7)、easy (p >= 0.7)。Stratified sampling：根據母體難度比例，分配 300 題子集中各層應抽取的題數。Search best subset：重複抽樣多次，找出最能保留完整題庫分數與排名的題組。Bootstrap analysis：對選出的子集進行重抽樣，檢查結果穩定性。Statistical validation：進行模型層級的統計檢定，作為輔助分析。📊 Output Files 解析輸出檔案程式執行完畢後，會生成以下目錄結構與檔案：Plaintext├── reduced_eval_set_300_population_ratio.csv          # 最終選出的 300 題縮小版評測集 (核心產出)
├── item_analysis/
│   └── item_statistics_with_difficulty.csv            # 完整題庫各題統計資訊 (p值、變異數、難度分組)
├── model_comparison/
│   └── model_accuracy_comparison_population_ratio.csv # 各模型在完整題庫 vs 子集上的表現差異
├── search_trials/
│   └── subset_search_trials_population_ratio.csv      # 歷次候選子集的搜尋與表現紀錄
├── bootstrap_results/                                 # Bootstrap 穩定性分析報表
│   ├── bootstrap_per_model_stability.csv
│   ├── bootstrap_overall_summary.csv
│   └── ...
└── significance_tests/                                # 統計檢定結果報表
    ├── significance_tests_across_models.csv
    └── significance_tests_per_model.csv
📈 Interpretation Guidelines 數據解讀指南1. 如何判斷縮小版評測集的好壞？請優先檢查以下五個關鍵指標：指標名稱理想狀態意義說明MAE (平均絕對誤差)越小越好子集與完整題庫相比，平均每個模型分數的誤差 (例：0.0078 代表差 0.78%)。Max abs diff越小越好所有模型中，最大偏差值為多少。Spearman rank corr接近 1子集與完整題庫的模型排名相關性。接近 1 代表相對排名幾乎完整保留。n_outside_tolerance越小越好 (最好為 0)誤差超過預設容忍值 (±3%) 的模型數量。Full accuracy in CI越接近總模型數越好完整題庫的分數，有多少落在子集 Bootstrap 所估計的 95% 信賴區間內。2. 如何解讀 Bootstrap 穩定性分析？Bootstrap 的目的不是產生新考卷，而是檢查「這 300 題子集若有微小的抽樣波動，整體結果是否依然穩定」。bootstrap_per_model_stability.csvbootstrap_mean：越接近完整題庫分數 (full_accuracy) 越好。bootstrap_std：數值越小代表越穩定。within_tolerance_rate：越接近 1 越好。full_in_bootstrap_ci：若為 True，表示與母體結果相容。bootstrap_overall_summary.csv重點觀察整體 MAE、Max abs diff (越小越好) 及 Spearman 相關係數 (接近 1)。3. 如何解讀統計檢定 (Significance Tests)？significance_tests_across_models.csv (整體系統性偏差)：若 p-value > 0.05，代表未觀察到顯著的整體偏移；若 p-value < 0.05，則可能存在系統性偏差。significance_tests_per_model.csv (單一模型檢定)：significant_at_0.05 = False 代表該模型在子集與完整題庫間未有顯著差異。(註：因單一模型檢定屬近似參考，建議搭配其他指標綜合評估。)💡 Practical Rule of Thumb 實用準則如果您的最終輸出大致符合以下條件，恭喜！這是一份品質極佳的縮小版評測集：[x] MAE 很小且 Max abs diff 不大[x] Spearman rank corr 非常接近 1[x] n_outside_tolerance = 0[x] 多數模型的 full_in_subset_ci = True[x] Bootstrap 下的 within_tolerance_rate 高[x] Across-model 檢定未顯示顯著系統性偏差📌 溫馨提醒：本專案的核心目標是在「測試成本/速度」與「代表性」之間取得平衡。統計檢定結果應與誤差指標、排名相關性綜合解讀。若未來 Benchmark 題庫或參測模型集合發生變動，建議重新執行本流程以確保子集品質。
