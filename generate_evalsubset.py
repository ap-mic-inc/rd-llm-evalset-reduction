# pip install pandas numpy scipy statsmodels

import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import spearmanr, ttest_1samp, wilcoxon
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


# =========================================================
# 參數設定
# =========================================================
INPUT_CSV = "tmmlu_model_results_merged_complete_rows_only.csv"

TARGET_N = 300                  # 目標縮小後試卷題數
N_TRIALS = 3000                 # 搜尋最佳 subset 的抽樣次數
BOOTSTRAP_N = 2000              # bootstrap 次數
ACCURACY_TOLERANCE = 0.03       # 可接受的單模型誤差門檻，例如 ±3%
GLOBAL_SEED = 42

# 輸出根目錄（可自行修改）
OUTPUT_BASE_DIR = Path(".")


# =========================================================
# 工具函式：建立輸出資料夾
# =========================================================
def prepare_output_dirs(base_dir: Path):
    """
    建立輸出資料夾結構。
    唯一不放進子資料夾的是最終 300 題 CSV。
    """
    dirs = {
        "base": base_dir,
        "item_analysis": base_dir / "item_analysis",
        "model_comparison": base_dir / "model_comparison",
        "search_trials": base_dir / "search_trials",
        "bootstrap_results": base_dir / "bootstrap_results",
        "significance_tests": base_dir / "significance_tests",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


# =========================================================
# 工具函式：自動辨識模型欄位
# =========================================================
def detect_model_columns(df: pd.DataFrame):
    """
    自動找出模型欄位。
    規則：
    - 排除 metadata 欄位
    - 剩餘欄位若唯一值幾乎都屬於 {0,1}，視為模型欄位
    """
    meta_cols = {"question_id", "nsubject", "question", "correct_answer"}
    candidate_cols = [c for c in df.columns if c not in meta_cols]

    model_cols = []
    for c in candidate_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        unique_vals = set(vals.dropna().unique().tolist())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            model_cols.append(c)

    return model_cols


# =========================================================
# 工具函式：計算題目統計
# =========================================================
def compute_item_stats(df: pd.DataFrame, model_cols):
    """
    計算每一題的基本統計：
    - p_value: 平均答對率
    - q_value = 1 - p_value
    - item_variance = p(1-p)
    - difficulty_group: hard / medium / easy
    """
    out = df.copy()
    out["p_value"] = out[model_cols].mean(axis=1)
    out["q_value"] = 1 - out["p_value"]
    out["item_variance"] = out["p_value"] * out["q_value"]

    def difficulty_group(p):
        if p < 0.3:
            return "hard"
        elif p < 0.7:
            return "medium"
        else:
            return "easy"

    out["difficulty_group"] = out["p_value"].apply(difficulty_group)
    return out


# =========================================================
# 工具函式：依母體難度比例分配題數
# =========================================================
def allocate_counts_from_population(df: pd.DataFrame, total_n: int):
    """
    根據母體 hard / medium / easy 的比例，換算出 subset 要抽幾題。
    保證三層題數總和 = total_n。
    """
    proportions = df["difficulty_group"].value_counts(normalize=True).to_dict()

    raw = {
        "hard": total_n * proportions.get("hard", 0),
        "medium": total_n * proportions.get("medium", 0),
        "easy": total_n * proportions.get("easy", 0),
    }

    floor_vals = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total_n - sum(floor_vals.values())

    frac_parts = sorted(
        [(k, raw[k] - floor_vals[k]) for k in raw.keys()],
        key=lambda x: x[1],
        reverse=True
    )

    for i in range(remainder):
        floor_vals[frac_parts[i][0]] += 1

    return floor_vals, proportions


# =========================================================
# 工具函式：分層抽樣
# =========================================================
def stratified_sample(df, counts_by_stratum, rng):
    """
    根據 difficulty_group 分層抽樣。
    """
    sampled_parts = []

    for stratum, n_take in counts_by_stratum.items():
        pool = df[df["difficulty_group"] == stratum]
        if len(pool) < n_take:
            raise ValueError(
                f"分層 {stratum} 題目不足，需要 {n_take} 題，但只有 {len(pool)} 題"
            )

        idx = rng.choice(pool.index.values, size=n_take, replace=False)
        sampled_parts.append(df.loc[idx])

    subset = pd.concat(sampled_parts, axis=0).sample(
        frac=1,
        random_state=int(rng.integers(1_000_000_000))
    )
    subset = subset.reset_index(drop=True)
    return subset


# =========================================================
# 工具函式：評估某份 subset 與母體的接近程度
# =========================================================
def evaluate_subset(full_df, subset_df, model_cols):
    """
    比較 subset 與 full set 對各模型分數的保留程度。
    回傳：
    - metrics: 綜合指標
    - detail_df: 每個模型的比較細節
    """
    full_acc = full_df[model_cols].mean(axis=0)
    subset_acc = subset_df[model_cols].mean(axis=0)

    diff = subset_acc - full_acc
    abs_diff = diff.abs()

    mae = abs_diff.mean()
    max_abs_diff = abs_diff.max()

    rank_corr, _ = spearmanr(full_acc.values, subset_acc.values)
    if pd.isna(rank_corr):
        rank_corr = -1.0

    n_outside_tol = int((abs_diff > ACCURACY_TOLERANCE).sum())

    detail_df = pd.DataFrame({
        "model": model_cols,
        "full_accuracy": full_acc.values,
        "subset_accuracy": subset_acc.values,
        "diff": diff.values,
        "abs_diff": abs_diff.values
    })

    # Wilson 95% CI for subset accuracy
    lower_list = []
    upper_list = []
    contains_full_list = []

    n_subset = len(subset_df)

    for _, row in detail_df.iterrows():
        k = int(round(row["subset_accuracy"] * n_subset))
        low, high = proportion_confint(count=k, nobs=n_subset, alpha=0.05, method="wilson")
        lower_list.append(low)
        upper_list.append(high)
        contains_full_list.append(low <= row["full_accuracy"] <= high)

    detail_df["subset_ci_lower_95"] = lower_list
    detail_df["subset_ci_upper_95"] = upper_list
    detail_df["full_in_subset_ci"] = contains_full_list

    metrics = {
        "mae": float(mae),
        "max_abs_diff": float(max_abs_diff),
        "spearman_rank_corr": float(rank_corr),
        "n_outside_tolerance": n_outside_tol,
        "n_full_acc_inside_subset_ci": int(detail_df["full_in_subset_ci"].sum())
    }

    return metrics, detail_df


# =========================================================
# 工具函式：綜合評分，選最佳 subset
# =========================================================
def score_candidate(metrics):
    """
    將多個評估指標組合成一個分數，越小越好。
    權重可自行調整。
    """
    score = (
        metrics["mae"] * 100
        + metrics["max_abs_diff"] * 50
        + metrics["n_outside_tolerance"] * 2
        + (1 - metrics["spearman_rank_corr"]) * 20
    )
    return score


# =========================================================
# Bootstrap：對最佳 subset 做穩定性驗證
# =========================================================
def bootstrap_subset_stability(full_df, subset_df, model_cols, n_bootstrap=2000, seed=42):
    """
    針對最佳 subset 進行 bootstrap 重抽驗證。

    做法：
    - 從 300 題 subset 中有放回抽樣 300 題
    - 每次都重新計算：
      1. 每個模型的 accuracy
      2. 與母體 accuracy 的 MAE
      3. 模型排名 Spearman correlation

    輸出：
    - per_model_bootstrap_df：每個模型在 bootstrap 下的分布摘要
    - overall_bootstrap_summary_df：整體指標（MAE / rank corr）的 bootstrap 摘要
    - raw_bootstrap_acc_df：每次 bootstrap 的模型 accuracy 明細
    - overall_bootstrap_df：每次 bootstrap 的整體指標明細
    """
    rng = np.random.default_rng(seed)

    full_acc = full_df[model_cols].mean(axis=0)

    bootstrap_acc_records = []
    overall_records = []

    n_subset = len(subset_df)

    for b in range(n_bootstrap):
        sample_idx = rng.choice(subset_df.index.values, size=n_subset, replace=True)
        boot_df = subset_df.loc[sample_idx]

        boot_acc = boot_df[model_cols].mean(axis=0)
        diff = boot_acc - full_acc
        abs_diff = diff.abs()

        mae = abs_diff.mean()
        max_abs_diff = abs_diff.max()

        rank_corr, _ = spearmanr(full_acc.values, boot_acc.values)
        if pd.isna(rank_corr):
            rank_corr = -1.0

        record = {"bootstrap_iter": b}
        for m in model_cols:
            record[m] = boot_acc[m]
        bootstrap_acc_records.append(record)

        overall_records.append({
            "bootstrap_iter": b,
            "mae": float(mae),
            "max_abs_diff": float(max_abs_diff),
            "spearman_rank_corr": float(rank_corr)
        })

    raw_bootstrap_acc_df = pd.DataFrame(bootstrap_acc_records)
    overall_bootstrap_df = pd.DataFrame(overall_records)

    # 每個模型的 bootstrap 摘要
    per_model_rows = []
    for m in model_cols:
        vals = raw_bootstrap_acc_df[m].values
        lower = np.percentile(vals, 2.5)
        upper = np.percentile(vals, 97.5)
        tol_hit_rate = np.mean(np.abs(vals - full_acc[m]) <= ACCURACY_TOLERANCE)

        per_model_rows.append({
            "model": m,
            "full_accuracy": float(full_acc[m]),
            "bootstrap_mean": float(np.mean(vals)),
            "bootstrap_std": float(np.std(vals, ddof=1)),
            "bootstrap_ci_lower_95": float(lower),
            "bootstrap_ci_upper_95": float(upper),
            "full_in_bootstrap_ci": bool(lower <= full_acc[m] <= upper),
            "within_tolerance_rate": float(tol_hit_rate)
        })

    per_model_bootstrap_df = pd.DataFrame(per_model_rows)

    # 整體 bootstrap 摘要
    overall_summary = {
        "metric": ["mae", "max_abs_diff", "spearman_rank_corr"],
        "mean": [
            overall_bootstrap_df["mae"].mean(),
            overall_bootstrap_df["max_abs_diff"].mean(),
            overall_bootstrap_df["spearman_rank_corr"].mean()
        ],
        "std": [
            overall_bootstrap_df["mae"].std(ddof=1),
            overall_bootstrap_df["max_abs_diff"].std(ddof=1),
            overall_bootstrap_df["spearman_rank_corr"].std(ddof=1)
        ],
        "ci_lower_95": [
            np.percentile(overall_bootstrap_df["mae"], 2.5),
            np.percentile(overall_bootstrap_df["max_abs_diff"], 2.5),
            np.percentile(overall_bootstrap_df["spearman_rank_corr"], 2.5)
        ],
        "ci_upper_95": [
            np.percentile(overall_bootstrap_df["mae"], 97.5),
            np.percentile(overall_bootstrap_df["max_abs_diff"], 97.5),
            np.percentile(overall_bootstrap_df["spearman_rank_corr"], 97.5)
        ]
    }
    overall_bootstrap_summary_df = pd.DataFrame(overall_summary)

    return per_model_bootstrap_df, overall_bootstrap_summary_df, raw_bootstrap_acc_df, overall_bootstrap_df


# =========================================================
# 統計檢定：整體 across-model 差異
# =========================================================
def across_model_significance_tests(model_compare_df):
    """
    對各模型的 diff = subset_accuracy - full_accuracy 做整體檢定。

    這裡把「每個模型的 accuracy 差值」視為一筆觀測值：
    - one-sample t-test：檢查平均差值是否顯著不等於 0
    - Wilcoxon signed-rank：非參數版本
    """
    diffs = model_compare_df["diff"].values

    t_stat, t_p = ttest_1samp(diffs, popmean=0.0)

    try:
        w_stat, w_p = wilcoxon(diffs)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    result_df = pd.DataFrame({
        "test": ["one_sample_t_test", "wilcoxon_signed_rank"],
        "statistic": [t_stat, w_stat],
        "p_value": [t_p, w_p],
        "interpretation_note": [
            "檢查 across models 的平均差值是否顯著不為 0",
            "非參數檢定，檢查 across models 的差值中位數是否顯著不為 0"
        ]
    })

    return result_df


# =========================================================
# 統計檢定：每個模型 individually 的比例差異檢定
# =========================================================
def per_model_proportion_tests(full_df, subset_df, model_cols):
    """
    對每個模型 individually 比較：
    full accuracy vs subset accuracy

    使用兩比例 z-test。
    注意：
    - 這裡把 full set 與 subset 視為兩組比例做近似比較
    - 但嚴格來說 subset 是 full 的子集，不完全獨立
    - 因此這個檢定建議視為 exploratory / approximate
    """
    results = []

    n_full = len(full_df)
    n_subset = len(subset_df)

    for m in model_cols:
        full_correct = int(full_df[m].sum())
        subset_correct = int(subset_df[m].sum())

        count = np.array([subset_correct, full_correct])
        nobs = np.array([n_subset, n_full])

        try:
            z_stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")
        except Exception:
            z_stat, p_value = np.nan, np.nan

        results.append({
            "model": m,
            "subset_correct": subset_correct,
            "subset_n": n_subset,
            "subset_accuracy": subset_correct / n_subset,
            "full_correct": full_correct,
            "full_n": n_full,
            "full_accuracy": full_correct / n_full,
            "z_stat": z_stat,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05) if pd.notna(p_value) else np.nan,
            "note": "近似兩比例 z-test；由於 subset 為 full 子集，結果請審慎解讀"
        })

    return pd.DataFrame(results)


# =========================================================
# 主程式
# =========================================================
def main():
    rng_global = np.random.default_rng(GLOBAL_SEED)

    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到檔案: {INPUT_CSV}")

    # 建立輸出資料夾
    output_dirs = prepare_output_dirs(OUTPUT_BASE_DIR)

    # -----------------------------------------------------
    # 1. 讀取資料
    # -----------------------------------------------------
    df = pd.read_csv(input_path)
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    model_cols = detect_model_columns(df)
    if not model_cols:
        raise ValueError("找不到模型欄位，請檢查 CSV 結構")

    print(f"偵測到 {len(model_cols)} 個模型欄位")
    print(model_cols)

    required_cols = ["question_id", "question", "correct_answer"] + model_cols
    if "nsubject" in df.columns:
        required_cols.append("nsubject")

    clean_df = df.dropna(subset=required_cols).copy()
    clean_df = compute_item_stats(clean_df, model_cols)

    print(f"\n可用題數: {len(clean_df)}")

    # -----------------------------------------------------
    # 2. 顯示母體難度分布，並依母體比例決定抽樣數
    # -----------------------------------------------------
    strata_counts = clean_df["difficulty_group"].value_counts().to_dict()
    print("\n母體難度分布（題數）:")
    for k in ["hard", "medium", "easy"]:
        print(f"{k}: {strata_counts.get(k, 0)}")

    counts_by_stratum, population_proportions = allocate_counts_from_population(clean_df, TARGET_N)

    print("\n母體難度分布（比例）:")
    for k in ["hard", "medium", "easy"]:
        print(f"{k}: {population_proportions.get(k, 0):.4f}")

    print("\n依母體比例換算後的目標抽樣數:")
    print(counts_by_stratum)

    # -----------------------------------------------------
    # 3. 重複抽樣，搜尋最佳 subset
    # -----------------------------------------------------
    best_score = float("inf")
    best_subset = None
    best_metrics = None
    best_detail = None
    trial_records = []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(rng_global.integers(1_000_000_000))

        try:
            subset_df = stratified_sample(clean_df, counts_by_stratum, rng)
            metrics, detail_df = evaluate_subset(clean_df, subset_df, model_cols)
            score = score_candidate(metrics)

            trial_records.append({
                "trial": trial,
                "score": score,
                **metrics
            })

            if score < best_score:
                best_score = score
                best_subset = subset_df.copy()
                best_metrics = metrics
                best_detail = detail_df.copy()

        except Exception as e:
            trial_records.append({
                "trial": trial,
                "score": np.inf,
                "error": str(e)
            })

    if best_subset is None:
        raise RuntimeError("沒有成功找到可用 subset，請檢查資料與參數設定")

    # -----------------------------------------------------
    # 4. Bootstrap 重抽驗證穩定性
    # -----------------------------------------------------
    print("\n開始進行 bootstrap 穩定性分析...")
    (
        per_model_bootstrap_df,
        overall_bootstrap_summary_df,
        raw_bootstrap_acc_df,
        overall_bootstrap_df
    ) = bootstrap_subset_stability(
        full_df=clean_df,
        subset_df=best_subset,
        model_cols=model_cols,
        n_bootstrap=BOOTSTRAP_N,
        seed=GLOBAL_SEED
    )

    # -----------------------------------------------------
    # 5. 統計檢定
    # -----------------------------------------------------
    print("開始進行統計檢定...")

    across_model_tests_df = across_model_significance_tests(best_detail)
    per_model_tests_df = per_model_proportion_tests(clean_df, best_subset, model_cols)

    # -----------------------------------------------------
    # 6. 定義輸出路徑
    # -----------------------------------------------------
    output_subset = output_dirs["base"] / "reduced_eval_set_300_population_ratio.csv"

    output_item_stats = output_dirs["item_analysis"] / "item_statistics_with_difficulty.csv"

    output_model_compare = output_dirs["model_comparison"] / "model_accuracy_comparison_population_ratio.csv"

    output_trials = output_dirs["search_trials"] / "subset_search_trials_population_ratio.csv"

    output_bootstrap_model = output_dirs["bootstrap_results"] / "bootstrap_per_model_stability.csv"
    output_bootstrap_overall = output_dirs["bootstrap_results"] / "bootstrap_overall_summary.csv"
    output_bootstrap_raw = output_dirs["bootstrap_results"] / "bootstrap_raw_model_accuracies.csv"
    output_bootstrap_overall_raw = output_dirs["bootstrap_results"] / "bootstrap_raw_overall_metrics.csv"

    output_across_model_tests = output_dirs["significance_tests"] / "significance_tests_across_models.csv"
    output_per_model_tests = output_dirs["significance_tests"] / "significance_tests_per_model.csv"

    # -----------------------------------------------------
    # 7. 輸出結果檔案
    # -----------------------------------------------------
    # 輸出最佳 subset 題目（唯一不放資料夾）
    subset_cols = ["question_id"]
    if "nsubject" in best_subset.columns:
        subset_cols.append("nsubject")
    subset_cols += ["question", "correct_answer", "p_value", "item_variance", "difficulty_group"] + model_cols
    best_subset[subset_cols].to_csv(output_subset, index=False, encoding="utf-8-sig")

    # 輸出全體題目統計
    item_stats_cols = ["question_id"]
    if "nsubject" in clean_df.columns:
        item_stats_cols.append("nsubject")
    item_stats_cols += ["question", "correct_answer", "p_value", "q_value", "item_variance", "difficulty_group"]
    clean_df[item_stats_cols].to_csv(output_item_stats, index=False, encoding="utf-8-sig")

    # 輸出模型比較
    best_detail.to_csv(output_model_compare, index=False, encoding="utf-8-sig")

    # 輸出 trial 搜尋過程
    pd.DataFrame(trial_records).to_csv(output_trials, index=False, encoding="utf-8-sig")

    # 輸出 bootstrap
    per_model_bootstrap_df.to_csv(output_bootstrap_model, index=False, encoding="utf-8-sig")
    overall_bootstrap_summary_df.to_csv(output_bootstrap_overall, index=False, encoding="utf-8-sig")
    raw_bootstrap_acc_df.to_csv(output_bootstrap_raw, index=False, encoding="utf-8-sig")
    overall_bootstrap_df.to_csv(output_bootstrap_overall_raw, index=False, encoding="utf-8-sig")

    # 輸出統計檢定
    across_model_tests_df.to_csv(output_across_model_tests, index=False, encoding="utf-8-sig")
    per_model_tests_df.to_csv(output_per_model_tests, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # 8. 終端摘要輸出
    # -----------------------------------------------------
    print("\n===================================")
    print("最佳 300 題 subset 已找到")
    print("===================================")
    print(f"最佳分數: {best_score:.6f}")
    print(f"MAE: {best_metrics['mae']:.4f}")
    print(f"Max abs diff: {best_metrics['max_abs_diff']:.4f}")
    print(f"Spearman rank corr: {best_metrics['spearman_rank_corr']:.4f}")
    print(f"超過容忍誤差 {ACCURACY_TOLERANCE:.2%} 的模型數: {best_metrics['n_outside_tolerance']}")
    print(f"full accuracy 落在 subset 95% CI 內的模型數: {best_metrics['n_full_acc_inside_subset_ci']} / {len(model_cols)}")

    print("\nBootstrap 整體摘要:")
    print(overall_bootstrap_summary_df)

    print("\nAcross-model significance tests:")
    print(across_model_tests_df)

    sig_count = int(per_model_tests_df["significant_at_0.05"].fillna(False).sum())
    print(f"\nPer-model approximate proportion tests 中，顯著差異模型數: {sig_count} / {len(model_cols)}")

    print("\n輸出檔案:")
    print(f"- {output_subset}")
    print(f"- {output_item_stats}")
    print(f"- {output_model_compare}")
    print(f"- {output_trials}")
    print(f"- {output_bootstrap_model}")
    print(f"- {output_bootstrap_overall}")
    print(f"- {output_bootstrap_raw}")
    print(f"- {output_bootstrap_overall_raw}")
    print(f"- {output_across_model_tests}")
    print(f"- {output_per_model_tests}")


if __name__ == "__main__":
    main()