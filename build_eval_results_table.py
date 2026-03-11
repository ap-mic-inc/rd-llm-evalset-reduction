# pip install datasets huggingface_hub pandas pyarrow

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

ORG_NAME = "twinkle-ai"
OUTPUT_CSV = "tmmlu_model_results_merged.csv"

api = HfApi()

# 1. 抓 twinkle-ai 底下所有 dataset
all_datasets = list(api.list_datasets(author=ORG_NAME))

# 2. 篩出 eval logs 資料集
target_datasets = [
    ds.id for ds in all_datasets
    if ds.id.startswith(f"{ORG_NAME}/") and ds.id.endswith("-eval-logs-and-scores")
]

print(f"找到 {len(target_datasets)} 個目標資料集")
for ds in target_datasets:
    print(ds)

base_df = None
model_columns = []

for dataset_id in target_datasets:
    print(f"\n處理中: {dataset_id}")

    # 3. 模型名稱 = 去掉 org 前綴與尾巴 "-eval-logs-and-scores"
    model_name = dataset_id.split("/", 1)[1].replace("-eval-logs-and-scores", "")

    try:
        # 4. 載入 Hugging Face dataset
        #    若只有 test split，這樣通常可直接取用
        ds = load_dataset(dataset_id, split="test")

        df = ds.to_pandas()

        required_cols = {"question_id", "question", "correct_answer", "is_correct"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"跳過 {dataset_id}，缺少欄位: {missing}")
            continue

        # 5. 保留需要欄位，並將 is_correct 轉成 1/0
        model_df = df[["question_id", "question", "correct_answer", "is_correct"]].copy()
        model_df[model_name] = model_df["is_correct"].astype(int)
        model_df = model_df.drop(columns=["is_correct"])

        # 6. 避免同 question_id 重複
        model_df = model_df.drop_duplicates(subset=["question_id"])

        # 7. 第一份資料當主表，之後逐份 merge
        if base_df is None:
            base_df = model_df
        else:
            # 只帶入 question_id + 模型欄位，避免 question / answer 重複衝突
            base_df = base_df.merge(
                model_df[["question_id", model_name]],
                on="question_id",
                how="outer"
            )

        model_columns.append(model_name)

    except Exception as e:
        print(f"處理 {dataset_id} 時失敗: {e}")

if base_df is None:
    raise RuntimeError("沒有成功讀到任何資料集，請檢查 dataset 名稱或網路連線。")

# 8. 欄位排序
final_columns = ["question_id", "question", "correct_answer"] + model_columns
final_df = base_df[final_columns]

# 9. 依 question_id 排序
final_df = final_df.sort_values(by="question_id").reset_index(drop=True)

# 10. 輸出 CSV
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n完成，已輸出: {OUTPUT_CSV}")
print(final_df.head())