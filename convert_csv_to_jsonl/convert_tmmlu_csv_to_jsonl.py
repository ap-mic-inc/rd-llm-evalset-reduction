import argparse
import json
import re
from pathlib import Path

import pandas as pd


OPTION_PATTERN = re.compile(
    r"^(?P<stem>.*?)\nA:\s*(?P<A>.*?)\nB:\s*(?P<B>.*?)\nC:\s*(?P<C>.*?)\nD:\s*(?P<D>.*?)\s*$",
    re.DOTALL,
)


def parse_question_block(text: str):
    if pd.isna(text):
        raise ValueError("question 欄位為空值")
    text = str(text).strip()
    match = OPTION_PATTERN.match(text)
    if not match:
        raise ValueError(f"無法解析題目與選項，原始內容如下：\n{text}")
    return {
        "question": match.group("stem").strip(),
        "A": match.group("A").strip(),
        "B": match.group("B").strip(),
        "C": match.group("C").strip(),
        "D": match.group("D").strip(),
    }


def convert_csv_to_jsonl(input_csv: str, output_jsonl: str):
    df = pd.read_csv(input_csv)

    required_columns = {"question_id", "question", "correct_answer"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要欄位: {sorted(missing)}")

    records = []
    for new_id, (_, row) in enumerate(df.iterrows()):
        parsed = parse_question_block(row["question"])
        answer = str(row["correct_answer"]).strip()

        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"第 {new_id} 筆資料的 correct_answer 不是 A/B/C/D：{answer}"
            )

        record = {
            "id": new_id,
            "original_question_id": int(row["question_id"]),
            "question": parsed["question"],
            "A": parsed["A"],
            "B": parsed["B"],
            "C": parsed["C"],
            "D": parsed["D"],
            "answer": answer,
        }
        records.append(record)

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} records")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="將代表性 300 題 CSV 轉成 TMMLU+ 評測框架使用的 jsonl 格式"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="輸入 CSV 路徑",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="輸出 jsonl 路徑",
    )
    args = parser.parse_args()

    convert_csv_to_jsonl(args.input_csv, args.output_jsonl)


if __name__ == "__main__":
    main()