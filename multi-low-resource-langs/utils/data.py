import numpy as np
import pandas as pd
from datasets import Dataset
import json


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided reference answer to answer the user's question. "
    "Do not repeat the reference answer word-for-word, but ensure all facts are correct. "
    "Provide ONLY the answer text. No thinking, no intro, no outro."
)


TEACHER_TEMPLATE = """\
Below is a question and a reference answer. Use the reference to write a final response.

Question: {question}

Reference Answer: {golden_answer}

Instruction: Write the answer in {expected_lang}. Do not repeat the question or the reference headers.

Answer:"""


def create_system_chains(
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
    teacher_template: str = TEACHER_TEMPLATE,
    golden_answer: str = None,
    expected_lang: str = None
):
    if golden_answer is not None and expected_lang is not None:
        return  [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": teacher_template.format(
                    question=question, golden_answer=golden_answer, expected_lang=expected_lang
                )},
            ]
    return [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": question},
    ]


def load_hf_sdft_data_from_csv(
    path: str,
    system_prompt: str = SYSTEM_PROMPT,
    teacher_template: str = TEACHER_TEMPLATE,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    old_cols = df.columns.tolist()
    df['expected_lang']  = df['subset'].str.split('_').str[0]
    df['prompt'] = df['input'].apply(lambda x: create_system_chains(x, system_prompt, teacher_template))
    df['teacher_prompt'] = df.apply(
        lambda x: create_system_chains(
            question=x['input'],
            system_prompt=system_prompt,
            teacher_template=teacher_template,
            golden_answer=x['output'],
            expected_lang=x['expected_lang']
        ),
        axis=1
    )
    df['answer'] = df['output']
    df.drop(columns=old_cols, inplace=True)
    return Dataset.from_pandas(df)


def load_json(fp:str) -> dict:
    with open(fp, 'r') as f:
        return json.load(f)


def tokenize_dataset_stats(
    data,
    tokenizer,
    question_col: str = "input",
    answer_col: str = "output",
    print_report: bool = True,
) -> dict:
    """Tokenize every question and answer in the dataset and return length statistics.

    Args:
        data: CSV file path, dict of lists, pandas DataFrame, or HuggingFace Dataset.
              Must contain `question_col` and `answer_col` columns.
        tokenizer: Any tokenizer with an ``encode`` method (e.g. a HuggingFace tokenizer).
        question_col: Column name for the question / input text.
        answer_col: Column name for the reference answer / output text.
        print_report: If True, print a human-readable summary to stdout.

    Returns:
        A dict with keys ``"questions"`` and ``"answers"``, each holding a stats sub-dict
        with keys: count, min, max, mean, median, p90, p95, p99, total_tokens.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = data.to_pandas()

    def _lengths(col):
        return [
            len(tokenizer.encode(str(text), add_special_tokens=False))
            for text in df[col]
        ]

    def _stats(lengths: list) -> dict:
        arr = np.array(lengths, dtype=np.int64)
        return {
            "count":        int(len(arr)),
            "min":          int(arr.min()),
            "max":          int(arr.max()),
            "mean":         round(float(arr.mean()), 2),
            "median":       round(float(np.median(arr)), 2),
            "p90":          round(float(np.percentile(arr, 90)), 2),
            "p95":          round(float(np.percentile(arr, 95)), 2),
            "p99":          round(float(np.percentile(arr, 99)), 2),
            "total_tokens": int(arr.sum()),
        }

    q_stats = _stats(_lengths(question_col))
    a_stats = _stats(_lengths(answer_col))
    report = {"questions": q_stats, "answers": a_stats}

    if print_report:
        _print_stats_report(report, question_col, answer_col)

    return report


def _print_stats_report(report: dict, question_col: str, answer_col: str) -> None:
    col_width = 14
    header = f"{'Stat':<{col_width}} {'Questions':>{col_width}} {'Answers':>{col_width}}"
    separator = "-" * len(header)
    rows = ["count", "min", "max", "mean", "median", "p90", "p95", "p99", "total_tokens"]

    print(f"\n=== Token-length statistics  ({question_col!r} / {answer_col!r}) ===")
    print(separator)
    print(header)
    print(separator)
    for row in rows:
        q_val = report["questions"][row]
        a_val = report["answers"][row]
        print(f"{row:<{col_width}} {str(q_val):>{col_width}} {str(a_val):>{col_width}}")
    print(separator)