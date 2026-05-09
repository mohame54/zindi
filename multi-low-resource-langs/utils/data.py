import numpy as np
import pandas as pd
from datasets import Dataset
import json


# Shared by student and teacher so the system-role tokens match; only the user message differs
# (question-only vs question + reference). That reduces an extra distribution shift from different
# system prefixes when DistilTrainer compares logits on the same completion.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's message. "
    "If it includes a reference answer, use it to ground your response; do not copy it verbatim "
    "when you can say the same thing more naturally. "
    "Answer in the same language as the user's question. "
    "Reply with ONLY the final answer text. No thinking, no intro, no outro."
)

TEACHER_TEMPLATE = """\
Question:
{question}

Reference answer (teacher context):
{golden_answer}

Write the final answer only, in the same language as the question."""


def create_student_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def create_teacher_messages(
    question: str,
    *,
    golden_answer: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    teacher_template: str = TEACHER_TEMPLATE,
) -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
      
    ]
    if golden_answer is not None:
        messages.append({
            "role": "user",
            "content": teacher_template.format(
                question=question,
                golden_answer=golden_answer,
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": question,
        })
    return messages


def load_hf_sdft_data_from_csv(
    path: str,
    system_prompt: str = SYSTEM_PROMPT,
    teacher_template: str = TEACHER_TEMPLATE,
    train: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    old_cols = df.columns.tolist()
    df['expected_lang'] = df['subset'].str.split('_').str[0]

    # SDFT: student prompt = question only; teacher_prompt = question + gold answer.
    # Separate list objects so online patching of teacher_prompt never mutates prompt.
    df['prompt'] = df.apply(
        lambda r: create_student_messages(r['input'], system_prompt=system_prompt),
        axis=1,
    )
    df['teacher_prompt'] = df.apply(
        lambda r: create_teacher_messages(
            r['input'],
            golden_answer=r['output'],
            system_prompt=system_prompt,
            teacher_template=teacher_template,
        ),
        axis=1,
    )
    df['answer'] = df['output']
    if train:
        df.drop([2814, 2929, 19431], inplace=True)
    df.drop(columns=old_cols, inplace=True)
    df.reset_index(drop=True,inplace=True)
    return Dataset.from_pandas(df)


def prepare_sft_dataset(dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
    def _add_messages(sample: dict) -> dict:
        sample["messages"] = sample["prompt"] + [
            {"role": "assistant", "content": sample["answer"]},
        ]
        return sample
    def length(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False))
    dataset = dataset.map(_add_messages)
    dataset = dataset.map(lambda x: {"length": length(x['messages'])})
    dataset = dataset.filter(lambda x: x["length"] <= max_length)
    return dataset


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