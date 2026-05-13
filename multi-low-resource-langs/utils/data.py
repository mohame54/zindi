import re
import numpy as np
import pandas as pd
from datasets import Dataset
import json
import os
import gdown


# Shared by student and teacher so the system-role tokens match; only the user message differs
# (question-only vs question + reference). That reduces an extra distribution shift from different
# system prefixes when DistilTrainer compares logits on the same completion.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's message. "
    "when you can say the same thing more naturally. "
    "Answer in the language of the user's question. "
    "Reply with ONLY the final answer text. No thinking, no intro, no outro."
)

# Maps the two-letter country code found in the subset ID to a full country name.
COUNTRY_MAP: dict[str, str] = {
    "Uga": "Uganda",
    "Gha": "Ghana",
    "Eth": "Ethiopia",
    "Ken": "Kenya",
}

STUDENT_TEMPLATE = """\
Answer in {language} as spoken in {country}
Question:
{question}"""

TEACHER_TEMPLATE = """\
Answer in {language} as spoken in {country}
Question:
{question}

Reference answer (use it to ground your response):
{golden_answer}"""


def strip_qwen_thinking_tokens(text: str) -> str:
    # Remove optional "assistant" prefix, then any <think>...</think> block (and its content)
    text = re.sub(r'^assistant\s*<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return text.strip()

def create_question(
    question: str,
    tokenizer,
    language: str = "",
    country: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    student_template: str = STUDENT_TEMPLATE,
    tokenize:bool = True,
    return_tensors: bool = "pt"
) -> list[dict]:

    kwargs = {
        "tokenize":tokenize,
        "return_tensors":return_tensors
    }
    kwargs = {k:v for k,v in kwargs.items() if v is not None}
    content = (
        student_template.format(question=question, language=language, country=country)
        if language or country
        else question
    )
    inputs =  tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        enable_thinking = False,
        **kwargs
    )
    return inputs

def batch_create_question(
    questions: list[str],
    tokenizer,
    language: list[str] | str = "",
    country: list[str] | str = "",
    system_prompt: str = SYSTEM_PROMPT,
    student_template: str = STUDENT_TEMPLATE,
) -> dict:
    # Normalise language and country to per-question lists
    if isinstance(language, str):
        languages = [language] * len(questions)
    else:
        languages = language

    if isinstance(country, str):
        countries = [country] * len(questions)
    else:
        countries = country

    # Apply chat template per conversation → plain strings first
    formatted_texts = []
    for qs, lang, ctry in zip(questions, languages, countries):
        text = create_question(
            qs,
            tokenizer,
            language=lang,
            country=ctry,
            system_prompt=system_prompt,
            student_template=student_template,
            return_tensors=None,
            tokenize=False
        )
        formatted_texts.append(text)
    # Batch-tokenize with left-padding (required for generation)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    return inputs   


def create_student_messages(
    question: str,
    *,
    language: str = "",
    country: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    student_template: str = STUDENT_TEMPLATE,
) -> list[dict]:
    content = (
        student_template.format(question=question, language=language, country=country)
        if language or country
        else question
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def create_teacher_messages(
    question: str,
    *,
    golden_answer: str | None = None,
    language: str = "",
    country: str = "",
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
                language=language,
                country=country,
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
    student_template: str = STUDENT_TEMPLATE,
    teacher_template: str = TEACHER_TEMPLATE,
    train: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    old_cols = df.columns.tolist()
    df['expected_lang'] = df['subset'].str.split('_').str[0]
    df['expected_country'] = df['subset'].str.split('_').str[1].map(COUNTRY_MAP).fillna("")

    # SDFT: student prompt = question only; teacher_prompt = question + gold answer.
    # Separate list objects so online patching of teacher_prompt never mutates prompt.
    df['prompt'] = df.apply(
        lambda r: create_student_messages(
            r['input'],
            language=r['expected_lang'],
            country=r['expected_country'],
            system_prompt=system_prompt,
            student_template=student_template,
        ),
        axis=1,
    )
    df['teacher_prompt'] = df.apply(
        lambda r: create_teacher_messages(
            r['input'],
            golden_answer=r['output'],
            language=r['expected_lang'],
            country=r['expected_country'],
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


NO_THINK_PREFIX = "<think>\n\n</think>\n\n"


def prepare_sft_dataset(dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
    def _add_messages(sample: dict) -> dict:
        # Prefix the assistant turn with an empty think block so the model
        # learns to output <think>\n\n</think> in non-thinking mode, matching
        # what enable_thinking=False pre-fills at inference time.
        sample["messages"] = sample["prompt"] + [
            {"role": "assistant", "content": NO_THINK_PREFIX + sample["answer"]},
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


def download_gdown_file(file_id: str, output_path: str, quiet: bool = False) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=quiet)
    return output_path