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