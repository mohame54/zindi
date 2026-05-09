"""Built-in reward callables for TRL ``GRPOTrainer`` (see ``reward_funcs``)."""

from __future__ import annotations

from typing import Any

from langs import InferenceModel
from utils.metrics import calculate_rouge_score


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def rouge_reward(
    prompts: list[Any],
    completions: list[Any],
    completion_ids: list | None = None,
    answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    del prompts, completion_ids, kwargs
    if answer is None:
        raise ValueError("rouge_reward requires an 'answer' column in the dataset")
    rewards: list[float] = []
    for comp, ref in zip(completions, answer, strict=True):
        ref_text = ref if ref is not None else ""
        hyp = _completion_text(comp)
        scores = calculate_rouge_score(ref_text, hyp)
        rewards.append(float(scores["score"]))
    return rewards


def lang_reward(
    prompts: list[Any],
    completions: list[Any],
    completion_ids: list | None = None,
    expected_lang: list[str] | None = None,
    lang_model: InferenceModel | None = None,
    **kwargs: Any,
) -> list[float]:
    if lang_model is None:
        raise ValueError(
            "lang_reward requires lang_model=... (e.g. functools.partial(lang_reward, lang_model=...))"
        )
    texts = [_completion_text(c) for c in completions]
    detected_langs, _ = lang_model.predict_compiled(texts)
    if expected_lang is None:
        return [1.0] * len(completions)
    rewards: list[float] = []
    for det, exp in zip(detected_langs, expected_lang, strict=True):
        if not exp:
            rewards.append(1.0)
        else:
            rewards.append(1.0 if det == exp else 0.0)
    return rewards
