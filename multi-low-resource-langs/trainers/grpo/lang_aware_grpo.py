"""Language-aware wrapper around TRL's ``GRPOTrainer`` (online prompt patching)."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Optional

from trl import GRPOTrainer

from langs import InferenceModel
from utils.metrics import calculate_rouge_score


def _add_feedback_to_prompt(prompt: list[dict], feedback: str) -> list[dict]:
    patched = [dict(msg) for msg in prompt]
    for msg in reversed(patched):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content + feedback
            break
    return patched


class LangAwareGRPOTrainer(GRPOTrainer):
    """
    Runs a preview generation pass, detects language / ROUGE quality, appends
    feedback to the conversational ``prompt`` (last user turn), then delegates
    to TRL's ``GRPOTrainer`` for the real generate + score + loss path.
    """

    def __init__(
        self,
        *args: Any,
        lang_model: InferenceModel,
        rouge_score_threshold: float = 0.6,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("rouge_score_threshold", None)
        super().__init__(*args, **kwargs)
        self.lang_model = lang_model
        self._rouge_score_threshold = rouge_score_threshold
        self._lang_stats: dict[str, list] = defaultdict(list)
        self.rouge_stats: dict[str, list] = defaultdict(list)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]

        gen_out = self._generate(prompts, images=None)
        if not isinstance(gen_out, tuple) or len(gen_out) < 2:
            raise RuntimeError("Unexpected return value from GRPOTrainer._generate")
        completion_ids_list = gen_out[1]

        completion_texts = [
            self.processing_class.decode(ids, skip_special_tokens=True)
            for ids in completion_ids_list
        ]
        detected_langs, _ = self.lang_model.predict_compiled(completion_texts)

        n_total = 0
        n_mismatch = 0
        per_expected: dict[str, int] = defaultdict(int)
        per_detected: dict[str, int] = defaultdict(int)

        patched_inputs: list[dict[str, Any]] = []
        rouge_prefix = f"{'eval_' if mode == 'eval' else ''}rouge"

        for sample, detected, generated in zip(inputs, detected_langs, completion_texts):
            expected = sample.get("expected_lang")
            answer = sample.get("answer")
            rouge_scores = calculate_rouge_score(answer or "", generated)
            patched = dict(sample)
            patched["prompt"] = copy.deepcopy(sample["prompt"])

            for metric_name, metric_value in rouge_scores.items():
                self.rouge_stats[f"{rouge_prefix}/{metric_name}"].append(metric_value)

            if expected:
                n_total += 1
                if detected != expected:
                    n_mismatch += 1
                    per_expected[str(expected)] += 1
                    per_detected[str(detected)] += 1

            patched_inputs.append(patched)

        if n_total > 0:
            prefix = f"{'eval_' if mode == 'eval' else ''}lang_drift"
            self._lang_stats[f"{prefix}/mismatch_count"].append(n_mismatch)
            self._lang_stats[f"{prefix}/mismatch_rate"].append(n_mismatch / n_total)
            for lang, count in per_expected.items():
                self._lang_stats[f"{prefix}/expected/{lang}"].append(count)
            for lang, count in per_detected.items():
                self._lang_stats[f"{prefix}/detected/{lang}"].append(count)

        return super()._generate_and_score_completions(patched_inputs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self._lang_stats:
            for key, values in self._lang_stats.items():
                logs[key] = sum(values) / len(values)
            self._lang_stats.clear()
        if self.rouge_stats:
            for key, values in self.rouge_stats.items():
                logs[key] = sum(values) / len(values)
            self.rouge_stats.clear()
        super().log(logs, start_time)
