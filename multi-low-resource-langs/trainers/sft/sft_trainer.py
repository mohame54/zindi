"""Supervised fine-tuning Trainer with ROUGE + per-language logging."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Optional

import torch
from transformers import TrainerCallback
from trl import SFTTrainer

from utils.metrics import calculate_rouge_score

from trainers.sft.collator import SFTQACollator
from trainers.sft.config import SFTQAConfig


def _print_rouge_report(
    overall: dict[str, float],
    per_lang: dict[str, dict[str, float]],
    step: int,
) -> None:
    print(f"\n=== ROUGE eval (step {step}) ===")
    print(f"{'metric':<14} {'value':>10}")
    print("-" * 26)
    for k, v in sorted(overall.items()):
        print(f"{k:<14} {v:>10.4f}")
    if per_lang:
        print("-" * 26)
        print("Per-language (expected_lang):")
        for lang in sorted(per_lang.keys()):
            row = per_lang[lang]
            print(
                f"  {lang}: rouge1_f1={row['rouge1_f1']:.4f} "
                f"rougeL_f1={row['rougeL_f1']:.4f} score={row['score']:.4f} (n={int(row['count'])})"
            )
    print("-" * 26)


class RougeEvalCallback(TrainerCallback):
    """Generation-based ROUGE on a fixed cadence; logs to W&B via ``trainer.log``."""

    def __init__(
        self,
        processing_class: Any,
        rouge_raw_dataset: Any,
        rouge_eval_steps: int,
        rouge_eval_num_samples: int,
        rouge_eval_max_new_tokens: int,
        log_multilingual_rouge: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.processing_class = processing_class
        self.rouge_raw_dataset = rouge_raw_dataset
        self.rouge_eval_steps = rouge_eval_steps
        self.rouge_eval_num_samples = rouge_eval_num_samples
        self.rouge_eval_max_new_tokens = rouge_eval_max_new_tokens
        self.log_multilingual_rouge = log_multilingual_rouge
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.trainer: Optional[SFTTrainer] = None

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None:
            return control
        if self.rouge_eval_steps <= 0:
            return control
        if state.global_step % self.rouge_eval_steps != 0:
            return control

        ds = self.rouge_raw_dataset
        n = len(ds)
        if n == 0:
            return control

        k = min(self.rouge_eval_num_samples, n)
        indices = random.sample(range(n), k=k)

        model = self.trainer.model
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        tok = self.processing_class
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        pad_id = int(tok.pad_token_id)

        sum_r1 = 0.0
        sum_rl = 0.0
        sum_score = 0.0
        per_lang_scores: dict[str, list[dict[str, float]]] = defaultdict(list)

        with torch.no_grad():
            for idx in indices:
                row = ds[idx]
                prompt = row["prompt"]
                answer = row.get("answer") or ""
                exp_lang = row.get("expected_lang")

                prompt_ids = tok.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(input_ids)
                gen_out = model.generate(
                    input_ids,
                    attention_mask=attn,
                    max_new_tokens=self.rouge_eval_max_new_tokens,
                    pad_token_id=pad_id,
                    do_sample=False,
                )
                new_tokens = gen_out[0, input_ids.shape[1] :]
                pred = tok.decode(new_tokens, skip_special_tokens=True).strip()

                scores = calculate_rouge_score(str(answer), pred)
                sum_r1 += float(scores["rouge1_f1"])
                sum_rl += float(scores["rougeL_f1"])
                sum_score += float(scores["score"])

                if self.log_multilingual_rouge and exp_lang and str(exp_lang).strip():
                    per_lang_scores[str(exp_lang)].append(
                        {
                            "rouge1_f1": float(scores["rouge1_f1"]),
                            "rougeL_f1": float(scores["rougeL_f1"]),
                            "score": float(scores["score"]),
                        }
                    )

        if was_training:
            model.train()

        inv_k = 1.0 / float(k)
        overall = {
            "rouge/rouge1_f1": sum_r1 * inv_k,
            "rouge/rougeL_f1": sum_rl * inv_k,
            "rouge/score": sum_score * inv_k,
        }

        log_payload: dict[str, float] = dict(overall)
        per_lang_avg: dict[str, dict[str, float]] = {}

        if self.log_multilingual_rouge and per_lang_scores:
            for lang, items in per_lang_scores.items():
                c = len(items)
                if c == 0:
                    continue
                r1 = sum(x["rouge1_f1"] for x in items) / c
                rl = sum(x["rougeL_f1"] for x in items) / c
                sc = sum(x["score"] for x in items) / c
                per_lang_avg[lang] = {
                    "rouge1_f1": r1,
                    "rougeL_f1": rl,
                    "score": sc,
                    "count": float(c),
                }
                log_payload[f"rouge/{lang}/rouge1_f1"] = r1
                log_payload[f"rouge/{lang}/rougeL_f1"] = rl
                log_payload[f"rouge/{lang}/score"] = sc

        self.trainer.log(log_payload)

        overall_flat = {
            "rouge1_f1": overall["rouge/rouge1_f1"],
            "rougeL_f1": overall["rouge/rougeL_f1"],
            "score": overall["rouge/score"],
        }
        per_lang_report = {
            lang: {
                "rouge1_f1": v["rouge1_f1"],
                "rougeL_f1": v["rougeL_f1"],
                "score": v["score"],
                "count": v["count"],
            }
            for lang, v in per_lang_avg.items()
        }
        _print_rouge_report(overall_flat, per_lang_report, int(state.global_step))
        return control


class SFTQATrainer(SFTTrainer):
    """TRL ``SFTTrainer`` with prompt-masked collator and ROUGE eval callback.

    Inherits from ``trl.SFTTrainer`` for full TRL compatibility (peft_config
    handling, packing, dataset_text_field, etc.).  TRL's own dataset
    tokenization is skipped via ``_prepare_dataset``; ``SFTQACollator`` owns
    all tokenization and prompt masking per batch.
    """

    def __init__(
        self,
        model: Any,
        args: SFTQAConfig,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        processing_class: Any | None = None,
        peft_config: Any | None = None,
        callbacks: list[TrainerCallback] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        collator = SFTQACollator(
            processing_class,
            max_seq_length=args.max_length or 2048,
            chat_template_kwargs=chat_template_kwargs,
        )

        rouge_source = eval_dataset if eval_dataset is not None else train_dataset
        rouge_cb = RougeEvalCallback(
            processing_class=processing_class,
            rouge_raw_dataset=rouge_source,
            rouge_eval_steps=args.rouge_eval_steps,
            rouge_eval_num_samples=args.rouge_eval_num_samples,
            rouge_eval_max_new_tokens=args.rouge_eval_max_new_tokens,
            log_multilingual_rouge=args.log_multilingual_rouge,
            chat_template_kwargs=chat_template_kwargs,
        )

        merged_callbacks = list(callbacks or []) + [rouge_cb]

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
            data_collator=collator,
            callbacks=merged_callbacks,
        )
        rouge_cb.trainer = self

    def _prepare_dataset(
        self,
        dataset: Any,
        processing_class: Any,
        args: Any,
        packing: bool,
        formatting_func: Any,
        dataset_name: str,
    ) -> Any:
        """Skip TRL's automatic chat-template tokenization.

        ``SFTQACollator`` applies the chat template and prompt masking per
        batch, so the dataset must arrive at the collator with raw ``messages``
        and ``prompt`` columns intact.
        """
        return dataset
