from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import SFTTrainer

from utils.metrics import calculate_rouge_score

from trainers.sft.collator import SFTQACollator
from trainers.sft.config import SFTQAConfig
from trainers.sft.dro import LangDROScheduler


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
    """Generation-based ROUGE on a fixed cadence; logs to W&B via ``trainer.log``.

    If a ``LangDROScheduler`` is attached, per-language ROUGE scores are fed
    into it after every eval so DRO weights are updated automatically.
    """

    def __init__(
        self,
        processing_class: Any,
        rouge_raw_dataset: Any,
        rouge_eval_steps: int,
        rouge_eval_num_samples: int,
        rouge_eval_max_new_tokens: int,
        log_multilingual_rouge: bool,
        rouge_eval_batch_size: int = 8,
        chat_template_kwargs: dict[str, Any] | None = None,
        dro_scheduler: Optional[LangDROScheduler] = None,
    ) -> None:
        self.processing_class = processing_class
        self.rouge_raw_dataset = rouge_raw_dataset
        self.rouge_eval_steps = rouge_eval_steps
        self.rouge_eval_num_samples = rouge_eval_num_samples
        self.rouge_eval_max_new_tokens = rouge_eval_max_new_tokens
        self.rouge_eval_batch_size = rouge_eval_batch_size
        self.log_multilingual_rouge = log_multilingual_rouge
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.dro_scheduler = dro_scheduler
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

        # ── Stratified sampling by language ──────────────────────────────────
        # Build per-language index lists so every language gets a fair quota
        # instead of random sampling that can leave rare languages out entirely.
        lang_to_indices: dict[str, list[int]] = defaultdict(list)
        has_lang = False
        for i in range(n):
            lang = ds[i].get("expected_lang")
            if lang and str(lang).strip():
                lang_to_indices[str(lang).strip()].append(i)
                has_lang = True
            else:
                lang_to_indices["_unknown"].append(i)

        k = min(self.rouge_eval_num_samples, n)
        if has_lang and len(lang_to_indices) > 1:
            num_langs = len(lang_to_indices)
            per_lang_quota = max(1, k // num_langs)
            indices: list[int] = []
            for lang_idxs in lang_to_indices.values():
                take = min(per_lang_quota, len(lang_idxs))
                indices.extend(random.sample(lang_idxs, k=take))
            # Top up from the remaining pool if we're under quota
            if len(indices) < k:
                picked = set(indices)
                remaining = [i for i in range(n) if i not in picked]
                extra = min(k - len(indices), len(remaining))
                indices.extend(random.sample(remaining, k=extra))
        else:
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

        rows = [ds[idx] for idx in indices]
        actual_k = len(rows)
        batch_size = max(1, self.rouge_eval_batch_size)

        with torch.no_grad():
            for batch_start in range(0, actual_k, batch_size):
                batch_rows = rows[batch_start : batch_start + batch_size]

                batch_prompt_ids = [
                    tok.apply_chat_template(
                        row["prompt"],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                        **self.chat_template_kwargs,
                    )
                    for row in batch_rows
                ]

                # Left-pad all prompts to the same length for batched generation
                max_prompt_len = max(len(ids) for ids in batch_prompt_ids)
                padded_ids = [
                    [pad_id] * (max_prompt_len - len(ids)) + ids
                    for ids in batch_prompt_ids
                ]
                attn_masks = [
                    [0] * (max_prompt_len - len(ids)) + [1] * len(ids)
                    for ids in batch_prompt_ids
                ]

                input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
                attn = torch.tensor(attn_masks, dtype=torch.long, device=device)

                gen_out = model.generate(
                    input_ids,
                    attention_mask=attn,
                    max_new_tokens=self.rouge_eval_max_new_tokens,
                    pad_token_id=pad_id,
                    do_sample=False,
                    repetition_penalty=1.1,
                )

                for i, row in enumerate(batch_rows):
                    answer = row.get("answer") or ""
                    exp_lang = row.get("expected_lang")
                    new_tokens = gen_out[i, max_prompt_len:]
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

        inv_k = 1.0 / float(actual_k)
        overall = {
            "rouge/rouge1_f1": sum_r1 * inv_k,
            "rouge/rougeL_f1": sum_rl * inv_k,
            "rouge/score": sum_score * inv_k,
        }

        log_payload: dict[str, float] = dict(overall)
        per_lang_avg: dict[str, dict[str, float]] = {}
        per_lang_rouge_scores: dict[str, float] = {}

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
                per_lang_rouge_scores[lang] = sc
                log_payload[f"rouge/{lang}/rouge1_f1"] = r1
                log_payload[f"rouge/{lang}/rougeL_f1"] = rl
                log_payload[f"rouge/{lang}/score"] = sc

        # ── Update DRO scheduler with ROUGE signal and log weights ───────────
        if self.dro_scheduler is not None and per_lang_rouge_scores:
            self.dro_scheduler.update_from_rouge(per_lang_rouge_scores)
            for lang, w in self.dro_scheduler.weights().items():
                log_payload[f"dro/weight/{lang}"] = w
            for lang, s in self.dro_scheduler.rouge_scores().items():
                log_payload[f"dro/ema_rouge/{lang}"] = s
            for lang, s in self.dro_scheduler.loss_scores().items():
                log_payload[f"dro/ema_loss_score/{lang}"] = s
            for lang, s in self.dro_scheduler.blended_scores().items():
                log_payload[f"dro/blended_score/{lang}"] = s

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

        # ── Print DRO weight table if active ──────────────────────────────────
        if self.dro_scheduler is not None:
            print("\n=== DRO weights (after ROUGE update) ===")
            print(f"{'Lang':<8} {'LossScore':>10} {'ROUGE EMA':>10} {'Blended':>9} {'Weight':>8}")
            print("-" * 50)
            wts = self.dro_scheduler.weights()
            ls = self.dro_scheduler.loss_scores()
            rs = self.dro_scheduler.rouge_scores()
            bl = self.dro_scheduler.blended_scores()
            for lang in sorted(wts):
                print(
                    f"{lang:<8} {ls.get(lang, 0.0):>10.4f} {rs.get(lang, 0.0):>10.4f}"
                    f" {bl.get(lang, 0.0):>9.4f} {wts[lang]:>8.3f}"
                )
            print("-" * 50)

        return control


class SFTQATrainer(SFTTrainer):
    """TRL ``SFTTrainer`` with prompt-masked collator, ROUGE eval callback,
    and optional adaptive Group DRO loss weighting.

    When ``dro_scheduler`` is supplied, ``compute_loss`` applies per-sample
    loss scaling based on the language weight from the scheduler.  The
    ``RougeEvalCallback`` feeds ROUGE scores back into the scheduler after
    every ROUGE eval step, closing the adaptive feedback loop.

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
        dro_scheduler: Optional[LangDROScheduler] = None,
    ) -> None:
        self._dro_scheduler = dro_scheduler

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
            rouge_eval_batch_size=args.rouge_eval_batch_size,
            chat_template_kwargs=chat_template_kwargs,
            dro_scheduler=dro_scheduler,
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

    # ── Adaptive Group DRO loss ───────────────────────────────────────────────

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        # Pop the language list before forwarding to the model — it's not a
        # tensor and the model doesn't expect it.
        langs: list[str | None] | None = inputs.pop("expected_lang", None)

        labels = inputs.get("labels")

        if self._dro_scheduler is None or langs is None or labels is None:
            # DRO disabled or language info absent — standard CE loss.
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # ── DRO path: per-sample weighted cross-entropy ───────────────────────
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # Shift: predict token[t+1] from token[t]
        shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, T-1)

        # Token-level CE with no reduction — keep (B, T-1) shape
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)

        # Per-sample mean over non-masked tokens
        valid = (shift_labels != -100).float()
        denom = valid.sum(dim=-1).clamp(min=1.0)
        per_sample_loss = (token_loss * valid).sum(dim=-1) / denom  # (B,)

        # Look up DRO weight for each sample's language
        weights = torch.tensor(
            [
                self._dro_scheduler.get_weight(str(lang)) if lang else 1.0
                for lang in langs
            ],
            device=per_sample_loss.device,
            dtype=per_sample_loss.dtype,
        )

        loss = (per_sample_loss * weights).mean()

        # Feed per-language mean loss back into the scheduler (loss signal).
        # This runs every step — no generation cost, high-frequency update.
        per_lang_losses: dict[str, list[float]] = {}
        detached = per_sample_loss.detach().float()
        for lang, sample_loss in zip(langs, detached.tolist()):
            if lang:
                key = str(lang)
                per_lang_losses.setdefault(key, []).append(sample_loss)
        if per_lang_losses:
            self._dro_scheduler.update_from_loss(
                {lang: sum(vs) / len(vs) for lang, vs in per_lang_losses.items()}
            )

        return (loss, outputs) if return_outputs else loss

    # ── TRL dataset preparation passthrough ──────────────────────────────────

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
