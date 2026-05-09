"""Data collator: chat template + label masking (loss only on assistant tokens)."""

from __future__ import annotations

from typing import Any

import torch


class SFTQACollator:
    """Apply chat template, mask prompt tokens in labels, pad to ``max_seq_length``.

    Uses ``processing_class`` (tokenizer) to build sequences so it works with any
    model's chat template without needing a hard-coded response template string.
    """

    def __init__(
        self,
        processing_class: Any,
        max_seq_length: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.processing_class = processing_class
        self.max_seq_length = max_seq_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token = self.processing_class.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feat in features:
            messages = feat["messages"]
            prompt = feat["prompt"]

            full_ids = self.processing_class.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                **self.chat_template_kwargs,
            )
            prompt_ids = self.processing_class.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )

            orig_prompt_len = len(prompt_ids)
            orig_full_len = len(full_ids)
            if orig_full_len > self.max_seq_length:
                offset = orig_full_len - self.max_seq_length
                input_ids = full_ids[offset:]
                prompt_len = max(0, orig_prompt_len - offset)
            else:
                input_ids = list(full_ids)
                prompt_len = orig_prompt_len

            labels = list(input_ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        pad_id = int(self.processing_class.pad_token_id)
        max_len = min(
            self.max_seq_length,
            max(len(x) for x in batch_input_ids) if batch_input_ids else 0,
        )

        padded_input: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for ids, lab in zip(batch_input_ids, batch_labels, strict=True):
            pad_amt = max_len - len(ids)
            padded_input.append(ids + [pad_id] * pad_amt)
            padded_labels.append(lab + [-100] * pad_amt)
            attention_mask.append([1] * len(ids) + [0] * pad_amt)

        return {
            "input_ids": torch.tensor(padded_input, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
