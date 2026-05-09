"""Training arguments for supervised QA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

from trl import SFTConfig


@dataclass
class SFTQAConfig(SFTConfig):
    """Extends TRL ``SFTConfig`` with ROUGE-eval and multilingual logging fields.

    Inherits from ``SFTConfig`` (which in turn extends ``TrainingArguments``) so
    all standard TRL SFT fields (``max_length``, ``packing``,
    ``dataset_text_field``, ``shuffle_dataset``, etc.) are available alongside
    the QA-specific ones.

    Note: TRL 1.x uses ``max_length`` (not ``max_seq_length``); the default
    here is raised to 2048 (TRL's own default is 1024).
    ``SFTQACollator`` reads ``args.max_length``.
    """

    max_length: int = 2048
    rouge_eval_steps: int = 50
    rouge_eval_num_samples: int = 50
    rouge_eval_max_new_tokens: int = 256
    rouge_eval_batch_size: int = 8
    log_multilingual_rouge: bool = True
