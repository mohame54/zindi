"""Training arguments for supervised QA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class SFTQAConfig(SFTConfig):
    """Extends TRL ``SFTConfig`` with ROUGE-eval, multilingual logging, and Group DRO fields.

    Inherits from ``SFTConfig`` (which in turn extends ``TrainingArguments``) so
    all standard TRL SFT fields (``max_length``, ``packing``,
    ``dataset_text_field``, ``shuffle_dataset``, etc.) are available alongside
    the QA-specific ones.

    Note: TRL 1.x uses ``max_length`` (not ``max_seq_length``); the default
    here is raised to 2048 (TRL's own default is 1024).
    ``SFTQACollator`` reads ``args.max_length``.
    """

    # ── ROUGE eval ────────────────────────────────────────────────────────────
    max_length: int = 2048
    rouge_eval_steps: int = 50
    rouge_eval_num_samples: int = 50
    rouge_eval_max_new_tokens: int = 512
    rouge_eval_batch_size: int = 8
    log_multilingual_rouge: bool = True

    # ── Adaptive Group DRO ───────────────────────────────────────────────────
    # Set dro_eta > 0 to enable adaptive per-language loss weighting.
    # Languages with lower ROUGE scores receive proportionally higher weights.
    dro_eta: float = 0.0           # sensitivity; 0 disables DRO
    dro_loss_beta: float = 0.9    # EMA smoothing for training loss (updated every step)
    dro_rouge_beta: float = 0.5   # EMA smoothing for ROUGE scores (updated every rouge_eval_steps)
    dro_rouge_weight: float = 0.5 # blend: 0.0=loss only, 1.0=ROUGE only, 0.5=equal blend
    dro_min_weight: float = 0.2   # weight floor — no language starved below this
    dro_max_weight: float = 5.0   # weight ceiling — no language dominates above this
    dro_init_score: float = 0.3   # assumed score before first data (neutral prior)
    dro_languages: list = field(default_factory=list)  # seed known languages; auto-discovered if empty
