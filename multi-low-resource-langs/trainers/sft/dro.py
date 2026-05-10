"""Adaptive Group DRO weight scheduler for multilingual SFT.

Two complementary signals drive the weights:

1. **Loss signal** (high-frequency, zero-cost)
   Called every training step via ``update_from_loss``.  Tracks a per-language
   EMA of the training loss so weights adjust within each epoch without waiting
   for a generation-based ROUGE eval.

2. **ROUGE signal** (low-frequency, held-out, competition metric)
   Called every ``rouge_eval_steps`` via ``update_from_rouge``.  Recalibrates
   the loss-based EMA toward actual generation quality so long-term drift is
   corrected.  Loss can fall (teacher-forced next-token) while ROUGE stays low
   (free-form generation); this signal catches that gap.

Weight formula (shared, applied to whichever score is active)
-------------------------------------------------------------
    ema_g   ←  (1 - beta) * ema_g  +  beta * new_value_g
    raw_g   =  exp(−eta * ema_g)
    w_g     =  clip( raw_g / mean(raw),  min_weight,  max_weight )

Loss scores are **inverted** before entering the same formula so that high
loss → high raw_g → high weight, mirroring the ROUGE convention where low
score → high weight.

Inversion:  loss_score_g  =  1 / (1 + ema_loss_g)   ∈ (0, 1)

This maps loss ∈ [0, ∞) to (0, 1] monotonically, making it commensurable
with ROUGE scores and allowing a weighted blend of both signals.

Parameters
----------
eta            : sensitivity — larger → more aggressive re-weighting
loss_beta      : EMA smoothing for training loss (updated every step)
rouge_beta     : EMA smoothing for ROUGE (updated every rouge_eval_steps)
rouge_weight   : blend factor — 0.0 = loss only, 1.0 = ROUGE only, 0.5 = equal
min_weight     : weight floor (no language starved below this)
max_weight     : weight ceiling (no language dominates above this)
init_score     : assumed score before any data is seen (neutral prior ≈ 0.3)
"""

from __future__ import annotations

import math
import threading
from typing import Optional


class LangDROScheduler:

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        eta: float = 2.0,
        loss_beta: float = 0.9,
        rouge_beta: float = 0.5,
        rouge_weight: float = 0.5,
        min_weight: float = 0.2,
        max_weight: float = 5.0,
        init_score: float = 0.3,
    ) -> None:
        self.eta = eta
        self.loss_beta = loss_beta
        self.rouge_beta = rouge_beta
        self.rouge_weight = rouge_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.init_score = init_score
        self._lock = threading.Lock()

        # Separate EMA trackers for each signal
        self._ema_loss: dict[str, float] = {}     # raw loss values
        self._ema_rouge: dict[str, float] = {}    # ROUGE scores ∈ [0, 1]
        self._weights: dict[str, float] = {}

        if languages:
            for lang in languages:
                # Initialise loss EMA to the implied loss for init_score:
                # init_score = 1/(1+loss) → loss = 1/init_score - 1
                self._ema_loss[lang] = max(0.0, 1.0 / max(init_score, 1e-6) - 1.0)
                self._ema_rouge[lang] = init_score
            self._recompute()

    # ── public API ────────────────────────────────────────────────────────────

    def update_from_loss(self, per_lang_losses: dict[str, float]) -> None:
        """Ingest per-language mean training losses from the current batch.

        Should be called every training step from ``compute_loss``.
        New languages are auto-registered on first observation.
        """
        with self._lock:
            for lang, loss_val in per_lang_losses.items():
                prev = self._ema_loss.get(lang, max(0.0, 1.0 / max(self.init_score, 1e-6) - 1.0))
                self._ema_loss[lang] = (1.0 - self.loss_beta) * prev + self.loss_beta * float(loss_val)
                # Register in rouge EMA if not yet seen so _recompute sees it
                if lang not in self._ema_rouge:
                    self._ema_rouge[lang] = self.init_score
            self._recompute()

    def update_from_rouge(self, per_lang_scores: dict[str, float]) -> None:
        """Ingest per-language ROUGE scores from a generation-based eval step.

        Should be called every ``rouge_eval_steps`` from ``RougeEvalCallback``.
        New languages are auto-registered on first observation.
        """
        with self._lock:
            for lang, score in per_lang_scores.items():
                prev = self._ema_rouge.get(lang, self.init_score)
                self._ema_rouge[lang] = (1.0 - self.rouge_beta) * prev + self.rouge_beta * float(score)
                # Register in loss EMA if not yet seen
                if lang not in self._ema_loss:
                    init_loss = max(0.0, 1.0 / max(self.init_score, 1e-6) - 1.0)
                    self._ema_loss[lang] = init_loss
            self._recompute()

    # Kept for backwards compatibility with code that calls .update()
    def update(self, per_lang_scores: dict[str, float]) -> None:
        """Alias for ``update_from_rouge``."""
        self.update_from_rouge(per_lang_scores)

    def get_weight(self, lang: str) -> float:
        """Return the current DRO weight for *lang* (default 1.0 if unknown)."""
        with self._lock:
            return self._weights.get(lang, 1.0)

    def weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def loss_scores(self) -> dict[str, float]:
        """Current inverted loss scores ∈ (0,1] (higher = better = lower loss)."""
        with self._lock:
            return {lang: 1.0 / (1.0 + v) for lang, v in self._ema_loss.items()}

    def rouge_scores(self) -> dict[str, float]:
        with self._lock:
            return dict(self._ema_rouge)

    def blended_scores(self) -> dict[str, float]:
        """The blended score that actually drives the weights."""
        with self._lock:
            return self._blend()

    def __repr__(self) -> str:
        w = {k: f"{v:.3f}" for k, v in sorted(self._weights.items())}
        return f"LangDROScheduler(weights={w}, rouge_weight={self.rouge_weight})"

    # ── internals ─────────────────────────────────────────────────────────────

    def _blend(self) -> dict[str, float]:
        """Merge loss and ROUGE signals into a single score per language."""
        all_langs = set(self._ema_loss) | set(self._ema_rouge)
        blended: dict[str, float] = {}
        rw = self.rouge_weight
        for lang in all_langs:
            loss_score = 1.0 / (1.0 + self._ema_loss.get(lang, max(0.0, 1.0 / max(self.init_score, 1e-6) - 1.0)))
            rouge_score = self._ema_rouge.get(lang, self.init_score)
            blended[lang] = (1.0 - rw) * loss_score + rw * rouge_score
        return blended

    def _recompute(self) -> None:
        blended = self._blend()
        if not blended:
            return
        raw = {lang: math.exp(-self.eta * s) for lang, s in blended.items()}
        mean_raw = sum(raw.values()) / len(raw)
        self._weights = {
            lang: max(self.min_weight, min(self.max_weight, r / mean_raw))
            for lang, r in raw.items()
        }
