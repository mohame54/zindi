from __future__ import annotations
import os
import sys
import math
import torch
from collections import defaultdict
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from utils.peft import load_lora_model, load_qlora_model
from utils.metrics import calculate_rouge_score
# Local SDFT trainer (add trainers/sdft to path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trainers", "sdft"))
from sdft import DistilTrainer
from config import DistilConfig
from langs import InferenceModel


def _add_feedback_to_teacher_prompt(teacher_prompt: list[dict], feedback: str) -> list[dict]:
    patched_prompt = [dict(msg) for msg in teacher_prompt]
    for msg in reversed(patched_prompt):
        if msg["role"] == "user":
            msg["content"] = msg["content"] + feedback
            break
    return patched_prompt


class LangAwareDistilTrainer(DistilTrainer):
    """
    Extends DistilTrainer to patch teacher_prompts online when the student's
    completion language mismatches expected_lang.

    Also tracks language-drift statistics every generation step and logs them
    to wandb (or any other reporter configured in DistilConfig.report_to):

      lang_drift/mismatch_rate        – fraction of completions in wrong language
      lang_drift/mismatch_count       – raw mismatch count in the logging window
      lang_drift/total_count          – total completions in the logging window
      lang_drift/expected/{lang}      – mismatches per expected language
      lang_drift/detected/{lang}      – which wrong languages were generated
    """

    def __init__(self, *args, lang_model: InferenceModel, rouge_score_threshold: float = 0.6, **kwargs):
        kwargs.pop("rouge_score_threshold", None)
        super().__init__(*args, **kwargs)
        self.lang_model = lang_model
        self._rouge_score_threshold = rouge_score_threshold
        self._lang_stats: dict[str, list] = defaultdict(list)
        self.rouge_stats: dict[str, list] = defaultdict(list)

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        generation_prompts = (
            [x["teacher_prompt"] for x in inputs]
            if self.generate_from_teacher
            else prompts
        )

        (
            _prompt_ids,
            completion_ids_list,
            _num_items,
            _logprobs,
            _fwd_kwargs,
        ) = self._generate(generation_prompts, images=None)

        completion_texts = [
            self.processing_class.decode(ids, skip_special_tokens=True)
            for ids in completion_ids_list
        ]
        detected_langs, _ = self.lang_model.predict_compiled(completion_texts)

        # ── Track language-drift stats ────────────────────────────────────────
        n_total = 0
        n_mismatch = 0
        per_expected: dict[str, int] = defaultdict(int)
        per_detected: dict[str, int] = defaultdict(int)

        patched_inputs = []
        rouge_prefix = f"{'eval_' if mode == 'eval' else ''}rouge"
        for sample, detected, generated in zip(inputs, detected_langs, completion_texts):
            expected = sample.get("expected_lang")
            answer = sample.get("answer")
            rouge_scores = calculate_rouge_score(answer, generated)
            patched = dict(sample)
            for metric_name, metric_value in rouge_scores.items():
                self.rouge_stats[f"{rouge_prefix}/{metric_name}"].append(metric_value)
            if expected:
                n_total += 1
                if detected != expected:
                    n_mismatch += 1
                    per_expected[expected] += 1
                    per_detected[detected] += 1
                    feedback = (
                        f"\n\nIMPORTANT: The previous answer was incorrectly in "
                        f"'{detected}'. NEVER answer in '{detected}'. "
                        f"You MUST answer in '{expected}' only."
                    )
                    patched["teacher_prompt"] = _add_feedback_to_teacher_prompt(
                        patched["teacher_prompt"], feedback
                    )

            if rouge_scores["score"] < self._rouge_score_threshold:
                feedback = (
                    f"\nIMPORTANT: Include only the answer and what's related to it. "
                    f"You MUST answer in answer like in the reference answer"
                )
                patched["teacher_prompt"] = _add_feedback_to_teacher_prompt(
                    patched["teacher_prompt"], feedback
                )

            patched_inputs.append(patched)

        if n_total > 0:
            prefix = f"{'eval_' if mode == 'eval' else ''}lang_drift"
            self._lang_stats[f"{prefix}/mismatch_count"].append(n_mismatch)
            self._lang_stats[f"{prefix}/total_count"].append(n_total)
            self._lang_stats[f"{prefix}/mismatch_rate"].append(n_mismatch / n_total)
            for lang, count in per_expected.items():
                self._lang_stats[f"{prefix}/expected/{lang}"].append(count)
            for lang, count in per_detected.items():
                self._lang_stats[f"{prefix}/detected/{lang}"].append(count)

        return super()._generate_and_score_completions(patched_inputs)

    def log(self, logs: dict, start_time=None) -> None:
        # Fold accumulated lang-drift stats into logs as averages, then clear.
        if self._lang_stats:
            for key, values in self._lang_stats.items():
                logs[key] = sum(values) / len(values)
            self._lang_stats.clear()
        if self.rouge_stats:
            for key, values in self.rouge_stats.items():
                logs[key] = sum(values) / len(values)
            self.rouge_stats.clear()
        super().log(logs, start_time)


class SaveEveryNEpochsCallback(TrainerCallback):

    def __init__(self, every_n_epochs: int):
        if every_n_epochs < 0:
            raise ValueError("every_n_epochs must be >= 0 (0 disables this callback)")
        self.every_n_epochs = every_n_epochs
        self._trainer = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.every_n_epochs == 0 or self._trainer is None:
            return control
        completed = max(1, int(math.floor(float(state.epoch) + 1e-6)))
        if completed % self.every_n_epochs != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{completed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self._trainer.save_model(ckpt_dir)
        proc = getattr(self._trainer, "processing_class", None)
        if proc is not None and hasattr(proc, "save_pretrained"):
            proc.save_pretrained(ckpt_dir)
        return control


def run_training(
    *,
    model_name: str,
    train_dataset: Any,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    # Models (optional — loaded from model_name if omitted)
    student: Optional[Any] = None,
    teacher: Optional[Any] = None,
    peft_config: Optional[Any] = None,
    # PEFT / precision
    use_peft: bool = False,
    qlora: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
    # Lang feedback
    lang_model: Optional[InferenceModel] = None,
    dynamic_lang_feedback: bool = False,
    rouge_score_threshold: float = 0.6,
    # Optim / schedule
    seed: int = 42,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    lr_scheduler_type: str = "cosine",
    max_grad_norm: float = 1.0,
    bf16: bool = True,
    fp16: bool = False,
    # Batch / generation
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
    num_generations: int = 1,
    num_iterations: int = 1,
    max_prompt_length: int = 1024,
    max_completion_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: Optional[int] = 20,
    repetition_penalty: float = 1.1,
    generation_kwargs: Optional[dict[str, Any]] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    # Objective
    alpha: float = 0.0,
    beta: float = 0.0,
    generate_from_teacher: bool = False,
    # Teacher sync
    sync_ref_model: bool = False,
    ref_model_sync_steps: int = 1,
    ref_model_mixup_alpha: float = 0.01,
    # Logging / saving
    num_train_epochs: int = 1,
    logging_steps: int = 1,
    save_steps: int = 200,
    save_every_n_epochs: int = 0,
    report_to: str = "wandb",
    log_completions: bool = True,
    num_loss_tokens_to_skip: int = 3,
    shuffle_dataset: bool = True,
    # Callbacks
    extra_callbacks: Optional[list] = None,
) -> DistilTrainer:
    """
    Build DistilConfig, attach optional epoch-save callback, run trainer.train().

    If student/teacher are None, they are loaded from ``model_name`` (teacher always
    bfloat16; student uses LoRA/QLoRA per flags). ``peft_config`` is ignored when
    student is passed explicitly unless you also pass a pre-built student that
    is already a PeftModel — prefer passing ``student=None`` and using use_peft/qlora.
    """
    if qlora:
        use_peft = True

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if teacher is None:
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

    if student is None:
        if qlora:
            student, peft_config = load_qlora_model(
                model_name,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                use_double_quant=use_double_quant,
            )
        elif use_peft:
            student, peft_config = load_lora_model(
                model_name,
            )
        else:
            student = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
            peft_config = None

    config = DistilConfig(
        output_dir=output_dir,
        seed=seed,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        num_iterations=num_iterations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        generation_kwargs=generation_kwargs,
        chat_template_kwargs=chat_template_kwargs,
        alpha=alpha,
        beta=beta,
        generate_from_teacher=generate_from_teacher,
        sync_ref_model=sync_ref_model,
        ref_model_sync_steps=ref_model_sync_steps,
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        log_completions=log_completions,
        num_loss_tokens_to_skip=num_loss_tokens_to_skip,
        shuffle_dataset=shuffle_dataset,
    )

    callbacks: list = list(extra_callbacks or [])
    epoch_cb: Optional[SaveEveryNEpochsCallback] = None
    if save_every_n_epochs > 0:
        epoch_cb = SaveEveryNEpochsCallback(save_every_n_epochs)
        callbacks.append(epoch_cb)

    TrainerClass = LangAwareDistilTrainer if dynamic_lang_feedback else DistilTrainer
    trainer_kwargs: dict[str, Any] = dict(
        model=student,
        ref_model=teacher,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks if callbacks else None,
        rouge_score_threshold=rouge_score_threshold,
    )
    if dynamic_lang_feedback:
        if lang_model is None:
            raise ValueError("dynamic_lang_feedback=True requires lang_model")
        trainer_kwargs["lang_model"] = lang_model

    trainer = TrainerClass(**trainer_kwargs)
    if epoch_cb is not None:
        epoch_cb.set_trainer(trainer)

    trainer.train()
    return trainer


if __name__ == "__main__":
    raise SystemExit(
        "Training CLI lives in main.py. Run: python main.py --help"
    )
