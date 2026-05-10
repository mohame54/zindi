from __future__ import annotations

import argparse
import importlib
import json
import os
from functools import partial
from typing import Optional

import torch

from train import run_training
from utils.data import SYSTEM_PROMPT, TEACHER_TEMPLATE, load_hf_sdft_data_from_csv
from utils.hf import download_checkpoint_from_hf, upload_folder_to_hf
from utils import rewards as reward_mod
from langs import InferenceModel


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _json_dict(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def resolve_reward_funcs(names: list[str], lang_model: Optional[InferenceModel]) -> list:
    """Map CLI reward names to callables for ``GRPOTrainer.reward_funcs``."""
    resolved: list = []
    for name in names:
        if name == "rouge":
            resolved.append(reward_mod.rouge_reward)
        elif name == "lang":
            if lang_model is None:
                raise ValueError(
                    "Reward 'lang' requires a language model; pass --lang_model_ckpt"
                )
            resolved.append(partial(reward_mod.lang_reward, lang_model=lang_model))
        elif "." in name:
            module_path, _, attr = name.rpartition(".")
            mod = importlib.import_module(module_path)
            resolved.append(getattr(mod, attr))
        else:
            raise ValueError(
                f"Unknown reward {name!r}. Use rouge, lang, or a dotted path like mypkg.mod.my_fn"
            )
    return resolved


def parse_args():
    p = argparse.ArgumentParser(
        description="Multilingual training: SDFT (DistilTrainer), GRPO (TRL), or SFT (transformers.Trainer)"
    )
    p.add_argument("--model_name", default="Qwen/Qwen3.5-2B")
    p.add_argument("--dataset_path", required=True, help="CSV path for load_hf_sdft_data_from_csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--trainer-type",
        choices=("sdft", "grpo", "sft"),
        default="sdft",
        help="sdft: DistilTrainer; grpo: GRPOTrainer (requires --reward-func); sft: supervised QA fine-tuning",
    )
    p.add_argument(
        "--reward-func",
        nargs="*",
        default=None,
        metavar="NAME",
        help="GRPO only: one or more of rouge, lang, or dotted import paths (e.g. mymod.rewards.my_reward)",
    )

    p.add_argument(
        "--lang_model_ckpt",
        default=None,
        help="Language-classifier checkpoint (required if --dynamic_lang_feedback)",
    )
    p.add_argument("--peft_config_path", default="configs/peft_config.json")
    p.add_argument("--rouge_score_threshold", type=float, default=0.6)
    p.add_argument("--lang_tokenizer", default="tokenizer.json")
    p.add_argument(
        "--lang_compile_bz",
        type=int,
        default=16,
        help="Batch size for lang model torch.compile",
    )

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_bf16", action="store_false", help="Disable bf16 mixed precision")
    p.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=768)
    p.add_argument("--num_generations", type=int, default=1)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument(
        "--generation_kwargs",
        type=_json_dict,
        default=None,
        help='JSON object merged into GenerationConfig, e.g. \'{"num_beams": 1}\'',
    )
    p.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Pass enable_thinking=False to tokenizer.apply_chat_template for supported models",
    )
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Do not pass the default enable_thinking=False chat template override",
    )
    p.add_argument(
        "--chat_template_kwargs",
        type=_json_dict,
        default=None,
        help='JSON object passed to apply_chat_template, e.g. \'{"enable_thinking": false}\'',
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="0=forward KL, 1=reverse KL, (0,1)=Jensen-Shannon",
    )
    p.add_argument("--beta", type=float, default=0.0, help="KL-to-base coefficient (0 disables ref KL term)")
    p.add_argument("--generate_from_teacher", action="store_true")

    p.add_argument("--sync_ref_model", action="store_true")
    p.add_argument("--ref_mixup_alpha", type=float, default=0.01)
    p.add_argument("--ref_model_sync_steps", type=int, default=1)

    p.add_argument("--dynamic_lang_feedback", action="store_true")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument(
        "--eval_steps",
        type=int,
        default=450,
        help="SFT: run eval-loss every N steps (0 = same as save_steps)",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep on disk; oldest are deleted. Set to 0 to keep all.",
    )
    p.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=0,
        help="If > 0, save under output_dir/checkpoint-epoch-{k} every k epochs",
    )
    p.add_argument("--report_to", default="wandb", help="e.g. wandb, none, tensorboard, or comma-separated")
    p.add_argument("--no_log_completions", action="store_true")
    p.add_argument("--num_loss_tokens_to_skip", type=int, default=3)
    p.add_argument("--no_shuffle_dataset", action="store_true")

    # SFT QA (trainer-type=sft)
    p.add_argument(
        "--eval_dataset_path",
        default=None,
        help="Optional CSV for validation (same schema as --dataset_path); enables eval loss + ROUGE on eval split",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="SFT: max tokens per example (prompt + answer) after truncation",
    )
    p.add_argument(
        "--rouge_eval_steps",
        type=int,
        default=100,
        help="SFT: run generation+ROUGE eval every N global steps",
    )
    p.add_argument(
        "--rouge_eval_num_samples",
        type=int,
        default=500,
        help="SFT: number of samples per ROUGE eval",
    )
    p.add_argument(
        "--rouge_eval_max_new_tokens",
        type=int,
        default=512,
        help="SFT: max new tokens when generating for ROUGE eval",
    )
    p.add_argument(
        "--rouge_eval_batch_size",
        type=int,
        default=8,
        help="SFT: number of prompts to generate in parallel during ROUGE eval",
    )
    p.add_argument(
        "--no_log_multilingual_rouge",
        action="store_true",
        help="SFT: disable per-language ROUGE breakdown (expected_lang)",
    )

    # Adaptive Group DRO
    p.add_argument(
        "--dro_eta",
        type=float,
        default=0.0,
        help=(
            "SFT: Group DRO sensitivity (0 disables DRO). "
            "Higher values up-weight poorly-performing languages more aggressively. "
            "Recommended starting value: 2.0"
        ),
    )
    p.add_argument(
        "--dro_loss_beta",
        type=float,
        default=0.9,
        help="SFT: DRO EMA smoothing for per-language training loss (updated every step; 0=no update, 1=no memory)",
    )
    p.add_argument(
        "--dro_rouge_beta",
        type=float,
        default=0.5,
        help="SFT: DRO EMA smoothing for per-language ROUGE scores (updated every rouge_eval_steps)",
    )
    p.add_argument(
        "--dro_rouge_weight",
        type=float,
        default=0.5,
        help=(
            "SFT: blend factor between loss and ROUGE signals for DRO. "
            "0.0 = loss only (high-frequency, zero-cost), "
            "1.0 = ROUGE only (low-frequency, competition metric), "
            "0.5 = equal blend (recommended)"
        ),
    )
    p.add_argument(
        "--dro_min_weight",
        type=float,
        default=0.2,
        help="SFT: DRO minimum per-language loss weight (prevents language starvation)",
    )
    p.add_argument(
        "--dro_max_weight",
        type=float,
        default=5.0,
        help="SFT: DRO maximum per-language loss weight (prevents one language dominating)",
    )
    p.add_argument(
        "--dro_init_score",
        type=float,
        default=0.3,
        help="SFT: assumed ROUGE score per language before the first eval (neutral prior)",
    )
    p.add_argument(
        "--dro_languages",
        nargs="*",
        default=None,
        metavar="LANG",
        help="SFT: seed language codes for DRO (e.g. Eng Aka Lug Amh Swa). Auto-discovered if omitted.",
    )

    # PEFT
    p.add_argument("--use_peft", action="store_true")
    p.add_argument("--qlora", action="store_true", help="4-bit QLoRA student; implies --use_peft")
    p.add_argument("--bnb_4bit_quant_type", default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--no_double_quant", action="store_true", help="Disable BnB double quant for QLoRA")

    # Data templates
    p.add_argument(
        "--system_prompt_file",
        default=None,
        help="UTF-8 file whose contents replace the default system prompt for CSV→dataset",
    )
    p.add_argument(
        "--teacher_template_file",
        default=None,
        help="UTF-8 file whose contents replace TEACHER_TEMPLATE (must include {question} and {golden_answer})",
    )

    # HF Hub
    p.add_argument("--upload_to_hf", action="store_true")
    p.add_argument("--hf_checkpoint_dir", default=None)
    p.add_argument("--hf_local_dir", default="hf_checkpoint")
    p.add_argument(
        "--hf_repo_path",
        default="",
        help=(
            "Sub-folder inside the HF repo to push to (default: repo root). "
            "E.g. 'runs/sft-v1' will upload output_dir contents to that path."
        ),
    )
    p.add_argument(
        "--push_to_hub",
        action="store_true",
        help="SFT: push checkpoints to HF Hub at every save_steps (uses HF_TOKEN + HF_REPO_ID env vars, or --hub_model_id)",
    )
    p.add_argument(
        "--hub_model_id",
        default=None,
        help="SFT: HF repo id to push to (e.g. 'username/my-model'). Falls back to HF_REPO_ID env var.",
    )
    p.add_argument(
        "--hub_strategy",
        default="every_save",
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
        help="SFT: when to push to Hub (default: every_save = at every checkpoint)",
    )

    return p.parse_args()


def main():
    args = parse_args()
    if args.qlora:
        args.use_peft = True
    log_completions = not args.no_log_completions
    shuffle_dataset = not args.no_shuffle_dataset
    bf16 = not args.no_bf16 and not args.fp16
    use_double_quant = not args.no_double_quant

    model_name = args.model_name
    if args.hf_checkpoint_dir:
        checkpoint_files = (
            os.listdir(args.hf_local_dir) if os.path.isdir(args.hf_local_dir) else []
        )
        model_name = download_checkpoint_from_hf(
            checkpoint_dir=args.hf_checkpoint_dir,
            local_dir=args.hf_local_dir,
            pathes=checkpoint_files
            or [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
            ],
        )

    system_prompt = (
        _read_text_file(args.system_prompt_file)
        if args.system_prompt_file
        else SYSTEM_PROMPT
    )
    teacher_template = (
        _read_text_file(args.teacher_template_file)
        if args.teacher_template_file
        else TEACHER_TEMPLATE
    )

    dataset = load_hf_sdft_data_from_csv(
        args.dataset_path,
        system_prompt=system_prompt,
        teacher_template=teacher_template,
    )

    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_hf_sdft_data_from_csv(
            args.eval_dataset_path,
            system_prompt=system_prompt,
            teacher_template=teacher_template,
            train=False,
        )

    need_lang_model = args.dynamic_lang_feedback or (
        args.trainer_type == "grpo"
        and args.reward_func
        and "lang" in args.reward_func
    )

    lang_model = None
    if need_lang_model:
        if not args.lang_model_ckpt:
            raise ValueError(
                "--lang_model_ckpt is required when using --dynamic_lang_feedback "
                "or GRPO with the 'lang' reward"
            )
        lang_model = InferenceModel.from_checkpoint(
            ckpt_path=args.lang_model_ckpt,
            tokenizer_path=args.lang_tokenizer,
            device=torch.device("cpu"),
        )
        lang_model.compile_model(constant_bz=args.lang_compile_bz, mode="max-autotune")

    chat_template_kwargs = dict(args.chat_template_kwargs or {})
    if not args.enable_thinking or args.disable_thinking:
        chat_template_kwargs["enable_thinking"] = False

    reward_funcs = None
    if args.trainer_type == "grpo":
        if not args.reward_func:
            raise ValueError("GRPO training requires at least one --reward-func (e.g. rouge or lang)")
        reward_funcs = resolve_reward_funcs(args.reward_func, lang_model)

    run_training(
        model_name=model_name,
        train_dataset=dataset,
        output_dir=args.output_dir,
        trainer_type=args.trainer_type,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        use_peft=args.use_peft,
        qlora=args.qlora,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        use_double_quant=use_double_quant,
        lang_model=lang_model,
        dynamic_lang_feedback=args.dynamic_lang_feedback,
        seed=args.seed,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        bf16=bf16,
        fp16=args.fp16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        generation_kwargs=args.generation_kwargs,
        chat_template_kwargs=chat_template_kwargs or None,
        alpha=args.alpha,
        beta=args.beta,
        generate_from_teacher=args.generate_from_teacher,
        sync_ref_model=args.sync_ref_model,
        ref_model_sync_steps=args.ref_model_sync_steps,
        ref_model_mixup_alpha=args.ref_mixup_alpha,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        save_every_n_epochs=args.save_every_n_epochs,
        report_to=args.report_to,
        log_completions=log_completions,
        num_loss_tokens_to_skip=args.num_loss_tokens_to_skip,
        shuffle_dataset=shuffle_dataset,
        max_seq_length=args.max_seq_length,
        rouge_eval_steps=args.rouge_eval_steps,
        rouge_eval_num_samples=args.rouge_eval_num_samples,
        rouge_eval_max_new_tokens=args.rouge_eval_max_new_tokens,
        rouge_eval_batch_size=args.rouge_eval_batch_size,
        log_multilingual_rouge=not args.no_log_multilingual_rouge,
        dro_eta=args.dro_eta,
        dro_loss_beta=args.dro_loss_beta,
        dro_rouge_beta=args.dro_rouge_beta,
        dro_rouge_weight=args.dro_rouge_weight,
        dro_min_weight=args.dro_min_weight,
        dro_max_weight=args.dro_max_weight,
        dro_init_score=args.dro_init_score,
        dro_languages=args.dro_languages or [],
        eval_steps=args.eval_steps or args.save_steps,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or os.getenv("HF_REPO_ID") or None,
        hub_strategy=args.hub_strategy,
    )

    if args.upload_to_hf:
        upload_folder_to_hf(
            local_dir=args.output_dir,
            path_in_repo=args.hf_repo_path,
            commit_message=f"Upload checkpoint: {os.path.basename(args.output_dir)}",
        )


if __name__ == "__main__":
    main()
