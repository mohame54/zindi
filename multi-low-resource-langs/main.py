from __future__ import annotations

import argparse
import json
import os

import torch

from train import run_training
from utils.data import SYSTEM_PROMPT, TEACHER_TEMPLATE, load_hf_sdft_data_from_csv
from utils.hf import download_checkpoint_from_hf, upload_file_paths_to_hf
from langs import InferenceModel


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _json_dict(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def parse_args():
    p = argparse.ArgumentParser(description="Multilingual self-distillation (SDFT) training")
    p.add_argument("--model_name", default="Qwen/Qwen3.5-2B")
    p.add_argument("--dataset_path", required=True, help="CSV path for load_hf_sdft_data_from_csv")
    p.add_argument("--output_dir", required=True)

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
        "--save_every_n_epochs",
        type=int,
        default=0,
        help="If > 0, save under output_dir/checkpoint-epoch-{k} every k epochs",
    )
    p.add_argument("--report_to", default="wandb", help="e.g. wandb, none, tensorboard, or comma-separated")
    p.add_argument("--no_log_completions", action="store_true")
    p.add_argument("--num_loss_tokens_to_skip", type=int, default=3)
    p.add_argument("--no_shuffle_dataset", action="store_true")

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
        help="UTF-8 file whose contents replace TEACHER_TEMPLATE (must include {question}, {golden_answer}, {lang_hint})",
    )

    # HF Hub
    p.add_argument("--upload_to_hf", action="store_true")
    p.add_argument("--hf_checkpoint_dir", default=None)
    p.add_argument("--hf_local_dir", default="hf_checkpoint")

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

    lang_model = None
    if args.dynamic_lang_feedback:
        if not args.lang_model_ckpt:
            raise ValueError("--dynamic_lang_feedback requires --lang_model_ckpt")
        lang_model = InferenceModel.from_checkpoint(
            ckpt_path=args.lang_model_ckpt,
            tokenizer_path=args.lang_tokenizer,
            device=torch.device("cpu"),
        )
        lang_model.compile_model(constant_bz=args.lang_compile_bz, mode="max-autotune")

    chat_template_kwargs = dict(args.chat_template_kwargs or {})
    if not args.enable_thinking or args.disable_thinking:
        chat_template_kwargs["enable_thinking"] = False

    run_training(
        model_name=model_name,
        train_dataset=dataset,
        output_dir=args.output_dir,
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
        save_every_n_epochs=args.save_every_n_epochs,
        report_to=args.report_to,
        log_completions=log_completions,
        num_loss_tokens_to_skip=args.num_loss_tokens_to_skip,
        shuffle_dataset=shuffle_dataset,
    )

    if args.upload_to_hf:
        ckpt_files = [
            os.path.join(args.output_dir, f)
            for f in os.listdir(args.output_dir)
            if os.path.isfile(os.path.join(args.output_dir, f))
        ]
        upload_file_paths_to_hf(ckpt_files)


if __name__ == "__main__":
    main()
