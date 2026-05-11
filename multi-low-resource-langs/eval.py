import argparse
import logging
import os
import warnings

import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer

from langs import InferenceModel
from utils.peft import load_lora_model, load_qlora_model
from utils.data import (
  batch_create_question,
  load_json,
  strip_qwen_thinking_tokens,
  download_gdown_file,
)
from utils.metrics import calculate_rouge_score


# Route warnings and transformer/accelerate logs through tqdm.write so they
# don't break the progress bar.
class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.tqdm.write(self.format(record))


logging.basicConfig(handlers=[_TqdmLoggingHandler()], level=logging.WARNING, force=True)
warnings.showwarning = lambda msg, *a, **kw: tqdm.tqdm.write(f"Warning: {msg}")

# Suppress verbose info-level messages from transformers (e.g. pad_token_id notices)
logging.getLogger("transformers").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

_THINK_TOKEN_ID = 248068   # Qwen3 <think> token — suppress to block re-generation


@torch.inference_mode()
def generate_answer(
    questions: list[str],
    model,
    tokenizer,
    langs,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.00,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 2.0,
) -> list[str]:
    torch.cuda.empty_cache()
    inputs = batch_create_question(questions, tokenizer, language=langs)
    prompt_len = int(inputs["input_ids"].size(1))
    dev = next(iter(model.parameters())).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        suppress_tokens=[_THINK_TOKEN_ID],
    )
    # batch_decode returns one string per sequence in the batch
    return tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation report
# ──────────────────────────────────────────────────────────────────────────────

def _col_width(values: list, header: str, min_w: int = 8) -> int:
    return max(min_w, len(header), max((len(str(v)) for v in values), default=0))


def print_eval_report(
    data: list[dict],
    df_expected_langs: pd.Series | None,
    dataset_path: str,
) -> None:
    """Print a formatted evaluation report to stdout."""
    results_df = pd.DataFrame(data)
    n = len(results_df)

    sep  = "=" * 65
    dash = "─" * 65
    print(f"\n{sep}")
    print("                    EVALUATION REPORT")
    print(sep)
    print(f"  Dataset : {dataset_path}")
    print(f"  Samples : {n}")

    # ── Overall ROUGE ─────────────────────────────────────────────────────────
    has_rouge = "rouge1_f1" in results_df.columns
    if has_rouge:
        print(f"\n{dash}")
        print("  Overall ROUGE")
        print(dash)
        for col in ("rouge1_f1", "rougeL_f1", "score"):
            mean_val = results_df[col].mean()
            print(f"  {col:<14}: {mean_val:.4f}")

    # ── Per-language ROUGE ────────────────────────────────────────────────────
    if has_rouge and "expected_lang" in results_df.columns:
        print(f"\n{dash}")
        print("  Per-Language ROUGE (mean)")
        print(dash)
        grouped = results_df.groupby("expected_lang")[
            ["rouge1_f1", "rougeL_f1", "score"]
        ].agg(["mean", "count"])
        # flatten MultiIndex columns → rouge1_f1_mean, rouge1_f1_count …
        grouped.columns = ["_".join(c) for c in grouped.columns]
        n_col = grouped["rouge1_f1_count"].astype(int)

        lang_col_w = max(6, max(len(str(l)) for l in grouped.index))
        header = (
            f"  {'Lang':<{lang_col_w}}  {'rouge1_f1':>10}  "
            f"{'rougeL_f1':>10}  {'score':>8}  {'n':>6}"
        )
        print(header)
        print("  " + "─" * (len(header) - 2))
        for lang, row in grouped.iterrows():
            print(
                f"  {str(lang):<{lang_col_w}}  "
                f"{row['rouge1_f1_mean']:>10.4f}  "
                f"{row['rougeL_f1_mean']:>10.4f}  "
                f"{row['score_mean']:>8.4f}  "
                f"{int(row['rouge1_f1_count']):>6}"
            )

    # ── Language fidelity ─────────────────────────────────────────────────────
    if "detected_lang" in results_df.columns and df_expected_langs is not None:
        print(f"\n{dash}")
        print("  Language Fidelity  (detected vs expected)")
        print(dash)

        results_df["lang_correct"] = (
            results_df["detected_lang"].str.lower()
            == results_df["expected_lang"].str.lower()
        )
        overall_acc = results_df["lang_correct"].mean()
        n_correct   = results_df["lang_correct"].sum()
        print(f"  Overall accuracy : {overall_acc:.1%}  ({n_correct} / {n})")

        print(f"\n  {'Lang':<12}  {'accuracy':>10}  {'correct':>8}  {'total':>6}")
        print("  " + "─" * 42)
        for lang, grp in results_df.groupby("expected_lang"):
            acc     = grp["lang_correct"].mean()
            correct = grp["lang_correct"].sum()
            total   = len(grp)
            print(
                f"  {str(lang):<12}  {acc:>10.1%}  "
                f"{correct:>8}  {total:>6}"
            )

        # Confusion: expected → detected distribution
        print(f"\n  Detected-language distribution (all samples):")
        dist = results_df["detected_lang"].value_counts()
        for lang, cnt in dist.items():
            print(f"    {lang:<12} : {cnt:>5}  ({cnt/n:.1%})")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Load generation model ─────────────────────────────────────────────────
    model_id = args.model_id
    if args.gdrive_dataset_id:
        dataset_path = download_gdown_file(args.gdrive_dataset_id, "data/Test.csv")
    else:
        dataset_path = args.data_csv_path
    if args.peft_type == "qlora":
        model, _ = load_qlora_model(model_id, from_peft_model=True)
    elif args.peft_type == "peft":
        model, _ = load_lora_model(model_id, from_peft_model=True)
    else:
        raise ValueError(
            f"Unknown peft_type {args.peft_type!r}. Use --use_peft or --qlora."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    # ── Load language classifier (optional) ───────────────────────────────────
    lang_model: InferenceModel | None = None
    if args.lang_model_ckpt and os.path.isfile(args.lang_model_ckpt):
        lang_tokenizer_path = args.lang_tokenizer_path or os.path.join(
            os.path.dirname(args.lang_model_ckpt), "tokenizer.json"
        )
        print(f"Loading language classifier from {args.lang_model_ckpt} …")
        lang_model = InferenceModel.from_checkpoint(
            args.lang_model_ckpt, tokenizer_path=lang_tokenizer_path
        )

    gen_kwargs  = load_json(args.gen_kwargs_path or "configs/gen_kwargs.json")
    mode        = args.mode
    batch_size  = args.batch_size
    df          = pd.read_csv(dataset_path)
    df['lang'] = df['subset'].apply(lambda x: x.split("_")[0])
    to_save_path = args.to_save_path or os.path.join(os.path.dirname(args.data_csv_path), f"{args.mode}_results.csv")
    if to_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(to_save_path)), exist_ok=True)

    labels         = [ "TargetRLF1", "TargetR1F1", "TargetLLM"]
    data: list[dict] = []

    # Placeholder used when generation fails or produces an empty string.
    # A non-empty string prevents the CSV round-trip ""→NaN bug (pandas read_csv
    # converts empty cells to NaN by default).
    _FALLBACK_ANSWER = "I am unable to provide an answer to this question at this time."

    for i in tqdm.tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch       = df.iloc[i : i + batch_size]
        qs          = batch["input"].values.tolist()
        batch_langs = batch["lang"].values.tolist()

        # ── Safe generation: catch OOM / any runtime error per batch ──────────
        try:
            answers = generate_answer(
                qs,
                model,
                tokenizer,
                langs=batch_langs,
                max_new_tokens=args.max_new_tokens,
                **gen_kwargs,
            )
            answers = [strip_qwen_thinking_tokens(a) for a in answers]
        except Exception as exc:
            tqdm.tqdm.write(f"[WARNING] batch {i}–{i+batch_size} failed ({exc}); using fallback answers.")
            answers = [_FALLBACK_ANSWER] * len(qs)

        # Guard: replace any empty string (model produced only EOS/pad tokens)
        # with the fallback so it never becomes NaN in the CSV round-trip.
        answers = [a if a.strip() else _FALLBACK_ANSWER for a in answers]

        # Detect language of each generated answer
        detected_langs: list[str] | None = None
        if lang_model is not None:
            detected_langs, _ = lang_model.predict(answers)

        if mode == "test":
            for j, (row_id, answer) in enumerate(zip(batch["ID"], answers)):
                item: dict = {"ID": row_id, "expected_lang": batch_langs[j]}
                item.update({lbl: answer for lbl in labels})
                if detected_langs is not None:
                    item["detected_lang"] = detected_langs[j]
                data.append(item)

        elif mode == "eval":
            for j, (row_id, gen_answer, gold_answer) in enumerate(
                zip(batch["ID"], answers, batch["output"])
            ):
                item = {
                    "ID": row_id,
                    "gen_answer": gen_answer,
                    "expected_lang": batch_langs[j],
                }
                item.update(calculate_rouge_score(gold_answer, gen_answer))
                if detected_langs is not None:
                    item["detected_lang"] = detected_langs[j]
                data.append(item)

        else:
            raise ValueError(f"Invalid mode: {repr(mode)}")

        if to_save_path:
            save_df = pd.DataFrame(data)
            if mode == "test":
                save_df = save_df[["ID"] + labels]
            save_df.to_csv(to_save_path, index=False)

    # ── Print evaluation report ───────────────────────────────────────────────
    print_eval_report(
        data,
        df_expected_langs=df["lang"] if "lang" in df.columns else None,
        dataset_path=args.data_csv_path,
    )

    if to_save_path:
        print(f"Results saved to: {to_save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate or run inference with a fine-tuned multilingual model."
    )
    p.add_argument(
        "--gdrive-dataset-id",
        default=None,
        help="Drive dataset ID to evaluate.",
    )
    p.add_argument(
        "--model_id",
        required=True,
        help="Path or HuggingFace hub ID of the (Q)LoRA checkpoint to evaluate.",
    )
    p.add_argument(
        "--data_csv_path",
        required=True,
        help="CSV file with at least 'ID', 'input', 'lang' columns "
             "(and 'output' for eval mode).",
    )
    p.add_argument(
        "--mode",
        choices=("test", "eval"),
        default="eval",
        help="'eval': compute ROUGE against gold answers; 'test': generate only.",
    )
    p.add_argument(
        "--to_save_path",
        default=None,
        help="Path to write the output CSV (optional but recommended).",
    )
    peft_group = p.add_mutually_exclusive_group()
    peft_group.add_argument(
        "--use_peft",
        dest="peft_type",
        action="store_const",
        const="peft",
        help="Load model with plain LoRA (default).",
    )
    peft_group.add_argument(
        "--qlora",
        dest="peft_type",
        action="store_const",
        const="qlora",
        help="Load model with 4-bit NF4 QLoRA.",
    )
    p.set_defaults(peft_type="peft")
    p.add_argument(
        "--gen_kwargs_path",
        default="configs/gen_kwargs.json",
        help="JSON file with extra generation kwargs (temperature, top_p, …).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per answer.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples to process in one forward pass.",
    )
    # Language classifier (for fidelity stats)
    p.add_argument(
        "--lang_model_ckpt",
        default="checkpoints/lang_classifier.pt",
        help="Path to the language-classifier .pt checkpoint. "
             "Set to '' to skip language fidelity stats.",
    )
    p.add_argument(
        "--lang_tokenizer_path",
        default=None,
        help="Path to the language-classifier tokenizer.json. "
             "Defaults to <lang_model_ckpt directory>/tokenizer.json.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
