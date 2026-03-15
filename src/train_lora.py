from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .eval_metrics import evaluate_prediction_rows, parse_json_object, validate_prediction_schema
from .storage import load_jsonl, write_jsonl


SYSTEM_PROMPT = (
    "你是一个信息抽取助手。请根据用户提供的商品页面文本，抽取并规范化商品信息。"
    "你必须只输出一个 JSON 对象，不要输出解释，不要输出 markdown 代码块，不要补充额外文本。"
)


@dataclass
class TrainRuntimeConfig:
    train_file: str
    val_file: str
    model_name_or_path: str
    output_dir: str
    mode: str
    max_seq_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    weight_decay: float
    lr_scheduler_type: str
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    max_eval_samples: int
    do_generation_eval: bool
    generation_max_new_tokens: int
    seed: int
    max_steps: int
    assistant_only_loss: bool
    resolved_model_path: str
    local_files_only: bool


def load_jsonl_dataset(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    rows = load_jsonl(file_path)
    if not rows:
        raise ValueError(f"Dataset is empty: {file_path}")
    return rows


def resolve_model_path(model_name_or_path: str) -> tuple[str, bool]:
    raw_path = Path(model_name_or_path).expanduser()
    if raw_path.exists():
        resolved = _find_local_model_dir(raw_path)
        return str(resolved), True
    return model_name_or_path, _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE")


def format_example(example: dict[str, Any], include_answer: bool = True) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()

    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n\n"
        f"输入文本：\n{input_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if include_answer:
        return prompt + f"{output_text}<|im_end|>\n"
    return prompt


def build_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for training. Install requirements-lora.txt.") from exc

    resolved_model_path, local_files_only = resolve_model_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model_and_peft_config(args: argparse.Namespace):
    try:
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("peft/transformers is required for LoRA training. Install requirements-lora.txt.") from exc

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    preferred_dtype = torch.bfloat16 if bf16_supported else torch.float16
    resolved_model_path, local_files_only = resolve_model_path(args.model_name_or_path)

    quantization_config = None
    if args.mode == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=preferred_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=preferred_dtype if args.mode == "lora" else None,
        device_map="auto",
        local_files_only=local_files_only,
    )

    if args.mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return model, peft_config, bf16_supported, resolved_model_path, local_files_only


def build_trainer(
    args: argparse.Namespace,
    model,
    tokenizer,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    peft_config,
):
    try:
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError("trl/datasets/transformers is required. Install requirements-lora.txt.") from exc

    train_ds = Dataset.from_list([{"text": format_example(row, include_answer=True)} for row in train_rows])
    val_ds = Dataset.from_list([{"text": format_example(row, include_answer=True)} for row in val_rows])

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = not bf16_supported
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        bf16=bf16_supported,
        fp16=fp16_enabled,
        seed=args.seed,
        report_to="none",
        load_best_model_at_end=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    data_collator = None
    assistant_only_loss_active = False
    if args.assistant_only_loss:
        # If this collator is available, only assistant span tokens contribute to loss.
        # Otherwise we fall back to full causal LM loss over the whole prompt.
        try:
            from trl import DataCollatorForCompletionOnlyLM

            data_collator = DataCollatorForCompletionOnlyLM(
                response_template="<|im_start|>assistant\n",
                tokenizer=tokenizer,
            )
            assistant_only_loss_active = True
        except Exception:
            assistant_only_loss_active = False

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        data_collator=data_collator,
    )

    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    return trainer, assistant_only_loss_active


def run_generation_eval(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("peft/transformers is required for generation eval.") from exc

    val_rows = load_jsonl_dataset(args.val_file)[: max(1, args.max_eval_samples)]
    final_adapter_dir = Path(args.output_dir) / "final_adapter"

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    preferred_dtype = torch.bfloat16 if bf16_supported else torch.float16
    resolved_model_path, local_files_only = resolve_model_path(args.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        torch_dtype=preferred_dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, str(final_adapter_dir))
    model.eval()

    pred_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    sample_failures: list[dict[str, Any]] = []

    for idx, row in enumerate(val_rows):
        prompt = format_example(row, include_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed_pred, parse_err = parse_json_object(raw_text)
        schema_ok, schema_err = validate_prediction_schema(parsed_pred)
        err = parse_err or schema_err

        gt, gt_err = parse_json_object(str(row.get("output", "")))
        if gt is None or gt_err:
            continue

        pred_rows.append(
            {
                "idx": idx,
                "raw_response": raw_text,
                "parsed_json": parsed_pred,
                "parse_ok": parse_err is None,
                "schema_ok": schema_ok,
                "error_reason": err,
            }
        )

        eval_rows.append(
            {
                "idx": idx,
                "sample_id": _sample_id(idx, gt),
                "ground_truth": gt,
                "raw_response": raw_text,
                "parsed_json": parsed_pred,
                "parse_ok": parse_err is None,
                "schema_ok": schema_ok,
                "error_reason": err,
            }
        )

        if err and len(sample_failures) < 20:
            sample_failures.append(
                {
                    "idx": idx,
                    "reason": err,
                    "raw_response_preview": raw_text[:400],
                }
            )

    pred_out = Path("outputs/lora_eval/val_predictions.jsonl")
    report_out = Path("outputs/lora_eval/val_generation_report.json")
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_out, pred_rows)

    metrics = evaluate_prediction_rows(eval_rows)
    report = {
        "total_evaluated": len(eval_rows),
        "json_parse_rate": metrics["json_parse_rate"],
        "schema_valid_rate": metrics["schema_valid_rate"],
        "field_accuracy": metrics["field_accuracy"],
        "numeric_error": metrics["numeric_error"],
        "sample_failures": sample_failures[:10],
        "predictions_path": str(pred_out),
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA/QLoRA for structured extraction")
    parser.add_argument("--train_file", default="outputs/sft/train.jsonl")
    parser.add_argument("--val_file", default="outputs/sft/val.jsonl")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="checkpoints/lora_qwen2.5_7b")
    parser.add_argument("--mode", choices=["qlora", "lora"], default="qlora")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--max_eval_samples", type=int, default=50)
    parser.add_argument("--do_generation_eval", action="store_true")
    parser.add_argument("--generation_max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--assistant_only_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try masking loss to assistant response only when TRL collator is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. No CUDA device detected.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl_dataset(args.train_file)
    val_rows = load_jsonl_dataset(args.val_file)
    tokenizer = build_tokenizer(args.model_name_or_path)
    model, peft_config, bf16_supported, resolved_model_path, local_files_only = build_model_and_peft_config(args)

    trainer, assistant_only_loss_active = build_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_rows=train_rows,
        val_rows=val_rows,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    train_metrics["assistant_only_loss_active"] = assistant_only_loss_active
    train_metrics["assistant_only_loss_requested"] = bool(args.assistant_only_loss)
    train_metrics["note"] = (
        "assistant-only loss masking enabled via DataCollatorForCompletionOnlyLM"
        if assistant_only_loss_active
        else "assistant-only masking unavailable in current TRL version; training used full causal LM loss"
    )
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    runtime_config = TrainRuntimeConfig(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        mode=args.mode,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        max_eval_samples=args.max_eval_samples,
        do_generation_eval=args.do_generation_eval,
        generation_max_new_tokens=args.generation_max_new_tokens,
        seed=args.seed,
        max_steps=args.max_steps,
        assistant_only_loss=args.assistant_only_loss,
        resolved_model_path=resolved_model_path,
        local_files_only=local_files_only,
    )
    config_dump = asdict(runtime_config)
    config_dump["bf16_supported"] = bf16_supported
    config_dump["assistant_only_loss_active"] = assistant_only_loss_active
    (output_dir / "training_config.json").write_text(
        json.dumps(config_dump, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.do_generation_eval:
        gen_report = run_generation_eval(args=args, tokenizer=tokenizer)
        print(json.dumps({"generation_eval": gen_report}, ensure_ascii=False))

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "final_adapter_dir": str(final_adapter_dir),
                "resolved_model_path": resolved_model_path,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "assistant_only_loss_active": assistant_only_loss_active,
            },
            ensure_ascii=False,
        )
    )


def _sample_id(idx: int, gt: dict[str, Any]) -> str:
    key_attr = gt.get("key_attributes") or {}
    upc = key_attr.get("upc")
    if isinstance(upc, str) and upc:
        return upc
    return str(idx)


def _find_local_model_dir(path: Path) -> Path:
    direct_markers = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    if any((path / marker).exists() for marker in direct_markers):
        return path

    snapshot_dirs = [p for p in path.rglob("*") if p.is_dir() and (p / "config.json").exists()]
    if not snapshot_dirs:
        raise FileNotFoundError(
            "Local model path exists but no Hugging Face-compatible model files were found under "
            f"{path}. Expected config.json/tokenizer files."
        )

    preferred = sorted(snapshot_dirs, key=lambda p: (0 if "snapshots" in p.parts else 1, len(p.parts)))
    return preferred[0]


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[train_lora] ERROR: {exc}", file=sys.stderr)
        raise
