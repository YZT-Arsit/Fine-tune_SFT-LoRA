from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .eval_metrics import evaluate_prediction_rows, parse_json_object, validate_prediction_schema
from .merge_lora import resolve_adapter_dir, resolve_dtype, resolve_model_path
from .storage import load_jsonl, write_jsonl
from .train_lora import build_tokenizer, format_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA or merged model on validation JSONL")
    parser.add_argument("--val_file", default="outputs/sft/val.jsonl")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Merged model directory. If provided, evaluation runs directly on this model.",
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path when evaluating adapter weights.",
    )
    parser.add_argument(
        "--adapter_dir",
        default=None,
        help="Adapter directory or training output directory. Required when --model_path is not set.",
    )
    parser.add_argument("--pred_out", default="outputs/lora_eval/val_predictions.jsonl")
    parser.add_argument("--report_out", default="outputs/lora_eval/val_generation_report.json")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="0 means evaluate all validation samples")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def run_lora_eval(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers/peft are required. Install requirements-lora.txt first.") from exc

    val_rows = load_jsonl(Path(args.val_file))
    if not val_rows:
        raise ValueError(f"Validation file is empty or missing: {args.val_file}")
    if args.max_eval_samples and args.max_eval_samples > 0:
        val_rows = val_rows[: args.max_eval_samples]

    tokenizer_source = args.model_path or args.base_model
    tokenizer = build_tokenizer(tokenizer_source)
    model, model_meta = _load_generation_model(args)
    model.eval()

    pred_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    sample_failures: list[dict[str, Any]] = []

    for idx, row in enumerate(val_rows):
        prompt = format_example(row, include_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        target_device = _infer_model_device(model)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed_pred, parse_err = parse_json_object(raw_text)
        schema_ok, schema_err = validate_prediction_schema(parsed_pred)
        error_reason = parse_err or schema_err

        ground_truth, gt_error = parse_json_object(str(row.get("output", "")))
        if ground_truth is None or gt_error is not None:
            continue

        sample_id = _sample_id(idx, ground_truth)
        pred_rows.append(
            {
                "idx": idx,
                "sample_id": sample_id,
                "model_name": model_meta["model_name"],
                "input": str(row.get("input", ""))[:500],
                "raw_response": raw_text[:2000],
                "parsed_json": parsed_pred,
                "parse_ok": parse_err is None,
                "schema_ok": schema_ok,
                "error_reason": error_reason,
            }
        )
        eval_rows.append(
            {
                "idx": idx,
                "sample_id": sample_id,
                "ground_truth": ground_truth,
                "raw_response": raw_text,
                "parsed_json": parsed_pred,
                "parse_ok": parse_err is None,
                "schema_ok": schema_ok,
                "error_reason": error_reason,
            }
        )

        if error_reason and len(sample_failures) < 10:
            sample_failures.append(
                {
                    "idx": idx,
                    "sample_id": sample_id,
                    "reason": error_reason,
                    "raw_response_preview": raw_text[:400],
                }
            )

    pred_path = Path(args.pred_out)
    report_path = Path(args.report_out)
    write_jsonl(pred_path, pred_rows)

    metrics = evaluate_prediction_rows(eval_rows)
    report = {
        "mode": "lora_eval",
        "model_name": model_meta["model_name"],
        "model_source": model_meta["model_source"],
        "resolved_model_path": model_meta["resolved_model_path"],
        "adapter_dir": model_meta.get("adapter_dir"),
        "val_path": str(args.val_file),
        "pred_out": str(pred_path),
        "total_requested": len(val_rows),
        "total_evaluated": len(eval_rows),
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "device_map": args.device_map,
        },
        "sample_failures": sample_failures,
    }
    report.update(metrics)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_generation_model(args: argparse.Namespace):
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers/peft are required. Install requirements-lora.txt first.") from exc

    dtype = resolve_dtype(args.dtype)
    resolved_device_map: str | dict[str, Any] = args.device_map
    if args.device_map == "cpu":
        resolved_device_map = {"": "cpu"}

    if args.model_path:
        resolved_model_path, local_files_only = resolve_model_path(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=resolved_device_map,
            local_files_only=local_files_only,
        )
        return model, {
            "model_name": Path(resolved_model_path).name,
            "model_source": "merged_model",
            "resolved_model_path": resolved_model_path,
        }

    if not args.adapter_dir:
        raise ValueError("--adapter_dir is required when --model_path is not provided")

    resolved_base_model, local_files_only = resolve_model_path(args.base_model)
    adapter_path = resolve_adapter_dir(Path(args.adapter_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=resolved_device_map,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    return model, {
        "model_name": adapter_path.name,
        "model_source": "adapter",
        "resolved_model_path": resolved_base_model,
        "adapter_dir": str(adapter_path),
    }


def _sample_id(idx: int, ground_truth: dict[str, Any]) -> str:
    key_attributes = ground_truth.get("key_attributes") or {}
    upc = key_attributes.get("upc")
    if isinstance(upc, str) and upc:
        return upc
    return str(idx)


def _infer_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device_map != "cpu":
        raise RuntimeError("CUDA is recommended for evaluation. Use --device_map cpu only for very small smoke runs.")
    report = run_lora_eval(args)
    print(
        json.dumps(
            {
                "pred_out": report["pred_out"],
                "report_out": args.report_out,
                "model_name": report["model_name"],
                "json_parse_rate": report["json_parse_rate"],
                "schema_valid_rate": report["schema_valid_rate"],
                "total_evaluated": report["total_evaluated"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[eval_lora_model] ERROR: {exc}", file=sys.stderr)
        raise
