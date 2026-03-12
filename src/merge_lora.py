from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="LoRA adapter directory, e.g. checkpoints/lora_qwen2.5_7b/final_adapter",
    )
    parser.add_argument("--output_dir", required=True, help="Merged model output directory")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Load/merge dtype",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Device map for loading model, e.g. "auto" or "cpu"',
    )
    parser.add_argument(
        "--safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save using safetensors when possible",
    )
    parser.add_argument(
        "--max_shard_size",
        default="5GB",
        help='Max shard size when saving merged model, e.g. "5GB"',
    )
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def merge_lora_adapter(
    *,
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    dtype: str,
    device_map: str,
    safe_serialization: bool,
    max_shard_size: str,
) -> dict[str, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers/peft are required. Install requirements-lora.txt first.") from exc

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_dtype(dtype)
    resolved_device_map: str | dict[str, Any] = device_map
    if device_map == "cpu":
        resolved_device_map = {"": "cpu"}

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=resolved_device_map,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(
        str(out_path),
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(str(out_path))

    merge_meta = {
        "base_model": base_model,
        "adapter_dir": str(adapter_path),
        "output_dir": str(out_path),
        "dtype": str(torch_dtype),
        "device_map": device_map,
        "safe_serialization": safe_serialization,
        "max_shard_size": max_shard_size,
    }
    (out_path / "merge_config.json").write_text(
        json.dumps(merge_meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return merge_meta


def main() -> None:
    args = parse_args()
    result = merge_lora_adapter(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[merge_lora] ERROR: {exc}", file=sys.stderr)
        raise
