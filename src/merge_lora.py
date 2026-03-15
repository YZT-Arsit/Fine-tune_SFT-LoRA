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
        help="LoRA adapter directory or training output directory, e.g. checkpoints/lora_qwen2.5_7b/final_adapter or checkpoints/lora_qwen2.5_7b",
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

    adapter_path = resolve_adapter_dir(Path(adapter_dir))

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


def resolve_adapter_dir(path: Path) -> Path:
    candidates = _collect_adapter_candidates(path)
    if candidates:
        return candidates[0]

    message = [
        f"Adapter directory not found or does not contain adapter files: {path}",
        "Expected one of these layouts:",
        f"  - {path}/adapter_config.json",
        f"  - {path}/final_adapter/adapter_config.json",
        f"  - {path}/checkpoint-*/adapter_config.json",
    ]
    if path.parent.exists():
        sibling_candidates = _collect_adapter_candidates(path.parent)
        if sibling_candidates:
            message.append("Found nearby adapter candidates:")
            for candidate in sibling_candidates[:5]:
                message.append(f"  - {candidate}")
    message.append("If training has not finished, you can also merge from the latest checkpoint directory.")
    raise FileNotFoundError("\n".join(message))


def _collect_adapter_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if not path.exists():
        return candidates

    if _is_adapter_dir(path):
        candidates.append(path)

    final_adapter = path / "final_adapter"
    if _is_adapter_dir(final_adapter):
        candidates.append(final_adapter)

    checkpoint_dirs = sorted(
        [child for child in path.glob("checkpoint-*") if child.is_dir()],
        key=_checkpoint_sort_key,
        reverse=True,
    )
    for checkpoint_dir in checkpoint_dirs:
        if _is_adapter_dir(checkpoint_dir):
            candidates.append(checkpoint_dir)
        adapter_subdir = checkpoint_dir / "final_adapter"
        if _is_adapter_dir(adapter_subdir):
            candidates.append(adapter_subdir)

    return _dedup_paths(candidates)


def _is_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.split("checkpoint-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _dedup_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


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
