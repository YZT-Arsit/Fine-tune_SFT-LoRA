from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_TAGS = {
    "long_input",
    "missing_category",
    "has_non_ascii",
    "normalization_error",
    "null_handling_error",
    "truncation_suspected",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment SFT data with badcase-driven perturbations")
    parser.add_argument("--train_in", required=True, help="Original training jsonl path")
    parser.add_argument("--badcase_in", required=True, help="Badcase jsonl path")
    parser.add_argument("--train_out", required=True, help="Augmented training jsonl path")
    parser.add_argument("--max_aug_per_sample", type=int, default=2, help="Max augmentations per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dedup",
        dest="dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate by instruction+input+output hash before writing",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = _load_jsonl(Path(args.train_in))
    badcase_rows = _load_jsonl(Path(args.badcase_in))

    active_strategies, badcase_stats = _derive_active_strategies(badcase_rows)
    strategy_counts = Counter()
    augmented_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(train_rows):
        augmented_rows.append(row)
        generated = 0
        used: set[str] = set()
        for strategy in _strategy_order(active_strategies, rng):
            if generated >= args.max_aug_per_sample:
                break
            if strategy in used:
                continue
            out_row = _apply_strategy(row, strategy, rng)
            if out_row is None:
                continue
            augmented_rows.append(out_row)
            generated += 1
            used.add(strategy)
            strategy_counts[strategy] += 1

        if generated == 0:
            fallback = _apply_strategy(row, "shuffle_fields", rng)
            if fallback is not None:
                augmented_rows.append(fallback)
                strategy_counts["shuffle_fields"] += 1

        _ = idx

    dedup_removed_count = 0
    if args.dedup:
        augmented_rows, dedup_removed_count = _deduplicate_rows(augmented_rows)

    train_out = Path(args.train_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_out, augmented_rows)

    report = {
        "original_samples": len(train_rows),
        "augmented_samples": len(augmented_rows) - len(train_rows),
        "final_samples": len(augmented_rows),
        "dedup_enabled": bool(args.dedup),
        "dedup_removed_count": dedup_removed_count,
        "strategy_usage_counts": dict(strategy_counts),
        "active_strategies": sorted(active_strategies),
        "badcase_count": len(badcase_rows),
        "badcase_coverage_rate": _safe_rate(badcase_stats["targeted_count"], len(badcase_rows)),
        "badcase_target_tag_counts": badcase_stats["tag_counts"],
    }

    report_path = train_out.parent / "train_augmented_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "train_out": str(train_out),
                "report_out": str(report_path),
                "original_samples": report["original_samples"],
                "augmented_samples": report["augmented_samples"],
                "final_samples": report["final_samples"],
            },
            ensure_ascii=False,
        )
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        key = _row_hash_key(row)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def _row_hash_key(row: dict[str, Any]) -> str:
    return "||".join(
        [
            str(row.get("instruction", "")),
            str(row.get("input", "")),
            str(row.get("output", "")),
        ]
    )


def _derive_active_strategies(badcases: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    active: set[str] = set()
    tag_counts = Counter()
    targeted_count = 0

    for row in badcases:
        tags = set(row.get("bucket_tags") or [])
        error_type = str(row.get("error_type_guess") or "")
        targeted = False

        for tag in tags:
            if tag in TARGET_TAGS:
                tag_counts[tag] += 1
                targeted = True
                if tag == "long_input":
                    active.add("trim_description")
                    active.add("description_repeat_variant")
                elif tag == "missing_category":
                    active.add("drop_category_line")
                elif tag == "has_non_ascii":
                    active.add("special_char_perturb")
                elif tag == "truncation_suspected":
                    active.add("trim_product_information")

        if error_type == "normalization_error":
            active.add("pi_numeric_surface_perturb")
            targeted = True
        if error_type == "null_handling_error":
            active.add("drop_stock_count_line")
            active.add("drop_upc_line")
            targeted = True

        if targeted:
            targeted_count += 1

    active.update({"shuffle_fields", "shuffle_product_information_order", "drop_description_line"})
    return active, {"tag_counts": dict(tag_counts), "targeted_count": targeted_count}


def _strategy_order(active: set[str], rng: random.Random) -> list[str]:
    ordered = list(active)
    rng.shuffle(ordered)
    return ordered


def _apply_strategy(row: dict[str, Any], strategy: str, rng: random.Random) -> dict[str, Any] | None:
    parsed = _parse_sample(row)
    if parsed is None:
        return None
    blocks, output_obj = parsed

    if strategy == "shuffle_fields":
        _shuffle_main_blocks(blocks, rng)
    elif strategy == "shuffle_product_information_order":
        rng.shuffle(blocks["pi_items"])
    elif strategy == "drop_category_line":
        blocks["category"] = None
        output_obj["category"] = None
    elif strategy == "drop_upc_line":
        _drop_pi_key(blocks, "UPC")
        key_attr = output_obj.get("key_attributes", {})
        if isinstance(key_attr, dict):
            key_attr["upc"] = None
    elif strategy == "drop_stock_count_line":
        blocks["availability"] = _strip_stock_count(blocks["availability"] or "")
        availability = output_obj.get("availability", {})
        if isinstance(availability, dict):
            availability["stock_count"] = None
    elif strategy == "trim_description":
        if not blocks["description"]:
            return None
        blocks["description"] = _trim_description(blocks["description"])
    elif strategy == "drop_description_line":
        if blocks["description"] is None:
            return None
        blocks["description"] = None
    elif strategy == "trim_product_information":
        if len(blocks["pi_items"]) <= 3:
            return None
        kept = blocks["pi_items"][: max(3, len(blocks["pi_items"]) // 2)]
        dropped_keys = {k for k, _ in blocks["pi_items"] if (k, _) not in kept}
        blocks["pi_items"] = kept
        _null_output_for_pi_drops(output_obj, dropped_keys)
    elif strategy == "special_char_perturb":
        _special_char_perturb(blocks)
    elif strategy == "pi_numeric_surface_perturb":
        if not _pi_numeric_surface_perturb(blocks):
            return None
    elif strategy == "description_repeat_variant":
        if not blocks["description"]:
            return None
        blocks["description"] = _repeat_description(blocks["description"])
    else:
        return None

    new_input = _render_input(blocks)
    new_output = _normalize_output_schema(output_obj)
    if new_input == row.get("input", "") and json.dumps(new_output, ensure_ascii=False, sort_keys=True) == str(row.get("output", "")):
        return None

    return {
        "instruction": row.get("instruction", ""),
        "input": new_input,
        "output": json.dumps(new_output, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def _parse_sample(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        output_obj = json.loads(str(row.get("output", "")))
    except json.JSONDecodeError:
        return None
    if not isinstance(output_obj, dict):
        return None

    input_text = str(row.get("input", ""))
    lines = input_text.splitlines()
    blocks: dict[str, Any] = {
        "title": None,
        "category": None,
        "price": None,
        "availability": None,
        "description": None,
        "pi_items": [],
        "main_order": ["title", "category", "price", "availability", "description"],
    }
    in_pi = False
    for line in lines:
        if line.startswith("Product Information:"):
            in_pi = True
            continue
        if in_pi:
            if ":" in line:
                k, v = line.split(":", 1)
                blocks["pi_items"].append((k.strip(), v.strip()))
            continue
        if line.startswith("Title:"):
            blocks["title"] = line[len("Title:") :].strip()
        elif line.startswith("Category:"):
            blocks["category"] = line[len("Category:") :].strip()
        elif line.startswith("Price:"):
            blocks["price"] = line[len("Price:") :].strip()
        elif line.startswith("Availability:"):
            blocks["availability"] = line[len("Availability:") :].strip()
        elif line.startswith("Description:"):
            blocks["description"] = line[len("Description:") :].strip()
    return blocks, copy.deepcopy(output_obj)


def _render_input(blocks: dict[str, Any]) -> str:
    lines: list[str] = []
    mapping = {
        "title": ("Title", blocks["title"]),
        "category": ("Category", blocks["category"]),
        "price": ("Price", blocks["price"]),
        "availability": ("Availability", blocks["availability"]),
        "description": ("Description", blocks["description"]),
    }
    for key in blocks["main_order"]:
        label, value = mapping[key]
        if value is None:
            continue
        lines.append(f"{label}: {value}")

    lines.append("Product Information:")
    if blocks["pi_items"]:
        for k, v in blocks["pi_items"]:
            lines.append(f"{k}: {v}")
    else:
        lines.append("(empty)")
    return "\n".join(lines).strip()


def _shuffle_main_blocks(blocks: dict[str, Any], rng: random.Random) -> None:
    ordered = list(blocks["main_order"])
    rng.shuffle(ordered)
    blocks["main_order"] = ordered


def _drop_pi_key(blocks: dict[str, Any], key_name: str) -> None:
    blocks["pi_items"] = [(k, v) for k, v in blocks["pi_items"] if k != key_name]


def _strip_stock_count(text: str) -> str:
    stripped = re.sub(r"\(\s*\d+\s+available\s*\)", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stripped).strip()


def _trim_description(text: str) -> str:
    if len(text) < 120:
        return text
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    keep = parts[: max(1, len(parts) // 2)]
    return " ".join(p.strip() for p in keep if p.strip())


def _null_output_for_pi_drops(output_obj: dict[str, Any], dropped_keys: set[str]) -> None:
    key_attr = output_obj.get("key_attributes")
    if not isinstance(key_attr, dict):
        return
    mapping = {
        "UPC": "upc",
        "Product Type": "product_type",
        "Price (excl. tax)": "price_excl_tax",
        "Price (incl. tax)": "price_incl_tax",
        "Tax": "tax",
        "Availability": "availability_text",
        "Number of reviews": "review_count",
    }
    for src, dst in mapping.items():
        if src in dropped_keys:
            key_attr[dst] = None


def _special_char_perturb(blocks: dict[str, Any]) -> None:
    replacements = {
        "'": "’",
        "-": "—",
        "£": "￡",
    }
    for field in ("title", "availability", "description"):
        value = blocks.get(field)
        if not isinstance(value, str):
            continue
        for src, dst in replacements.items():
            value = value.replace(src, dst)
        blocks[field] = value
    new_items = []
    for k, v in blocks["pi_items"]:
        vv = v
        for src, dst in replacements.items():
            vv = vv.replace(src, dst)
        new_items.append((k, vv))
    blocks["pi_items"] = new_items


def _pi_numeric_surface_perturb(blocks: dict[str, Any]) -> bool:
    changed = False
    out_items = []
    for k, v in blocks["pi_items"]:
        vv = v
        if any(token in k for token in ("Price", "Tax")) and "£" in vv:
            vv = vv.replace("£", "GBP ")
            changed = True
        out_items.append((k, vv))
    blocks["pi_items"] = out_items
    return changed


def _repeat_description(text: str) -> str:
    pieces = re.split(r"(?<=[.!?。！？])\s+", text)
    first = pieces[0].strip() if pieces else text
    if not first:
        return text
    return f"{text} {first}"


def _normalize_output_schema(output_obj: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(output_obj)
    result.setdefault("title", "")
    result.setdefault("category", None)
    result.setdefault("price", None)
    result.setdefault("currency", "GBP")
    availability = result.setdefault("availability", {})
    if not isinstance(availability, dict):
        availability = {}
        result["availability"] = availability
    availability.setdefault("in_stock", False)
    availability.setdefault("stock_count", None)

    result.setdefault("rating", None)
    key_attr = result.setdefault("key_attributes", {})
    if not isinstance(key_attr, dict):
        key_attr = {}
        result["key_attributes"] = key_attr
    for key in [
        "upc",
        "product_type",
        "price_excl_tax",
        "price_incl_tax",
        "tax",
        "availability_text",
        "review_count",
    ]:
        key_attr.setdefault(key, None)
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    main()
