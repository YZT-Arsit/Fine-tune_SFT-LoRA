from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FIELD_PATHS = [
    "title",
    "category",
    "price",
    "currency",
    "availability.in_stock",
    "availability.stock_count",
    "rating",
    "key_attributes.upc",
    "key_attributes.product_type",
    "key_attributes.price_excl_tax",
    "key_attributes.price_incl_tax",
    "key_attributes.tax",
    "key_attributes.availability_text",
    "key_attributes.review_count",
]

NUMERIC_FIELDS = {
    "price",
    "availability.stock_count",
    "rating",
    "key_attributes.price_excl_tax",
    "key_attributes.price_incl_tax",
    "key_attributes.tax",
    "key_attributes.review_count",
}

ABS_ERROR_FIELDS = {
    "price",
    "key_attributes.price_excl_tax",
    "key_attributes.price_incl_tax",
    "key_attributes.tax",
    "availability.stock_count",
    "rating",
    "key_attributes.review_count",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze badcases for structured extraction predictions")
    parser.add_argument("--gold", required=True, help="Gold val.jsonl path")
    parser.add_argument("--pred", required=True, help="Prediction jsonl path")
    parser.add_argument("--summary_out", required=True, help="Summary json output path")
    parser.add_argument("--badcases_out", required=True, help="Badcases jsonl output path")
    parser.add_argument("--long_input_threshold", type=int, default=1200, help="Threshold for long_input tag")
    args = parser.parse_args()

    gold_rows = _load_jsonl(Path(args.gold))
    pred_rows = _load_jsonl(Path(args.pred))

    total = min(len(gold_rows), len(pred_rows))
    records: list[dict[str, Any]] = []

    field_match_counts = Counter()
    field_total_counts = Counter()
    numeric_errors: dict[str, list[float]] = defaultdict(list)
    bucket_counter = Counter()
    error_type_counter = Counter()

    parse_ok_count = 0
    schema_ok_count = 0

    for idx in range(total):
        gold_row = gold_rows[idx]
        pred_row = pred_rows[idx]

        gold_output = _safe_json_loads(str(gold_row.get("output", "")))
        pred_output = pred_row.get("parsed_json")
        if not isinstance(pred_output, dict):
            pred_output = None

        parse_ok = bool(pred_row.get("parse_ok", False))
        schema_ok = bool(pred_row.get("schema_ok", False))
        if parse_ok:
            parse_ok_count += 1
        if schema_ok:
            schema_ok_count += 1

        mismatch_fields: list[str] = []
        if isinstance(gold_output, dict):
            for path in FIELD_PATHS:
                g = _get_path(gold_output, path)
                p = _get_path(pred_output, path) if pred_output else None
                field_total_counts[path] += 1
                if _value_equal(g, p, path):
                    field_match_counts[path] += 1
                else:
                    mismatch_fields.append(path)
                if path in ABS_ERROR_FIELDS:
                    err = _abs_error(g, p)
                    if err is not None:
                        numeric_errors[path].append(err)

        input_text = str(gold_row.get("input", ""))
        bucket_tags = _build_bucket_tags(
            input_text=input_text,
            gold_output=gold_output,
            parse_ok=parse_ok,
            schema_ok=schema_ok,
            threshold=args.long_input_threshold,
        )
        for tag in bucket_tags:
            bucket_counter[tag] += 1

        error_type = _guess_error_type(
            mismatch_fields=mismatch_fields,
            parse_ok=parse_ok,
            schema_ok=schema_ok,
            pred_row=pred_row,
            pred_output=pred_output,
        )
        error_type_counter[error_type] += 1

        if mismatch_fields or not parse_ok or not schema_ok:
            records.append(
                {
                    "idx": pred_row.get("idx", idx),
                    "input_preview": input_text[:500],
                    "gold_output": gold_output,
                    "pred_output": pred_output,
                    "mismatch_fields": mismatch_fields,
                    "bucket_tags": bucket_tags,
                    "error_type_guess": error_type,
                }
            )

    field_accuracy = {
        field: _safe_rate(field_match_counts[field], field_total_counts[field])
        for field in FIELD_PATHS
    }

    numeric_summary = {}
    for field in sorted(ABS_ERROR_FIELDS):
        vals = numeric_errors[field]
        numeric_summary[field] = {
            "abs_error_avg": _avg(vals),
            "abs_error_p50": _percentile(vals, 50),
            "abs_error_p95": _percentile(vals, 95),
        }

    summary = {
        "total_aligned_samples": total,
        "parse_rate": _safe_rate(parse_ok_count, total),
        "schema_rate": _safe_rate(schema_ok_count, total),
        "field_accuracy": field_accuracy,
        "numeric_error": numeric_summary,
        "badcase_count": len(records),
        "bucket_stats": dict(bucket_counter),
        "top_error_types": error_type_counter.most_common(10),
        "sample_badcases": _sample_badcases_by_type(records, per_type=3),
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    badcases_path = Path(args.badcases_out)
    badcases_path.parent.mkdir(parents=True, exist_ok=True)
    with badcases_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "total_aligned_samples": total,
                "badcase_count": len(records),
                "summary_out": str(summary_path),
                "badcases_out": str(badcases_path),
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


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _get_path(obj: dict[str, Any] | None, path: str) -> Any:
    if obj is None:
        return None
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _value_equal(gold: Any, pred: Any, field: str) -> bool:
    if field in NUMERIC_FIELDS:
        gnum = _to_number(gold)
        pnum = _to_number(pred)
        if gnum is None and pnum is None:
            return True
        if gnum is None or pnum is None:
            return False
        return abs(gnum - pnum) < 1e-9
    return gold == pred


def _to_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _abs_error(gold: Any, pred: Any) -> float | None:
    gnum = _to_number(gold)
    pnum = _to_number(pred)
    if gnum is None or pnum is None:
        return None
    return abs(gnum - pnum)


def _build_bucket_tags(
    *,
    input_text: str,
    gold_output: dict[str, Any] | None,
    parse_ok: bool,
    schema_ok: bool,
    threshold: int,
) -> list[str]:
    tags: list[str] = []
    if len(input_text) > threshold:
        tags.append("long_input")
    if any(ord(ch) > 127 for ch in input_text):
        tags.append("has_non_ascii")
    if not parse_ok:
        tags.append("parse_fail")
    if not schema_ok:
        tags.append("schema_fail")
    if isinstance(gold_output, dict) and gold_output.get("category") is None:
        tags.append("missing_category")
    return tags


def _guess_error_type(
    *,
    mismatch_fields: list[str],
    parse_ok: bool,
    schema_ok: bool,
    pred_row: dict[str, Any],
    pred_output: dict[str, Any] | None,
) -> str:
    if not parse_ok:
        raw = str(pred_row.get("raw_response", ""))
        if raw and "}" not in raw:
            return "truncation_suspected"
        return "extraction_error"
    if not schema_ok:
        return "type_error"
    if len(mismatch_fields) >= 4:
        return "multi_field_error"
    if any("category" in f or "stock_count" in f for f in mismatch_fields):
        if pred_output is not None and (
            pred_output.get("category") is None
            or _get_path(pred_output, "availability.stock_count") is None
        ):
            return "null_handling_error"
    if any("price" in f or "tax" in f for f in mismatch_fields):
        return "normalization_error"
    if any("title" in f or "product_type" in f for f in mismatch_fields):
        return "hallucination"
    return "extraction_error"


def _sample_badcases_by_type(records: list[dict[str, Any]], per_type: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["error_type_guess"]].append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for error_type, items in grouped.items():
        result[error_type] = [
            {
                "idx": item["idx"],
                "mismatch_fields": item["mismatch_fields"],
                "bucket_tags": item["bucket_tags"],
                "input_preview": item["input_preview"][:200],
            }
            for item in items[:per_type]
        ]
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    rank = max(0, math.ceil((percentile / 100) * len(arr)) - 1)
    return round(arr[rank], 6)


if __name__ == "__main__":
    main()
