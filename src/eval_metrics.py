from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any


def parse_json_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = (raw_text or "").strip()
    if not text:
        return None, "empty_output"

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "parse_fail"

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "parse_fail"

    if not isinstance(parsed, dict):
        return None, "parse_fail"
    return parsed, None


def validate_prediction_schema(payload: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "invalid_output_json"

    if not isinstance(payload.get("title"), str):
        return False, "schema_title_invalid"

    category = payload.get("category")
    if category is not None and not isinstance(category, str):
        return False, "schema_category_invalid"

    if not _is_number(payload.get("price")):
        return False, "schema_price_invalid"

    if payload.get("currency") != "GBP":
        return False, "schema_currency_invalid"

    availability = payload.get("availability")
    if not isinstance(availability, dict):
        return False, "schema_availability_invalid"
    if not isinstance(availability.get("in_stock"), bool):
        return False, "schema_in_stock_invalid"
    stock_count = availability.get("stock_count")
    if stock_count is not None and not _is_int(stock_count):
        return False, "schema_stock_count_invalid"

    rating = payload.get("rating")
    if rating is not None and (not _is_int(rating) or rating < 1 or rating > 5):
        return False, "schema_rating_invalid"

    key_attributes = payload.get("key_attributes")
    if not isinstance(key_attributes, dict):
        return False, "schema_key_attributes_invalid"

    if not isinstance(key_attributes.get("upc"), str):
        return False, "schema_upc_invalid"
    if not isinstance(key_attributes.get("product_type"), str):
        return False, "schema_product_type_invalid"
    if not _is_number(key_attributes.get("price_excl_tax")):
        return False, "schema_price_excl_tax_invalid"
    if not _is_number(key_attributes.get("price_incl_tax")):
        return False, "schema_price_incl_tax_invalid"
    if not _is_number(key_attributes.get("tax")):
        return False, "schema_tax_invalid"
    if not isinstance(key_attributes.get("availability_text"), str):
        return False, "schema_availability_text_invalid"
    if not _is_int(key_attributes.get("review_count")):
        return False, "schema_review_count_invalid"

    return True, None


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_ok_count = sum(1 for row in rows if row["parse_ok"])
    schema_ok_count = sum(1 for row in rows if row["schema_ok"])

    title_matches = 0
    category_matches = 0
    in_stock_matches = 0
    stock_count_matches = 0
    upc_matches = 0

    price_errors: list[float] = []
    price_excl_tax_errors: list[float] = []
    tax_errors: list[float] = []
    failure_reasons = Counter()
    sample_badcases: list[dict[str, Any]] = []

    for row in rows:
        gt = row["ground_truth"]
        pred = row.get("parsed_json")
        reason = row.get("error_reason")
        if reason:
            failure_reasons[reason] += 1
            if len(sample_badcases) < 10:
                sample_badcases.append(
                    {
                        "sample_id": row["sample_id"],
                        "reason": reason,
                        "raw_response_preview": (row.get("raw_response") or "")[:500],
                    }
                )

        if not isinstance(pred, dict):
            continue

        title_matches += pred.get("title") == gt.get("title")
        category_matches += pred.get("category") == gt.get("category")

        pred_availability = pred.get("availability") or {}
        gt_availability = gt.get("availability") or {}
        in_stock_matches += pred_availability.get("in_stock") == gt_availability.get("in_stock")
        stock_count_matches += pred_availability.get("stock_count") == gt_availability.get("stock_count")

        pred_key = pred.get("key_attributes") or {}
        gt_key = gt.get("key_attributes") or {}
        upc_matches += pred_key.get("upc") == gt_key.get("upc")

        _append_abs_error(price_errors, pred.get("price"), gt.get("price"))
        _append_abs_error(price_excl_tax_errors, pred_key.get("price_excl_tax"), gt_key.get("price_excl_tax"))
        _append_abs_error(tax_errors, pred_key.get("tax"), gt_key.get("tax"))

    return {
        "total_samples": total,
        "json_parse_rate": _safe_rate(parse_ok_count, total),
        "schema_valid_rate": _safe_rate(schema_ok_count, total),
        "field_accuracy": {
            "title_exact_match_rate": _safe_rate(title_matches, total),
            "category_exact_match_rate": _safe_rate(category_matches, total),
            "in_stock_match_rate": _safe_rate(in_stock_matches, total),
            "stock_count_match_rate": _safe_rate(stock_count_matches, total),
            "upc_match_rate": _safe_rate(upc_matches, total),
        },
        "numeric_error": {
            "price_abs_error_avg": _avg(price_errors),
            "price_abs_error_p50": _percentile(price_errors, 50),
            "price_abs_error_p95": _percentile(price_errors, 95),
            "price_excl_tax_abs_error_avg": _avg(price_excl_tax_errors),
            "tax_abs_error_avg": _avg(tax_errors),
        },
        "failure_reasons_top10": failure_reasons.most_common(10),
        "sample_badcases": sample_badcases,
    }


def _append_abs_error(bucket: list[float], predicted: Any, truth: Any) -> None:
    if not _is_number(predicted) or not _is_number(truth):
        return
    bucket.append(abs(float(predicted) - float(truth)))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(float(ordered[rank]), 4)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
