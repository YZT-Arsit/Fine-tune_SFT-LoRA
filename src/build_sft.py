from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .storage import load_jsonl, write_jsonl

try:
    from ftfy import fix_text as ftfy_fix_text
except ImportError:  # pragma: no cover
    ftfy_fix_text = None


INSTRUCTION = "请从给定的商品页面文本中抽取并规范化商品信息，输出符合 JSON Schema 的结果。"
MOJIBAKE_MARKERS = ("Â£", "â", "Ã", "â€™", "â€œ", "â€", "â€“")
FALLBACK_REPLACEMENTS = {
    "Â£": "£",
    "Â": "",
    "â": "’",
    "â": "‘",
    "â": "“",
    "â": "”",
    "â": "–",
    "â": "—",
    "â¦": "…",
}
RATING_TEXT_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
HALF_DUP_THRESHOLD = 0.92
MIN_DESCRIPTION_KEEP_RATIO = 0.30
INVALID_SENTINEL = object()


def build_sft_dataset(
    input_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    *,
    strict: bool = True,
    drop_default_category: bool = False,
    val_out_path: Path | None = None,
    split_by: str = "upc",
    val_ratio: float = 0.05,
) -> dict[str, Any]:
    products = load_jsonl(input_path)
    normalized_ratio = _normalize_val_ratio(val_ratio)
    report = _init_report(
        strict,
        split_by=split_by,
        val_ratio=normalized_ratio,
        val_enabled=val_out_path is not None,
    )
    accepted: list[dict[str, Any]] = []

    for product in products:
        report["total_read"] += 1
        try:
            prepared = _prepare_sample(
                product,
                report,
                strict=strict,
                drop_default_category=drop_default_category,
            )
        except Exception:  # noqa: BLE001
            _drop_sample(report, product, "other_exception", _sample_excerpt(product))
            continue

        if prepared is None:
            continue
        accepted.append(prepared)

    train_rows, val_rows, split_usage = _split_rows(
        accepted,
        val_out_path=val_out_path,
        split_by=split_by,
        val_ratio=normalized_ratio,
    )

    write_jsonl(output_path, [row["sft_row"] for row in train_rows])
    if val_out_path is not None:
        write_jsonl(val_out_path, [row["sft_row"] for row in val_rows])

    report["total_written"] = len(accepted)
    report["dropped_count"] = report["total_read"] - len(accepted)
    report["output_quality"]["schema_valid_rate"] = round(
        (len(accepted) / report["total_read"]) if report["total_read"] else 0.0,
        4,
    )
    normalized_products = [row["normalized_product"] for row in accepted]
    report["stats"] = _build_stats(normalized_products)
    report["text_quality"] = _build_text_quality(report.pop("_description_metrics"))
    report["label_quality"]["rating_distribution"] = _rating_distribution(normalized_products)
    report["splits"] = _build_split_report(train_rows, val_rows)
    report["split_config"]["upc_samples_count"] = split_usage["upc"]
    report["split_config"]["product_url_fallback_count"] = split_usage["product_url"]

    final_report_path = report_path or Path("outputs/sft/data_quality_report.json")
    final_report_path.parent.mkdir(parents=True, exist_ok=True)

    rating_audit_path = final_report_path.parent / "rating_audit_samples.json"
    rating_audit_samples = _build_rating_audit_samples(accepted)
    rating_audit_path.write_text(
        json.dumps(rating_audit_samples, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report["rating_audit_path"] = str(rating_audit_path)

    final_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _prepare_sample(
    product: dict[str, Any],
    report: dict[str, Any],
    *,
    strict: bool,
    drop_default_category: bool,
) -> dict[str, Any] | None:
    normalized_product = _sanitize_product(
        product,
        report,
        strict=strict,
        drop_default_category=drop_default_category,
    )
    if normalized_product is None:
        return None

    input_text = _format_input(normalized_product)
    output_payload = _build_output_payload(normalized_product)
    output_json = _stable_json_dumps(output_payload)

    try:
        json.loads(output_json)
    except json.JSONDecodeError:
        _drop_sample(report, product, "invalid_output_json", output_json[:200])
        return None

    if _contains_mojibake(output_json):
        _drop_sample(report, product, "mojibake_detected", output_json[:200])
        return None

    return {
        "product_url": normalized_product["product_url"],
        "normalized_product": normalized_product,
        "sft_row": {
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": output_json,
        },
    }


def _sanitize_product(
    product: dict[str, Any],
    report: dict[str, Any],
    *,
    strict: bool,
    drop_default_category: bool,
) -> dict[str, Any] | None:
    title = _fix_text(product.get("title"), report)
    category = _normalize_category(product.get("category"), report)
    if category is None and drop_default_category:
        _drop_sample(report, product, "category_default_dropped", _sample_excerpt(product))
        return None

    description_raw = _fix_text(product.get("description"), report)
    description = _clean_description(description_raw, product, report)
    availability_text = _fix_text(product.get("availability_text"), report)
    product_information = _sanitize_product_information(product.get("product_information"), report)

    price = _parse_float(product.get("price"))
    if price is None or price <= 0:
        _drop_sample(report, product, "price_parse_failed", _sample_excerpt(product))
        return None

    rating = _normalize_rating(
        product.get("rating"),
        product.get("raw_rating_text"),
        report,
        product,
        strict=strict,
    )
    if rating is INVALID_SENTINEL:
        return None

    key_attributes = _normalize_key_attributes(
        product_information,
        report,
        product,
        strict=strict,
    )
    if key_attributes is None:
        return None

    return {
        "product_url": str(product.get("product_url", "")),
        "raw_breadcrumb": product.get("raw_breadcrumb") if isinstance(product.get("raw_breadcrumb"), list) else [],
        "raw_rating_text": _clean_optional_str(product.get("raw_rating_text"), report),
        "title": title,
        "category": category,
        "price": round(price, 2),
        "currency": "GBP",
        "availability_text": availability_text,
        "in_stock": bool(product.get("in_stock", False)),
        "stock_count": _parse_int(product.get("stock_count")),
        "rating": rating,
        "description": description,
        "product_information": product_information,
        "key_attributes": key_attributes,
    }


def _sanitize_product_information(raw_info: Any, report: dict[str, Any]) -> dict[str, str]:
    if not isinstance(raw_info, dict):
        return {}

    cleaned: dict[str, str] = {}
    for key, value in raw_info.items():
        clean_key = _fix_text(key, report)
        clean_value = _fix_text(value, report)
        cleaned[str(clean_key)] = str(clean_value)
    return cleaned


def _build_output_payload(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "availability": {
            "in_stock": product["in_stock"],
            "stock_count": product["stock_count"],
        },
        "category": product["category"],
        "currency": "GBP",
        "key_attributes": product["key_attributes"],
        "price": product["price"],
        "rating": product["rating"],
        "title": product["title"],
    }


def _format_input(product: dict[str, Any]) -> str:
    info_lines = [
        f"{key}: {value}"
        for key, value in (product.get("product_information") or {}).items()
    ]
    category_value = "null" if product.get("category") is None else product.get("category", "")
    parts = [
        f"Title: {product.get('title', '')}",
        f"Category: {category_value}",
        f"Price: {product.get('price'):.2f} GBP",
        f"Availability: {product.get('availability_text', '')}",
        f"Description: {product.get('description', '')}",
        "Product Information:",
        "\n".join(info_lines) if info_lines else "(empty)",
    ]
    return "\n".join(parts).strip()


def _normalize_key_attributes(
    product_information: dict[str, str],
    report: dict[str, Any],
    product: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, Any] | None:
    key_attributes = {
        "availability_text": _clean_optional_str(product_information.get("Availability"), report),
        "price_excl_tax": None,
        "price_incl_tax": None,
        "product_type": _clean_optional_str(product_information.get("Product Type"), report),
        "review_count": None,
        "tax": None,
        "upc": _clean_optional_str(product_information.get("UPC"), report),
    }

    numeric_specs = [
        ("Price (excl. tax)", "price_excl_tax", _extract_money),
        ("Price (incl. tax)", "price_incl_tax", _extract_money),
        ("Tax", "tax", _extract_money),
        ("Number of reviews", "review_count", _extract_int),
    ]

    for source_key, target_key, parser in numeric_specs:
        raw_value = product_information.get(source_key)
        if raw_value in (None, ""):
            continue
        parsed_value = parser(raw_value, report)
        if parsed_value is None:
            if strict:
                _drop_sample(report, product, "key_attr_parse_failed", f"{source_key}: {raw_value}"[:200])
                return None
            key_attributes[target_key] = None
            continue
        report["fixes_applied"]["key_attr_numeric_cast"] += 1
        key_attributes[target_key] = parsed_value

    return key_attributes


def _normalize_category(raw_value: Any, report: dict[str, Any]) -> str | None:
    category = _clean_optional_str(raw_value, report)
    if category in {"", "Default"}:
        report["fixes_applied"]["category_default_to_null"] += 1
        report["category_missing_count"] += 1
        return None
    return category


def _normalize_rating(
    raw_value: Any,
    raw_rating_text: Any,
    report: dict[str, Any],
    product: dict[str, Any],
    *,
    strict: bool,
) -> int | None | object:
    numeric_rating = _parse_int(raw_value)
    mapped_rating = _map_rating_text(raw_rating_text)

    if numeric_rating is None:
        numeric_rating = mapped_rating
    elif mapped_rating is not None and numeric_rating != mapped_rating:
        report["label_quality"]["rating_mismatch_count"] += 1
        if strict:
            _drop_sample(
                report,
                product,
                "rating_out_of_range",
                f"raw_rating={raw_value}, raw_rating_text={raw_rating_text}",
            )
            return INVALID_SENTINEL
        numeric_rating = mapped_rating

    if numeric_rating is None:
        return None

    if 1 <= numeric_rating <= 5:
        return numeric_rating

    report["label_quality"]["rating_out_of_range_count"] += 1
    if strict:
        _drop_sample(report, product, "rating_out_of_range", _sample_excerpt(product))
        return INVALID_SENTINEL
    return None


def _map_rating_text(raw_rating_text: Any) -> int | None:
    if raw_rating_text in (None, ""):
        return None
    text = str(raw_rating_text).strip().casefold()
    if text in RATING_TEXT_MAP:
        return RATING_TEXT_MAP[text]
    parts = re.split(r"[\s_-]+", text)
    for part in parts:
        if part in RATING_TEXT_MAP:
            return RATING_TEXT_MAP[part]
    return None


def _fix_text(value: Any, report: dict[str, Any]) -> str:
    text = "" if value is None else str(value)
    original = text
    if ftfy_fix_text is not None:
        text = ftfy_fix_text(text)
    else:
        text = _fallback_fix_text(text)
    if text != original:
        report["fixes_applied"]["mojibake_fixed"] += 1
    return _normalize_whitespace(text)


def _fallback_fix_text(text: str) -> str:
    if _contains_mojibake(text):
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    for bad, good in FALLBACK_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def _clean_description(text: str, product: dict[str, Any], report: dict[str, Any]) -> str:
    cleaned = _normalize_whitespace(text)
    before_len = len(cleaned)
    candidate = _sentence_level_dedup(cleaned)
    candidate = _trim_repeated_half(candidate)
    after_len = len(candidate)

    metrics = report["_description_metrics"]
    metrics["before_lengths"].append(before_len)
    metrics["after_lengths"].append(after_len)

    over_shrink = before_len > 0 and after_len < max(1, math.floor(before_len * MIN_DESCRIPTION_KEEP_RATIO))
    if over_shrink:
        metrics["over_shrink_count"] += 1
        candidate = cleaned
        after_len = before_len

    shrink_ratio = round((after_len / before_len), 4) if before_len else 1.0
    metrics["shrink_ratios"].append(shrink_ratio)

    if candidate != cleaned:
        metrics["dedup_applied_count"] += 1
        report["fixes_applied"]["description_deduped"] += 1
        if len(report["examples"]["sample_dedup_cases"]) < 10:
            report["examples"]["sample_dedup_cases"].append(
                {
                    "product_url": str(product.get("product_url", "")),
                    "before_len": before_len,
                    "after_len": after_len,
                    "shrink_ratio": shrink_ratio,
                    "before_head_preview": cleaned[:200],
                    "after_head_preview": candidate[:200],
                    "before_mid_preview": _mid_preview(cleaned),
                    "after_mid_preview": _mid_preview(candidate),
                }
            )

    return candidate


def _sentence_level_dedup(text: str) -> str:
    if not text:
        return ""
    pieces = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    seen: set[str] = set()
    deduped: list[str] = []

    for piece in pieces:
        normalized = _normalize_whitespace(piece)
        if not normalized:
            continue
        marker = re.sub(r"\s+", "", normalized).casefold()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(normalized)
    return " ".join(deduped)


def _trim_repeated_half(text: str, threshold: float = HALF_DUP_THRESHOLD) -> str:
    if len(text) < 40:
        return text
    midpoint = len(text) // 2
    left = _normalize_whitespace(text[:midpoint])
    right = _normalize_whitespace(text[midpoint:])
    if not left or not right:
        return text
    if SequenceMatcher(None, left, right).ratio() > threshold:
        return left
    return text


def _extract_money(value: Any, report: dict[str, Any]) -> float | None:
    text = _fix_text(value, report)
    if "£" in text:
        report["fixes_applied"]["currency_symbol_stripped"] += 1
    text = text.replace("£", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return round(float(match.group(0)), 2)
    except ValueError:
        return None


def _extract_int(value: Any, report: dict[str, Any]) -> int | None:
    text = _fix_text(value, report)
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if value in (None, ""):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if value in (None, ""):
        return None
    match = re.search(r"-?\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _clean_optional_str(value: Any, report: dict[str, Any]) -> str | None:
    text = _fix_text(value, report)
    return text or None


def _contains_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    val_out_path: Path | None,
    split_by: str,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if val_out_path is None or not rows or val_ratio <= 0:
        split_usage = {"upc": 0, "product_url": 0}
        for row in rows:
            _, actual_source = _resolve_group_value(row["normalized_product"], split_by)
            split_usage[actual_source] += 1
        return rows, [], split_usage

    threshold = int(round(val_ratio * 100))
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    split_usage = {"upc": 0, "product_url": 0}
    for row in rows:
        group_value, actual_source = _resolve_group_value(row["normalized_product"], split_by)
        split_usage[actual_source] += 1
        if _bucket(group_value) < threshold:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if not val_rows and train_rows:
        val_rows.append(train_rows.pop())
    if not train_rows and val_rows:
        train_rows.append(val_rows.pop())
    return train_rows, val_rows, split_usage


def _resolve_group_value(product: dict[str, Any], field: str) -> tuple[str, str]:
    if field == "upc":
        upc = (product.get("key_attributes") or {}).get("upc")
        if upc not in (None, ""):
            return str(upc), "upc"
        return str(product.get("product_url") or "null"), "product_url"

    value = product.get(field)
    if value in (None, ""):
        value = product.get("product_url")
        return str(value or "null"), "product_url"
    return str(value), field if field in {"upc", "product_url"} else "product_url"


def _mid_preview(text: str, width: int = 200) -> str:
    if len(text) <= width:
        return text
    start = max(0, (len(text) // 2) - (width // 2))
    end = start + width
    return text[start:end]


def _bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _normalize_val_ratio(val_ratio: float) -> float:
    if val_ratio < 0:
        return 0.0
    if val_ratio > 1:
        return 1.0
    return float(val_ratio)


def _init_report(
    strict: bool,
    *,
    split_by: str,
    val_ratio: float,
    val_enabled: bool,
) -> dict[str, Any]:
    return {
        "total_read": 0,
        "total_written": 0,
        "dropped_count": 0,
        "strict": strict,
        "category_missing_count": 0,
        "drop_reasons": {
            "invalid_output_json": 0,
            "mojibake_detected": 0,
            "price_parse_failed": 0,
            "key_attr_parse_failed": 0,
            "category_default_dropped": 0,
            "rating_out_of_range": 0,
            "other_exception": 0,
        },
        "fixes_applied": {
            "mojibake_fixed": 0,
            "currency_symbol_stripped": 0,
            "category_default_to_null": 0,
            "description_deduped": 0,
            "key_attr_numeric_cast": 0,
        },
        "text_quality": {},
        "output_quality": {
            "stable_json_serialization_enabled": True,
            "schema_valid_rate": 0.0,
        },
        "label_quality": {
            "rating_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "null": 0},
            "rating_out_of_range_count": 0,
            "rating_mismatch_count": 0,
        },
        "stats": {},
        "examples": {
            "sample_dedup_cases": [],
        },
        "sample_badcases": [],
        "split_config": {
            "enabled": val_enabled,
            "split_by": split_by,
            "val_ratio": val_ratio,
        },
        "splits": {},
        "_description_metrics": {
            "before_lengths": [],
            "after_lengths": [],
            "shrink_ratios": [],
            "dedup_applied_count": 0,
            "over_shrink_count": 0,
        },
    }


def _drop_sample(
    report: dict[str, Any],
    product: dict[str, Any],
    reason: str,
    snippet: str,
) -> None:
    report["drop_reasons"].setdefault(reason, 0)
    report["drop_reasons"][reason] += 1
    if len(report["sample_badcases"]) < 20:
        report["sample_badcases"].append(
            {
                "product_url": str(product.get("product_url", "")),
                "reason": reason,
                "snippet": snippet[:200],
            }
        )


def _sample_excerpt(product: dict[str, Any]) -> str:
    parts = [
        str(product.get("title", "")),
        str(product.get("category", "")),
        str(product.get("description", "")),
    ]
    return " | ".join(part for part in parts if part)[:200]


def _build_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "description_len_avg": 0,
            "description_len_p50": 0,
            "description_len_p95": 0,
            "price_avg": 0,
            "price_p50": 0,
            "price_p95": 0,
            "rating_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "null": 0},
            "category_top20": [],
        }

    description_lengths = [len(row.get("description", "")) for row in rows]
    prices = [float(row["price"]) for row in rows if isinstance(row.get("price"), (int, float))]
    category_counter = Counter(
        "null" if row.get("category") is None else str(row.get("category"))
        for row in rows
    )

    return {
        "description_len_avg": round(sum(description_lengths) / len(description_lengths), 2),
        "description_len_p50": _percentile(description_lengths, 50),
        "description_len_p95": _percentile(description_lengths, 95),
        "price_avg": round(sum(prices) / len(prices), 2) if prices else 0,
        "price_p50": _percentile(prices, 50),
        "price_p95": _percentile(prices, 95),
        "rating_distribution": _rating_distribution(rows),
        "category_top20": category_counter.most_common(20),
    }


def _build_text_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    before_lengths = metrics["before_lengths"]
    after_lengths = metrics["after_lengths"]
    shrink_ratios = metrics["shrink_ratios"]
    return {
        "description_len_before_avg": round(sum(before_lengths) / len(before_lengths), 2) if before_lengths else 0,
        "description_len_before_p50": _percentile(before_lengths, 50),
        "description_len_before_p95": _percentile(before_lengths, 95),
        "description_len_after_avg": round(sum(after_lengths) / len(after_lengths), 2) if after_lengths else 0,
        "description_len_after_p50": _percentile(after_lengths, 50),
        "description_len_after_p95": _percentile(after_lengths, 95),
        "description_dedup_applied_count": metrics["dedup_applied_count"],
        "description_dedup_avg_shrink_ratio": round(sum(shrink_ratios) / len(shrink_ratios), 4) if shrink_ratios else 1.0,
        "description_dedup_over_shrink_count": metrics["over_shrink_count"],
    }


def _rating_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    rating_counter = Counter(
        "null" if row.get("rating") is None else str(row.get("rating"))
        for row in rows
    )
    return {
        "1": rating_counter.get("1", 0),
        "2": rating_counter.get("2", 0),
        "3": rating_counter.get("3", 0),
        "4": rating_counter.get("4", 0),
        "5": rating_counter.get("5", 0),
        "null": rating_counter.get("null", 0),
    }


def _build_split_report(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_category_top10": _top_categories(train_rows, 10),
        "val_category_top10": _top_categories(val_rows, 10),
    }


def _top_categories(rows: list[dict[str, Any]], limit: int) -> list[tuple[str, int]]:
    counter = Counter(
        "null"
        if row["normalized_product"].get("category") is None
        else str(row["normalized_product"].get("category"))
        for row in rows
    )
    return counter.most_common(limit)


def _build_rating_audit_samples(
    rows: list[dict[str, Any]],
    sample_size: int = 20,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(row["product_url"].encode("utf-8")).hexdigest(),
    )
    samples: list[dict[str, Any]] = []
    for row in ordered[:sample_size]:
        product = row["normalized_product"]
        samples.append(
            {
                "product_url": product.get("product_url", ""),
                "raw_rating_text": product.get("raw_rating_text"),
                "mapped_rating": product.get("rating"),
            }
        )
    return samples


def _percentile(values: list[float | int], percentile: int) -> float:
    if not values:
        return 0
    sorted_values = sorted(values)
    rank = max(0, math.ceil((percentile / 100) * len(sorted_values)) - 1)
    return round(float(sorted_values[rank]), 2)
