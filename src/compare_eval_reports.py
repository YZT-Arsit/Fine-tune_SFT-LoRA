from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HIGHER_IS_BETTER = {
    "json_parse_rate",
    "schema_valid_rate",
}

LOWER_IS_BETTER = {
    "price_abs_error_avg",
    "price_abs_error_p50",
    "price_abs_error_p95",
    "price_excl_tax_abs_error_avg",
    "price_excl_tax_abs_error_p50",
    "price_excl_tax_abs_error_p95",
    "price_incl_tax_abs_error_avg",
    "price_incl_tax_abs_error_p50",
    "price_incl_tax_abs_error_p95",
    "tax_abs_error_avg",
    "tax_abs_error_p50",
    "tax_abs_error_p95",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and LoRA evaluation reports")
    parser.add_argument("--baseline_report", default="outputs/baseline/baseline_eval_report.json")
    parser.add_argument("--lora_report", default="outputs/lora_eval/val_generation_report.json")
    parser.add_argument("--out", default="outputs/analysis/model_comparison.json")
    parser.add_argument("--md_out", default="outputs/analysis/model_comparison.md")
    return parser.parse_args()


def compare_reports(baseline_report_path: Path, lora_report_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    lora = json.loads(lora_report_path.read_text(encoding="utf-8"))

    comparison = {
        "baseline_model": baseline.get("model_name"),
        "lora_model": lora.get("model_name"),
        "total_evaluated": {
            "baseline": baseline.get("total_evaluated"),
            "lora": lora.get("total_evaluated"),
        },
        "headline_metrics": _compare_headline_metrics(baseline, lora),
        "field_accuracy_delta": _compare_mapping(
            baseline.get("field_accuracy", {}),
            lora.get("field_accuracy", {}),
            higher_is_better=True,
        ),
        "numeric_error_delta": _compare_mapping(
            baseline.get("numeric_error", {}),
            lora.get("numeric_error", {}),
            higher_is_better=False,
        ),
        "wins": _summarize_wins(baseline, lora),
        "interview_summary": _build_interview_summary(baseline, lora),
    }
    return comparison


def _compare_headline_metrics(baseline: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for key in ("json_parse_rate", "schema_valid_rate"):
        metrics[key] = _build_delta_entry(baseline.get(key), lora.get(key), higher_is_better=True)
    return metrics


def _compare_mapping(
    baseline_values: dict[str, Any],
    lora_values: dict[str, Any],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    keys = sorted(set(baseline_values) | set(lora_values))
    return {
        key: _build_delta_entry(
            baseline_values.get(key),
            lora_values.get(key),
            higher_is_better=(key not in LOWER_IS_BETTER if higher_is_better else False),
        )
        for key in keys
    }


def _build_delta_entry(baseline_value: Any, lora_value: Any, *, higher_is_better: bool) -> dict[str, Any]:
    delta = None
    improved = None
    if _is_number(baseline_value) and _is_number(lora_value):
        delta = round(float(lora_value) - float(baseline_value), 4)
        improved = delta >= 0 if higher_is_better else delta <= 0
    return {
        "baseline": baseline_value,
        "lora": lora_value,
        "delta": delta,
        "improved": improved,
    }


def _summarize_wins(baseline: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    wins: dict[str, list[str]] = {"headline": [], "field_accuracy": [], "numeric_error": []}
    headline = _compare_headline_metrics(baseline, lora)
    for key, value in headline.items():
        if value["improved"]:
            wins["headline"].append(key)

    for key, value in _compare_mapping(
        baseline.get("field_accuracy", {}),
        lora.get("field_accuracy", {}),
        higher_is_better=True,
    ).items():
        if value["improved"]:
            wins["field_accuracy"].append(key)

    for key, value in _compare_mapping(
        baseline.get("numeric_error", {}),
        lora.get("numeric_error", {}),
        higher_is_better=False,
    ).items():
        if value["improved"]:
            wins["numeric_error"].append(key)
    return wins


def _build_interview_summary(baseline: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    parse_delta = _safe_delta(lora.get("json_parse_rate"), baseline.get("json_parse_rate"))
    schema_delta = _safe_delta(lora.get("schema_valid_rate"), baseline.get("schema_valid_rate"))
    strongest_fields = _top_improvements(
        _compare_mapping(baseline.get("field_accuracy", {}), lora.get("field_accuracy", {}), higher_is_better=True)
    )
    strongest_numeric = _top_improvements(
        _compare_mapping(baseline.get("numeric_error", {}), lora.get("numeric_error", {}), higher_is_better=False),
        reverse=False,
    )
    return {
        "elevator_pitch": (
            "我先用基座模型建立了结构化抽取 baseline，再基于 badcase 分析做数据治理与回环增强，"
            "最后用 QLoRA 在单卡 24GB 环境完成微调，并用统一验证集做同口径评测。"
        ),
        "quantified_takeaway": (
            f"LoRA 相比 baseline 的 JSON 解析率变化 {parse_delta:+.4f}，"
            f"Schema 合法率变化 {schema_delta:+.4f}。"
            if parse_delta is not None and schema_delta is not None
            else "LoRA 与 baseline 已按统一口径完成对比，可直接查看 headline_metrics / field_accuracy_delta / numeric_error_delta。"
        ),
        "top_field_gains": strongest_fields,
        "top_numeric_improvements": strongest_numeric,
        "risk_notes": [
            "如果 eval_loss 为 NaN，不影响任务级评测结论，面试时以 parse/schema/field accuracy 为主。",
            "若某些字段未提升，需要结合 badcase bucket 解释数据分布和 schema 约束带来的上限。",
        ],
    }


def _top_improvements(metric_map: dict[str, dict[str, Any]], *, reverse: bool = True) -> list[dict[str, Any]]:
    ranked = []
    for key, value in metric_map.items():
        delta = value.get("delta")
        if delta is None or math.isnan(delta):
            continue
        ranked.append({"metric": key, "delta": delta, "baseline": value.get("baseline"), "lora": value.get("lora")})
    ranked.sort(key=lambda item: item["delta"], reverse=reverse)
    return ranked[:5]


def write_markdown_summary(out_path: Path, comparison: dict[str, Any]) -> None:
    headline = comparison["headline_metrics"]
    lines = [
        "# Baseline vs LoRA 对比",
        "",
        f"- Baseline: `{comparison.get('baseline_model')}`",
        f"- LoRA: `{comparison.get('lora_model')}`",
        f"- Baseline evaluated: `{comparison['total_evaluated'].get('baseline')}`",
        f"- LoRA evaluated: `{comparison['total_evaluated'].get('lora')}`",
        "",
        "## Headline Metrics",
        "",
    ]
    for key, value in headline.items():
        lines.append(
            f"- `{key}`: baseline={value['baseline']} | lora={value['lora']} | delta={value['delta']} | improved={value['improved']}"
        )

    lines.extend(["", "## Interview Summary", ""])
    interview_summary = comparison["interview_summary"]
    lines.append(interview_summary["elevator_pitch"])
    lines.append("")
    lines.append(interview_summary["quantified_takeaway"])
    lines.append("")
    lines.append("### Top Field Gains")
    for item in interview_summary["top_field_gains"]:
        lines.append(
            f"- `{item['metric']}`: baseline={item['baseline']} | lora={item['lora']} | delta={item['delta']}"
        )
    lines.append("")
    lines.append("### Top Numeric Improvements")
    for item in interview_summary["top_numeric_improvements"]:
        lines.append(
            f"- `{item['metric']}`: baseline={item['baseline']} | lora={item['lora']} | delta={item['delta']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_delta(left: Any, right: Any) -> float | None:
    if not _is_number(left) or not _is_number(right):
        return None
    return round(float(left) - float(right), 4)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    args = parse_args()
    baseline_report_path = Path(args.baseline_report)
    lora_report_path = Path(args.lora_report)
    comparison = compare_reports(baseline_report_path, lora_report_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_summary(Path(args.md_out), comparison)
    print(
        json.dumps(
            {
                "comparison_out": str(out_path),
                "markdown_out": args.md_out,
                "baseline_model": comparison["baseline_model"],
                "lora_model": comparison["lora_model"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
