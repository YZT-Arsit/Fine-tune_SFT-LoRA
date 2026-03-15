from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .baseline import run_baseline
from .build_sft import build_sft_dataset
from .compare_eval_reports import compare_reports, write_markdown_summary
from .config import RuntimeConfig, ensure_output_dirs
from .env_loader import load_project_env
from .eval_lora_model import run_lora_eval
from .fetch import Fetcher, RobotsBlockedError
from .parse_detail import parse_product_detail
from .parse_list import parse_list_page
from .robots import RobotsChecker
from .storage import JsonlStorage, load_jsonl


def main() -> None:
    load_project_env()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "crawl":
        handle_crawl(args)
    elif args.command == "build_sft":
        handle_build_sft(args)
    elif args.command == "baseline":
        handle_baseline(args)
    elif args.command == "lora_eval":
        handle_lora_eval(args)
    elif args.command == "compare_eval":
        handle_compare_eval(args)
    elif args.command == "stats":
        handle_stats(args)
    else:
        parser.print_help()


def handle_crawl(args: argparse.Namespace) -> None:
    config = RuntimeConfig(
        output_root=Path(args.out),
        save_html=args.save_html,
        user_agent=args.user_agent,
    )
    ensure_output_dirs(config)
    logger = setup_logger(config.logs_dir / "crawl.log", "crawl")
    robots_checker = RobotsChecker(
        config.base_url,
        config.user_agent,
        logger,
        strict_robots=args.strict_robots,
    )
    robots_checker.load()
    fetcher = Fetcher(config, logger, robots_checker)
    raw_html_dir = config.raw_html_dir if config.save_html else None
    storage = JsonlStorage(
        parsed_path=config.parsed_products_path,
        failed_urls_path=config.failed_urls_path,
        blocked_urls_path=config.blocked_urls_path,
        raw_html_dir=raw_html_dir,
        logger=logger,
    )

    pages_crawled = 0
    products_written = 0
    failed_urls_count = 0
    blocked_urls_count = 0

    if args.strict_robots and not robots_checker.policy_available:
        logger.error("Strict robots mode enabled and robots policy is unavailable; stopping crawl.")
        storage.append_failed_url(robots_checker.robots_url, "robots_unavailable_in_strict_mode")
        failed_urls_count += 1
        print(
            json.dumps(
                {
                    "pages_crawled": pages_crawled,
                    "products_written": products_written,
                    "failed_urls_count": failed_urls_count,
                    "blocked_urls_count": blocked_urls_count,
                    "parsed_path": str(config.parsed_products_path),
                },
                ensure_ascii=False,
            )
        )
        return

    page_url = config.base_url
    stop_crawl = False

    while page_url and pages_crawled < args.max_pages:
        try:
            response = fetcher.get(page_url)
        except RobotsBlockedError as exc:
            logger.warning("List page blocked by robots %s: %s", page_url, exc)
            storage.append_blocked_url(page_url, str(exc))
            blocked_urls_count += 1
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch list page %s: %s", page_url, exc)
            storage.append_failed_url(page_url, f"list_fetch_error: {exc}")
            failed_urls_count += 1
            break

        pages_crawled += 1
        product_urls, next_page_url = parse_list_page(
            html=response.text,
            page_url=page_url,
            base_url=config.base_url,
        )
        logger.info(
            "Parsed list page %s with %s product urls", page_url, len(product_urls)
        )

        for product_url in product_urls:
            if storage.has_product(product_url):
                logger.info("Skipping already stored product %s", product_url)
                continue

            try:
                detail_response = fetcher.get(product_url)
                if config.save_html:
                    storage.save_html(product_url, detail_response.text)
                record = parse_product_detail(
                    html=detail_response.text,
                    page_url=product_url,
                    base_url=config.base_url,
                )
                if storage.append_product(record):
                    products_written += 1
            except RobotsBlockedError as exc:
                logger.warning("Robots blocked product %s: %s", product_url, exc)
                storage.append_blocked_url(product_url, str(exc))
                blocked_urls_count += 1
                if args.strict_robots:
                    stop_crawl = True
                    break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to process product %s: %s", product_url, exc)
                storage.append_failed_url(product_url, f"detail_error: {exc}")
                failed_urls_count += 1

        if stop_crawl:
            break

        page_url = next_page_url

    print(
        json.dumps(
            {
                "pages_crawled": pages_crawled,
                "products_written": products_written,
                "failed_urls_count": failed_urls_count,
                "blocked_urls_count": blocked_urls_count,
                "parsed_path": str(config.parsed_products_path),
            },
            ensure_ascii=False,
        )
    )


def handle_build_sft(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.out)
    report_path = Path(args.report)
    val_out_path = Path(args.val_out) if args.val_out else None
    report = build_sft_dataset(
        input_path,
        output_path,
        report_path=report_path,
        strict=args.strict,
        drop_default_category=args.drop_default_category,
        val_out_path=val_out_path,
        split_by=args.split_by,
        val_ratio=args.val_ratio,
    )
    logger = setup_logger(_build_sft_log_path(output_path, report_path), "build_sft")
    logger.info(
        "build_sft completed total_read=%s total_written=%s train_count=%s val_count=%s dropped_count=%s fixes=%s drops=%s",
        report["total_read"],
        report["total_written"],
        report["splits"]["train_count"],
        report["splits"]["val_count"],
        report["dropped_count"],
        json.dumps(report["fixes_applied"], ensure_ascii=False, sort_keys=True),
        json.dumps(report["drop_reasons"], ensure_ascii=False, sort_keys=True),
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "val_output": str(val_out_path) if val_out_path else None,
                "report": str(report_path),
                "rows_written": report["splits"]["train_count"],
                "total_written": report["total_written"],
                "val_written": report["splits"]["val_count"],
                "dropped_count": report["dropped_count"],
            },
            ensure_ascii=False,
        )
    )


def handle_stats(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    if not rows:
        print("No records found.")
        return

    total = len(rows)
    tracked_fields = [
        "title",
        "category",
        "price",
        "currency",
        "availability_text",
        "in_stock",
        "stock_count",
        "rating",
        "description",
        "product_information",
        "image_url",
    ]

    missing_rates = {
        field: round(
            sum(1 for row in rows if _is_missing(row.get(field))) / total,
            4,
        )
        for field in tracked_fields
    }

    avg_description_length = round(
        sum(len((row.get("description") or "")) for row in rows) / total,
        2,
    )

    category_counter = Counter((row.get("category") or "UNKNOWN") for row in rows)
    prices = [row.get("price") for row in rows if isinstance(row.get("price"), (int, float))]
    price_stats = _price_distribution(prices)

    print(f"Samples: {total}")
    print("Missing Rates:")
    for field, rate in missing_rates.items():
        print(f"  {field}: {rate:.2%}")
    print(f"Average description length: {avg_description_length}")
    print("Category Distribution:")
    for category, count in category_counter.most_common():
        print(f"  {category}: {count}")
    print("Price Distribution:")
    for label, value in price_stats.items():
        print(f"  {label}: {value}")


def handle_baseline(args: argparse.Namespace) -> None:
    val_path = Path(args.val)
    pred_out_path = Path(args.pred_out)
    report_out_path = Path(args.report_out)
    if args.mode == "api" and not os.getenv("LLM_API_KEY"):
        print(
            json.dumps(
                {
                    "error": "missing_api_key",
                    "message": "LLM_API_KEY is not set. Put it in .env or export it in your shell.",
                },
                ensure_ascii=False,
            )
        )
        return
    report = run_baseline(
        val_path,
        pred_out_path,
        report_out_path,
        mode=args.mode,
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    logger = setup_logger(_build_baseline_log_path(report_out_path), "baseline")
    logger.info(
        "baseline completed mode=%s model=%s evaluated=%s json_parse_rate=%s schema_valid_rate=%s",
        report["mode"],
        report["model_name"],
        report["total_evaluated"],
        report["json_parse_rate"],
        report["schema_valid_rate"],
    )
    print(
        json.dumps(
            {
                "val": str(val_path),
                "pred_out": str(pred_out_path),
                "report_out": str(report_out_path),
                "mode": report["mode"],
                "model_name": report["model_name"],
                "guided_json_enabled": report["guided_json_enabled"],
                "total_evaluated": report["total_evaluated"],
                "json_parse_rate": report["json_parse_rate"],
                "schema_valid_rate": report["schema_valid_rate"],
            },
            ensure_ascii=False,
        )
    )


def handle_lora_eval(args: argparse.Namespace) -> None:
    report = run_lora_eval(args)
    logger = setup_logger(_build_lora_eval_log_path(Path(args.report_out)), "lora_eval")
    logger.info(
        "lora_eval completed model=%s evaluated=%s json_parse_rate=%s schema_valid_rate=%s",
        report["model_name"],
        report["total_evaluated"],
        report["json_parse_rate"],
        report["schema_valid_rate"],
    )
    print(
        json.dumps(
            {
                "val": args.val_file,
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


def handle_compare_eval(args: argparse.Namespace) -> None:
    comparison = compare_reports(Path(args.baseline_report), Path(args.lora_report))
    comparison_path = Path(args.out)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_summary(Path(args.md_out), comparison)
    logger = setup_logger(_build_compare_eval_log_path(comparison_path), "compare_eval")
    logger.info(
        "compare_eval completed baseline=%s lora=%s json_parse_delta=%s schema_delta=%s",
        comparison["baseline_model"],
        comparison["lora_model"],
        comparison["headline_metrics"]["json_parse_rate"]["delta"],
        comparison["headline_metrics"]["schema_valid_rate"]["delta"],
    )
    print(
        json.dumps(
            {
                "comparison_out": str(comparison_path),
                "markdown_out": args.md_out,
                "baseline_model": comparison["baseline_model"],
                "lora_model": comparison["lora_model"],
            },
            ensure_ascii=False,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Books to Scrape data pipeline")
    subparsers = parser.add_subparsers(dest="command")

    crawl = subparsers.add_parser("crawl", help="Crawl product data")
    crawl.add_argument("--max_pages", type=int, default=50, help="Maximum list pages to crawl")
    crawl.add_argument("--out", default="outputs", help="Output directory")
    crawl.add_argument("--save-html", action="store_true", help="Save raw detail HTML")
    crawl.add_argument(
        "--strict_robots",
        action="store_true",
        help="Stop crawling when robots blocks a URL or robots.txt is unavailable",
    )
    crawl.add_argument(
        "--user-agent",
        default=RuntimeConfig.user_agent,
        help="Custom User-Agent string",
    )

    build_sft = subparsers.add_parser("build_sft", help="Build SFT dataset")
    build_sft.add_argument("--in", dest="input", required=True, help="Input parsed JSONL")
    build_sft.add_argument("--out", required=True, help="Output SFT JSONL")
    build_sft.add_argument(
        "--report",
        default="outputs/sft/data_quality_report.json",
        help="Output data quality report JSON",
    )
    build_sft.add_argument(
        "--strict",
        dest="strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop samples when critical field parsing fails",
    )
    build_sft.add_argument(
        "--drop_default_category",
        action="store_true",
        help="Drop samples whose category is empty or Default",
    )
    build_sft.add_argument(
        "--val_out",
        default=None,
        help="Optional validation JSONL output path",
    )
    build_sft.add_argument(
        "--split_by",
        choices=["upc", "product_url", "title", "category"],
        default="upc",
        help="Group field used for train/val split",
    )
    build_sft.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (0.0-1.0)",
    )

    baseline = subparsers.add_parser("baseline", help="Run baseline inference and evaluation")
    baseline.add_argument("--val", required=True, help="Validation JSONL path")
    baseline.add_argument(
        "--pred_out",
        default="outputs/baseline/baseline_predictions.jsonl",
        help="Prediction JSONL output path",
    )
    baseline.add_argument(
        "--report_out",
        default="outputs/baseline/baseline_eval_report.json",
        help="Evaluation report output path",
    )
    baseline.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="Inference mode",
    )
    baseline.add_argument(
        "--model_name",
        default=None,
        help="Model name override; otherwise uses LLM_MODEL",
    )
    baseline.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample cap for quick debugging",
    )
    baseline.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum generation tokens",
    )

    lora_eval = subparsers.add_parser("lora_eval", help="Evaluate LoRA adapter or merged model on validation set")
    lora_eval.add_argument("--val_file", default="outputs/sft/val.jsonl")
    lora_eval.add_argument("--model_path", default=None, help="Merged model directory")
    lora_eval.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    lora_eval.add_argument("--adapter_dir", default=None, help="Adapter directory or training output directory")
    lora_eval.add_argument(
        "--pred_out",
        default="outputs/lora_eval/val_predictions.jsonl",
        help="LoRA predictions JSONL output path",
    )
    lora_eval.add_argument(
        "--report_out",
        default="outputs/lora_eval/val_generation_report.json",
        help="LoRA evaluation report output path",
    )
    lora_eval.add_argument("--max_eval_samples", type=int, default=0, help="0 means evaluate all validation samples")
    lora_eval.add_argument("--max_new_tokens", type=int, default=256)
    lora_eval.add_argument("--temperature", type=float, default=0.0)
    lora_eval.add_argument("--top_p", type=float, default=1.0)
    lora_eval.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    lora_eval.add_argument("--device_map", default="auto")

    compare_eval = subparsers.add_parser("compare_eval", help="Compare baseline and LoRA evaluation reports")
    compare_eval.add_argument("--baseline_report", default="outputs/baseline/baseline_eval_report.json")
    compare_eval.add_argument("--lora_report", default="outputs/lora_eval/val_generation_report.json")
    compare_eval.add_argument("--out", default="outputs/analysis/model_comparison.json")
    compare_eval.add_argument("--md_out", default="outputs/analysis/model_comparison.md")

    stats = subparsers.add_parser("stats", help="Show dataset statistics")
    stats.add_argument("--in", dest="input", required=True, help="Input parsed JSONL")

    return parser


def setup_logger(log_path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _build_sft_log_path(output_path: Path, report_path: Path) -> Path:
    candidates = [output_path, report_path]
    for candidate in candidates:
        if candidate.parent.name == "sft":
            return candidate.parent.parent / "logs" / "build_sft.log"
    return report_path.parent / "logs" / "build_sft.log"


def _build_baseline_log_path(report_path: Path) -> Path:
    if report_path.parent.name == "baseline":
        return report_path.parent.parent / "logs" / "baseline.log"
    return report_path.parent / "logs" / "baseline.log"


def _build_lora_eval_log_path(report_path: Path) -> Path:
    if report_path.parent.name == "lora_eval":
        return report_path.parent.parent / "logs" / "lora_eval.log"
    return report_path.parent / "logs" / "lora_eval.log"


def _build_compare_eval_log_path(report_path: Path) -> Path:
    if report_path.parent.name == "analysis":
        return report_path.parent.parent / "logs" / "compare_eval.log"
    return report_path.parent / "logs" / "compare_eval.log"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _price_distribution(prices: list[float]) -> dict[str, str]:
    if not prices:
        return {"count": "0", "min": "n/a", "median": "n/a", "max": "n/a"}
    sorted_prices = sorted(prices)
    count = len(sorted_prices)
    middle = count // 2
    if count % 2 == 1:
        median = sorted_prices[middle]
    else:
        median = (sorted_prices[middle - 1] + sorted_prices[middle]) / 2
    return {
        "count": str(count),
        "min": f"{sorted_prices[0]:.2f}",
        "median": f"{median:.2f}",
        "max": f"{sorted_prices[-1]:.2f}",
    }


if __name__ == "__main__":
    main()
