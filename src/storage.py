from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable


class JsonlStorage:
    def __init__(
        self,
        parsed_path: Path,
        failed_urls_path: Path,
        blocked_urls_path: Path,
        raw_html_dir: Path | None,
        logger: logging.Logger,
    ) -> None:
        self.parsed_path = parsed_path
        self.failed_urls_path = failed_urls_path
        self.blocked_urls_path = blocked_urls_path
        self.raw_html_dir = raw_html_dir
        self.logger = logger
        self._seen_urls = self._load_seen_urls()

    def has_product(self, product_url: str) -> bool:
        return product_url in self._seen_urls

    def append_product(self, record: dict[str, Any]) -> bool:
        product_url = record.get("product_url", "")
        if not product_url or product_url in self._seen_urls:
            return False

        self.parsed_path.parent.mkdir(parents=True, exist_ok=True)
        with self.parsed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        self._seen_urls.add(product_url)
        self.logger.info("Stored product %s", product_url)
        return True

    def append_failed_url(self, url: str, reason: str) -> None:
        self.failed_urls_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failed_urls_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{url}\t{reason}\n")
            handle.flush()

    def append_blocked_url(self, url: str, reason: str) -> None:
        self.blocked_urls_path.parent.mkdir(parents=True, exist_ok=True)
        with self.blocked_urls_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{url}\t{reason}\n")
            handle.flush()

    def save_html(self, url: str, html: str) -> Path | None:
        if self.raw_html_dir is None:
            return None
        self.raw_html_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html"
        path = self.raw_html_dir / filename
        path.write_text(html, encoding="utf-8")
        return path

    def iter_records(self) -> Iterable[dict[str, Any]]:
        if not self.parsed_path.exists():
            return
        with self.parsed_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Skipping invalid JSONL line %s in %s: %s",
                        line_number,
                        self.parsed_path,
                        exc,
                    )

    def _load_seen_urls(self) -> set[str]:
        seen: set[str] = set()
        if not self.parsed_path.exists():
            return seen
        with self.parsed_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                product_url = record.get("product_url")
                if product_url:
                    seen.add(product_url)
        return seen


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
