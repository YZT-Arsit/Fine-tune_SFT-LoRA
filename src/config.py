from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_URL = "https://books.toscrape.com/"
DEFAULT_LIST_PATH = ""
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; BooksToScrapeTrainer/1.0; +https://books.toscrape.com/)"
)


@dataclass(frozen=True)
class RuntimeConfig:
    base_url: str = DEFAULT_BASE_URL
    list_path: str = DEFAULT_LIST_PATH
    user_agent: str = DEFAULT_USER_AGENT
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 2.0
    timeout_seconds: float = 20.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    output_root: Path = Path("outputs")
    save_html: bool = False

    @property
    def parsed_dir(self) -> Path:
        return self.output_root / "parsed"

    @property
    def parsed_products_path(self) -> Path:
        return self.parsed_dir / "products.jsonl"

    @property
    def sft_dir(self) -> Path:
        return self.output_root / "sft"

    @property
    def sft_train_path(self) -> Path:
        return self.sft_dir / "train.jsonl"

    @property
    def raw_html_dir(self) -> Path:
        return self.output_root / "raw_html"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def failed_urls_path(self) -> Path:
        return self.logs_dir / "failed_urls.txt"

    @property
    def blocked_urls_path(self) -> Path:
        return self.logs_dir / "blocked_urls.txt"


def ensure_output_dirs(config: RuntimeConfig) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.parsed_dir.mkdir(parents=True, exist_ok=True)
    config.sft_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    if config.save_html:
        config.raw_html_dir.mkdir(parents=True, exist_ok=True)
