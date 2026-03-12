from __future__ import annotations

import logging
import random
import time
from typing import Optional

import requests

from .config import RuntimeConfig
from .robots import RobotsChecker


class RobotsBlockedError(PermissionError):
    pass


class Fetcher:
    def __init__(
        self,
        config: RuntimeConfig,
        logger: logging.Logger,
        robots_checker: RobotsChecker,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.robots_checker = robots_checker
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self._last_request_at = 0.0

    def get(self, url: str) -> requests.Response:
        if not self.robots_checker.can_fetch(url):
            raise RobotsBlockedError(f"Blocked by robots.txt: {url}")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self.session.get(url, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                self._normalize_response_encoding(response)
                self._last_request_at = time.monotonic()
                self.logger.info("Fetched %s (attempt=%s)", url, attempt)
                return response
            except requests.RequestException as exc:
                last_error = exc
                self._last_request_at = time.monotonic()
                self.logger.warning(
                    "Request failed for %s (attempt=%s/%s): %s",
                    url,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    backoff = self.config.backoff_factor ** (attempt - 1)
                    sleep_for = backoff + random.uniform(0.1, 0.5)
                    self.logger.info("Backing off %.2f seconds before retrying %s", sleep_for, url)
                    time.sleep(sleep_for)
        assert last_error is not None
        raise last_error

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        jitter = random.uniform(
            self.config.min_delay_seconds,
            self.config.max_delay_seconds,
        )
        elapsed = now - self._last_request_at
        if self._last_request_at and elapsed < jitter:
            sleep_for = jitter - elapsed
            self.logger.debug("Sleeping %.2f seconds before next request", sleep_for)
            time.sleep(sleep_for)

    def _normalize_response_encoding(self, response: requests.Response) -> None:
        encoding = (response.encoding or "").lower()
        if encoding in {"", "iso-8859-1", "latin-1"}:
            apparent = response.apparent_encoding
            if apparent:
                response.encoding = apparent
