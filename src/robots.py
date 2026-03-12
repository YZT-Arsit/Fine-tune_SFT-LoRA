from __future__ import annotations

from dataclasses import dataclass
import logging
from urllib.parse import urljoin, urlparse

import requests


@dataclass(frozen=True)
class RobotsPolicy:
    rules: list[tuple[str, str]]
    allow_all: bool
    source_available: bool

    def can_fetch(self, path: str) -> bool:
        if self.allow_all:
            return True

        matched_directive = ""
        matched_length = -1
        for directive, prefix in self.rules:
            if not prefix:
                if directive == "disallow":
                    continue
            if path.startswith(prefix):
                prefix_length = len(prefix)
                if prefix_length > matched_length:
                    matched_directive = directive
                    matched_length = prefix_length
                elif prefix_length == matched_length and directive == "allow":
                    matched_directive = directive

        if matched_length == -1:
            return True
        return matched_directive == "allow"


def _fallback_policy(strict_robots: bool) -> RobotsPolicy:
    return RobotsPolicy(
        rules=[],
        allow_all=not strict_robots,
        source_available=False,
    )


def _parse_robots_text(text: str) -> RobotsPolicy:
    rules: list[tuple[str, str]] = []
    current_group_is_star = False
    current_group_has_rules = False
    in_group = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue

        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()

        if field == "user-agent":
            if in_group and current_group_has_rules:
                current_group_is_star = False
                current_group_has_rules = False
            in_group = True
            if value == "*":
                current_group_is_star = True
            continue

        if field not in {"allow", "disallow"}:
            continue
        if not in_group:
            continue

        current_group_has_rules = True
        if not current_group_is_star:
            continue
        if field == "disallow" and value == "":
            continue
        rules.append((field, value))

    return RobotsPolicy(
        rules=rules,
        allow_all=False,
        source_available=True,
    )


def self_check_robots_404_allows_when_not_strict() -> None:
    policy = _fallback_policy(strict_robots=False)
    assert policy.allow_all is True
    assert policy.can_fetch("/catalogue/page-2.html") is True


class RobotsChecker:
    def __init__(
        self,
        base_url: str,
        user_agent: str,
        logger: logging.Logger,
        strict_robots: bool = False,
    ) -> None:
        self.base_url = base_url
        self.user_agent = user_agent
        self.logger = logger
        self.strict_robots = strict_robots
        self._policy: RobotsPolicy | None = None
        self._loaded = False
        self._robots_url = urljoin(base_url, "/robots.txt")

    def load(self, timeout: float = 10.0) -> None:
        if self._loaded:
            return
        try:
            response = requests.get(
                self._robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=timeout,
            )
            if response.status_code == 404:
                self._policy = _fallback_policy(self.strict_robots)
                if self.strict_robots:
                    self.logger.error(
                        "robots.txt not found at %s; strict robots mode forbids crawling.",
                        self._robots_url,
                    )
                else:
                    self.logger.warning(
                        "robots.txt not found at %s; defaulting to allow-all and continuing.",
                        self._robots_url,
                    )
                self._loaded = True
                return

            response.raise_for_status()
            self._policy = _parse_robots_text(response.text)
            self._loaded = True
            self.logger.info("Loaded robots.txt from %s", self._robots_url)
        except requests.RequestException as exc:
            self._policy = _fallback_policy(self.strict_robots)
            if self.strict_robots:
                self.logger.error(
                    "Failed to fetch robots.txt (%s). Strict robots mode forbids crawling.",
                    exc,
                )
            else:
                self.logger.warning(
                    "Failed to fetch robots.txt (%s). Defaulting to allow-all and continuing.",
                    exc,
                )
            self._loaded = True

    def can_fetch(self, url: str) -> bool:
        if not self._loaded:
            self.load()
        if self._policy is None:
            return False
        normalized = self._normalize_url(url)
        allowed = self._policy.can_fetch(normalized)
        if not allowed:
            self.logger.warning("Blocked by robots.txt, skipping %s", normalized)
        return allowed

    @property
    def robots_url(self) -> str:
        return self._robots_url

    @property
    def allow_all(self) -> bool:
        return self._policy.allow_all if self._policy is not None else False

    @property
    def policy_available(self) -> bool:
        return self._policy.source_available if self._policy is not None else False

    def _normalize_url(self, url: str) -> str:
        absolute = urljoin(self.base_url, url)
        parsed = urlparse(absolute)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return path


self_check_robots_404_allows_when_not_strict()
