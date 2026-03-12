from __future__ import annotations

from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup


def parse_list_page(html: str, page_url: str, base_url: str) -> tuple[list[str], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    product_urls: list[str] = []

    for article in soup.select("article.product_pod h3 a"):
        href = article.get("href")
        if not href:
            continue
        product_urls.append(_normalize_product_url(href, page_url, base_url))

    next_link = soup.select_one("li.next a")
    next_page_url = None
    if next_link and next_link.get("href"):
        next_page_url = urljoin(page_url, next_link["href"])

    return product_urls, next_page_url


def _normalize_product_url(href: str, page_url: str, base_url: str) -> str:
    del base_url
    return urljoin(page_url, href)
