from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

CURRENCY_SYMBOL_MAP = {
    "£": "GBP",
    "$": "USD",
    "€": "EUR",
}
NOISE_CATEGORY_PATTERNS = ("add a comment", "review", "reviews", "comment")


def parse_product_detail(html: str, page_url: str, base_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    title = _safe_text(soup.select_one("div.product_main h1"))
    raw_breadcrumb = _parse_raw_breadcrumb(soup)
    category = _parse_category(raw_breadcrumb)
    price_text = _safe_text(soup.select_one("p.price_color"))
    price, currency = _parse_price(price_text)
    availability_text = _normalize_space(_safe_text(soup.select_one("p.availability")))
    in_stock = "in stock" in availability_text.lower()
    stock_count = _parse_stock_count(availability_text)
    rating_node = soup.select_one("p.star-rating")
    rating = _parse_rating(rating_node)
    raw_rating_text = _parse_raw_rating_text(rating_node)
    description = _parse_description(soup)
    product_information = _parse_product_information(soup)
    image_url = _parse_image_url(soup, page_url, base_url)

    return {
        "product_url": page_url,
        "title": title,
        "raw_breadcrumb": raw_breadcrumb,
        "category": category,
        "price": price,
        "currency": currency,
        "availability_text": availability_text,
        "in_stock": in_stock,
        "stock_count": stock_count,
        "rating": rating,
        "raw_rating_text": raw_rating_text,
        "description": description,
        "product_information": product_information,
        "image_url": image_url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_raw_breadcrumb(soup: BeautifulSoup) -> list[str]:
    breadcrumb_items: list[str] = []
    for node in soup.select("ul.breadcrumb li"):
        text = _normalize_space(_safe_text(node))
        if text:
            breadcrumb_items.append(text)
    return breadcrumb_items


def _parse_category(raw_breadcrumb: list[str]) -> str | None:
    filtered = []
    for token in raw_breadcrumb:
        normalized = _normalize_space(token)
        lowered = normalized.casefold()
        if normalized in {"", "Home", "Books"}:
            continue
        if any(pattern in lowered for pattern in NOISE_CATEGORY_PATTERNS):
            continue
        filtered.append(normalized)
    if not filtered:
        return None
    return filtered[0]


def _parse_price(price_text: str) -> tuple[float | None, str]:
    if not price_text:
        return None, ""
    currency = ""
    for symbol, code in CURRENCY_SYMBOL_MAP.items():
        if symbol in price_text:
            currency = code
            price_text = price_text.replace(symbol, "")
            break
    cleaned = re.sub(r"[^0-9.]+", "", price_text)
    try:
        return float(cleaned), currency
    except ValueError:
        return None, currency


def _parse_stock_count(text: str) -> int | None:
    if not text:
        return None
    match = re.search(r"\((\d+)\s+available\)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_rating(node: Any) -> int | None:
    if node is None:
        return None
    classes = node.get("class", [])
    for class_name in classes:
        if class_name in RATING_MAP:
            return RATING_MAP[class_name]
    return None


def _parse_raw_rating_text(node: Any) -> str:
    if node is None:
        return ""
    classes = node.get("class", [])
    for class_name in classes:
        if class_name in RATING_MAP:
            return class_name
    return " ".join(
        str(class_name)
        for class_name in classes
        if class_name and class_name != "star-rating"
    )


def _parse_description(soup: BeautifulSoup) -> str:
    description_header = soup.select_one("#product_description")
    if not description_header:
        return ""
    paragraph = description_header.find_next_sibling("p")
    if paragraph is None:
        return ""
    return _normalize_space(_safe_text(paragraph))


def _parse_product_information(soup: BeautifulSoup) -> dict[str, str]:
    table = soup.select_one("table.table.table-striped")
    if table is None:
        return {}
    result: dict[str, str] = {}
    for row in table.select("tr"):
        header = _safe_text(row.select_one("th"))
        value = _normalize_space(_safe_text(row.select_one("td")))
        if header:
            result[header] = value
    return result


def _parse_image_url(soup: BeautifulSoup, page_url: str, base_url: str) -> str:
    image = soup.select_one("div.carousel-inner img, div.item.active img, div.thumbnail img")
    if image is None:
        image = soup.select_one("img")
    if image is None:
        return ""
    src = image.get("src", "")
    if not src:
        return ""
    absolute = urljoin(page_url, src)
    if "/../" in absolute:
        absolute = absolute.replace("/../", "/")
    if absolute.startswith(base_url):
        return absolute
    return urljoin(base_url, src)


def _safe_text(node: Any) -> str:
    if node is None:
        return ""
    return node.get_text(" ", strip=True)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
