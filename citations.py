"""Visual citations with bounding boxes, powered by LiteParse v2.

Given a PDF and a phrase (e.g. a snippet of a retrieved RAG chunk), this module
locates where the phrase appears on the page and renders the page screenshot
with the matching region(s) highlighted. Useful for showing users *where* an
answer came from in the source document.

Why a hand-rolled matcher instead of ``liteparse.search_items``:
the native ``search_items`` in liteparse 2.1.x rejects the ``TextItem`` objects
returned by ``parse()`` ("'TextItem' object is not an instance of 'PyTextItem'"),
so we match in pure Python. This is also more flexible: it handles phrases that
span several adjacent text items, which the docs note can happen.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

from liteparse import LiteParse

# (x, y, width, height) in PDF points, origin top-left.
BBox = Tuple[float, float, float, float]


def _normalize(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


# Markdown decorations that LiteParse adds but that never appear in the raw
# page text_items we search against.
_MD_PREFIX = re.compile(r"^\s*(#{1,6}\s+|>+\s*|[-*+]\s+|\d+[.)]\s+)")
_MD_INLINE = re.compile(r"[*_`~|]+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _clean_markdown_line(line: str) -> str:
    """Strip Markdown decoration from a single line, leaving plain text."""
    line = _MD_IMAGE.sub("", line)
    line = _MD_LINK.sub(r"\1", line)          # keep link text, drop the URL
    line = _MD_PREFIX.sub("", line)
    line = _MD_INLINE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def extract_search_phrases(chunk_text: str, min_words: int = 4,
                           max_words: int = 12, max_phrases: int = 8) -> List[str]:
    """Turn a Markdown chunk into clean plain-text phrases to highlight.

    Markdown headings/lists/tables don't match the raw page text items, so each
    line is de-decorated; short/noise lines are dropped and long lines are
    truncated to a leading window (long phrases rarely match a contiguous run)."""
    phrases: List[str] = []
    seen = set()
    for raw_line in (chunk_text or "").splitlines():
        cleaned = _clean_markdown_line(raw_line)
        words = cleaned.split()
        if len(words) < min_words:
            continue
        phrase = " ".join(words[:max_words])
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def find_matches(text_items, phrase: str, case_sensitive: bool = False,
                 max_span: int = 12) -> List[BBox]:
    """Return bounding boxes (PDF points) where ``phrase`` occurs in a page.

    Strategy:
      1. Single text item containing the phrase -> that item's box.
      2. Otherwise, a phrase spanning consecutive items -> the union box of the
         smallest run of items (up to ``max_span``) whose joined text contains it.
    """
    phrase = (phrase or "").strip()
    if not phrase:
        return []

    target = _normalize(phrase, case_sensitive)
    boxes: List[BBox] = []

    # 1) Single-item substring matches.
    for it in text_items:
        if target in _normalize(it.text, case_sensitive):
            boxes.append((it.x, it.y, it.width, it.height))
    if boxes:
        return boxes

    # 2) Spanning match across consecutive items.
    n = len(text_items)
    for i in range(n):
        combined = ""
        for j in range(i, min(i + max_span, n)):
            combined = (combined + " " + text_items[j].text).strip()
            if target in _normalize(combined, case_sensitive):
                window = text_items[i:j + 1]
                xs = [w.x for w in window]
                ys = [w.y for w in window]
                xe = [w.x + w.width for w in window]
                ye = [w.y + w.height for w in window]
                boxes.append((min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)))
                break
    return boxes


def _resolve_local_path(pdf_source: str):
    """Return (local_path, cleanup_flag). Downloads http(s) sources to a temp file
    because LiteParse.screenshot() needs a real file path (not bytes)."""
    if pdf_source.startswith("http"):
        response = requests.get(pdf_source)
        response.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(response.content)
        tmp.close()
        return tmp.name, True
    if not os.path.exists(pdf_source):
        raise FileNotFoundError(f"The file {pdf_source} does not exist.")
    return pdf_source, False


def _highlight_core(
    pdf_source: str,
    phrases: List[str],
    *,
    restrict_page: Optional[int] = None,
    dpi: float = 150,
    case_sensitive: bool = False,
    color: Tuple[int, int, int] = (255, 215, 0),
    fill_alpha: int = 80,
    outline_width: int = 2,
) -> Dict[int, bytes]:
    """Highlight one or more phrases on a PDF and return ``{page_num: png_bytes}``.

    When ``restrict_page`` is given, only that page is parsed and rendered
    (via LiteParse ``target_pages``), which keeps the explorer responsive — a
    retrieved chunk already knows which page it came from.
    """
    phrases = [p for p in phrases if p and p.strip()]
    if not phrases:
        return {}

    local_path, cleanup = _resolve_local_path(pdf_source)
    try:
        parser = LiteParse(
            output_format="markdown",
            dpi=dpi,
            target_pages=str(restrict_page) if restrict_page else None,
        )
        result = parser.parse(local_path)

        # Page -> list of boxes (PDF points), de-duplicated across phrases.
        matches: Dict[int, List[BBox]] = {}
        for page in result.pages:
            seen: set = set()
            page_boxes: List[BBox] = []
            for phrase in phrases:
                for box in find_matches(page.text_items, phrase, case_sensitive):
                    key = tuple(round(v, 1) for v in box)
                    if key not in seen:
                        seen.add(key)
                        page_boxes.append(box)
            if page_boxes:
                matches[page.page_num] = page_boxes
        if not matches:
            return {}

        pages_by_num = {p.page_num: p for p in result.pages}
        shots = parser.screenshot(local_path, page_numbers=list(matches.keys()))

        out: Dict[int, bytes] = {}
        for shot in shots:
            page = pages_by_num.get(shot.page_num)
            if page is None or not page.width or not page.height:
                continue

            # PDF points -> screenshot pixels (use the real ratio, not an assumed DPI).
            sx = shot.width / page.width
            sy = shot.height / page.height

            base = Image.open(io.BytesIO(shot.image_bytes)).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for (x, y, w, h) in matches[shot.page_num]:
                rect = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
                draw.rectangle(
                    rect,
                    fill=color + (fill_alpha,),
                    outline=color + (255,),
                    width=outline_width,
                )

            composed = Image.alpha_composite(base, overlay).convert("RGB")
            buf = io.BytesIO()
            composed.save(buf, format="PNG")
            out[shot.page_num] = buf.getvalue()

        return out
    finally:
        if cleanup:
            os.remove(local_path)


def highlight_phrase(pdf_source: str, phrase: str, **kwargs) -> Dict[int, bytes]:
    """Highlight a single phrase across all pages of a PDF.

    Returns ``{page_num: png_bytes}`` for pages with at least one match.
    ``pdf_source`` may be a local path or an http(s) URL.
    """
    return _highlight_core(pdf_source, [phrase], **kwargs)


def highlight_chunk(pdf_source: str, chunk_text: str,
                    page: Optional[int] = None, **kwargs) -> Dict[int, bytes]:
    """Highlight where a retrieved RAG chunk lives in its source PDF.

    The chunk's Markdown is reduced to clean plain-text phrases and matched
    against the page. Pass ``page`` (from the chunk's metadata) to parse/render
    only that page for speed. Returns ``{page_num: png_bytes}``.
    """
    phrases = extract_search_phrases(chunk_text)
    if not phrases:
        return {}
    return _highlight_core(pdf_source, phrases, restrict_page=page, **kwargs)
