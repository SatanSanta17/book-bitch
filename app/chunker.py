import re
from typing import Iterable


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap: int = 80,
    approx_chars_per_token: int = 4,
) -> Iterable[str]:
    if not text:
        return []

    window = max_tokens * approx_chars_per_token
    stride = max(1, window - overlap * approx_chars_per_token)
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + window, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start += stride
    return chunks