import os
from pathlib import Path
from typing import Iterable

import pdfplumber
from openai import OpenAI
import pinecone

from .chunker import clean_text, chunk_text
from .config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

_pinecone_initialized = False
_index = None


def init_pinecone() -> pinecone.index.Index:
    global _pinecone_initialized, _index
    if settings.use_faiss:
        raise RuntimeError("FAISS path not implemented yet.")
    if not _pinecone_initialized:
        pinecone.init(api_key=settings.pinecone_api_key, environment=settings.pinecone_env)
        if settings.pinecone_index not in pinecone.list_indexes():
            pinecone.create_index(settings.pinecone_index, dimension=1536, metric="cosine")
        _index = pinecone.Index(settings.pinecone_index)
        _pinecone_initialized = True
    return _index


def extract_text_from_pdf(path: str | Path) -> Iterable[tuple[int, str]]:
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            yield page_number, text


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=settings.openai_embed_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_pdf(path: str | Path, book_id: str) -> int:
    index = init_pinecone()
    pages = list(extract_text_from_pdf(path))

    all_chunks = []
    for page, raw_text in pages:
        cleaned = clean_text(raw_text)
        for chunk_idx, chunk in enumerate(chunk_text(cleaned)):
            vector_id = f"{book_id}-{page}-{chunk_idx}"
            metadata = {"book_id": book_id, "page": page, "chunk_idx": chunk_idx, "text": chunk}
            all_chunks.append((vector_id, chunk, metadata))

    batch_size = 32
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        texts = [item[1] for item in batch]
        embeddings = embed_texts(texts)
        index.upsert([(item[0], vector, item[2]) for item, vector in zip(batch, embeddings)])
    return len(all_chunks)