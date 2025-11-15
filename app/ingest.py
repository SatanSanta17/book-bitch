from pathlib import Path
from typing import Iterable

import pdfplumber
from openai import OpenAI
from pinecone import Pinecone, PodSpec, ServerlessSpec

from .chunker import clean_text, chunk_text
from .config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

_pinecone_client: Pinecone | None = None
_pinecone_index: any = None
EMBEDDING_DIMENSION = 1536

def init_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        pc = _get_pinecone_client()
        _ensure_index_exists(pc)
        _pinecone_index = pc.Index(settings.pinecone_index)
    return _pinecone_index


def _get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if settings.use_faiss:
        raise RuntimeError("FAISS path not implemented yet.")
    if _pinecone_client is None:
        if not settings.pinecone_api_key:
            raise RuntimeError("Pinecone API key is not configured.")
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def _ensure_index_exists(pc: Pinecone) -> None:
    existing = pc.list_indexes()
    index_names = existing.names() if hasattr(existing, "names") else [item["name"] for item in existing]
    if settings.pinecone_index in index_names:
        return

    if settings.pinecone_env:
        parts = settings.pinecone_env.split("-")
        if len(parts) >= 2:
            cloud = parts[-1]
            region = "-".join(parts[:-1])
            if cloud in {"aws", "gcp", "azure"}:
                pc.create_index(
                    name=settings.pinecone_index,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                return
        # Fallback to pod-based spec if env is non-standard
        pc.create_index(
            name=settings.pinecone_index,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=PodSpec(environment=settings.pinecone_env, pod_type="p1.x1"),
        )
        return

    # Default serverless spec if no environment configured
    pc.create_index(
        name=settings.pinecone_index,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def init_pinecone() -> any:
    global _pinecone_index
    if _pinecone_index is None:
        pc = _get_pinecone_client()
        _ensure_index_exists(pc)
        _pinecone_index = pc.Index(settings.pinecone_index)
    return _pinecone_index


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