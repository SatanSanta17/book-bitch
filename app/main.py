import json
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from .config import Settings, get_settings
from .ingest import ingest_pdf, init_pinecone
from .models import AskRequest, AskResponse, BookInfo, MatchMetadata, UploadResponse

settings = get_settings()
data_dir = Path(settings.data_dir)
data_dir.mkdir(parents=True, exist_ok=True)
books_file = data_dir / "books.json"

app = FastAPI(title="Book Bitch RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=settings.openai_api_key)

def get_index() -> any:
    if settings.use_faiss:
        raise HTTPException(status_code=501, detail="FAISS backend not implemented yet.")
    return init_pinecone()


def load_books() -> list[BookInfo]:
    if not books_file.exists():
        return []
    with books_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [BookInfo(**item) for item in data]


def save_books(books: list[BookInfo]) -> None:
    with books_file.open("w", encoding="utf-8") as f:
        json.dump([book.dict() for book in books], f, indent=2)


@app.get("/books", response_model=list[BookInfo])
def list_books(_: Settings = Depends(get_settings)):
    return load_books()


@app.post("/upload/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), _: Settings = Depends(get_settings)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    book_id = Path(file.filename).stem
    dest = data_dir / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunk_count = ingest_pdf(dest, book_id)

    books = load_books()
    if not any(book.book_id == book_id for book in books):
        books.append(BookInfo(book_id=book_id, filename=file.filename))
        save_books(books)

    return UploadResponse(status="ok", chunks=chunk_count, book_id=book_id)


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest, index: any = Depends(get_index)):
    books = load_books()
    if not any(book.book_id == payload.book_id for book in books):
        raise HTTPException(status_code=404, detail="Unknown book_id; upload the book first.")

    query_embedding = openai_client.embeddings.create(
        input=[payload.question],
        model=settings.openai_embed_model,
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"book_id": {"$eq": payload.book_id}},
    )

    matches = results.matches
    extracts = []
    for match in matches:
        metadata = match.metadata
        extracts.append(MatchMetadata(**metadata))

    context = "\n\n".join(
        [f"[Page {m.page}] {m.text}" for m in extracts if m.text]
    ) or "No context retrieved."

    system_message = (
        "You are a tutor that MUST ONLY use the provided extracts. "
        "If the answer cannot be found in the extracts, reply "
        "'Insufficient evidence in book' and suggest where to look."
    )
    user_prompt = (
        f"Question: {payload.question}\n\n"
        f"Extracts:\n{context}\n\n"
        "Return a short answer and list evidence lines with page numbers."
    )

    completion = openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    answer = completion.choices[0].message.content
    return AskResponse(answer=answer, evidence=extracts)