from pydantic import BaseModel


class UploadResponse(BaseModel):
    status: str
    chunks: int
    book_id: str


class AskRequest(BaseModel):
    question: str
    book_id: str


class MatchMetadata(BaseModel):
    book_id: str
    page: int | None = None
    chunk_idx: int | None = None
    text: str | None = None


class AskResponse(BaseModel):
    answer: str
    evidence: list[MatchMetadata]

class BookInfo(BaseModel):
    book_id: str
    filename: str