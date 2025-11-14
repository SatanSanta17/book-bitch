from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    pinecone_api_key: str | None = None
    pinecone_env: str | None = None
    pinecone_index: str = "books-demo"
    use_faiss: bool = False
    data_dir: str = "./data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()