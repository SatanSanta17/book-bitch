from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    groq_api_key: str | None = None
    groq_model: str = "llama3-8b-8192"  

    pinecone_api_key: str | None = None
    pinecone_env: str | None = None
    pinecone_index: str = "book-bitch"

    use_faiss: bool = False
    data_dir: str = "./data"
    
    llm_provider: str = Field(default="openai", description="openai or groq")


    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()