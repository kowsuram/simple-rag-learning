"""Configuration settings for Healthcare RAG Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file in project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class AppConfig(BaseModel):
    """Application settings schema."""
    
    # Database
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    db_host: str = Field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = Field(default_factory=lambda: int(os.getenv("DB_PORT", "5433")))
    db_name: str = Field(default_factory=lambda: os.getenv("DB_NAME", "healthcaredb"))
    db_user: str = Field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgrespassword"))

    # Ollama
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2:latest"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))

    # Chunking & Retrieval
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "450")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "80")))
    top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "3")))

    # Paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data" / "medications"

    @property
    def postgres_dsn(self) -> str:
        """Return DATABASE_URL when configured, otherwise build the local DSN."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )


# Global singleton instance
config = AppConfig()
