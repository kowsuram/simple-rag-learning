"""Configuration settings for Healthcare RAG Agent."""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

# Load environment variables from .env file in project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _default_ai_provider() -> str:
    configured_provider = os.getenv("AI_PROVIDER")
    if configured_provider:
        return configured_provider.lower()
    if os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN"):
        return "cloudflare"
    return "ollama"


def _default_llm_model() -> str:
    configured_model = os.getenv("LLM_MODEL")
    if configured_model:
        return configured_model
    return (
        "@cf/meta/llama-3.2-3b-instruct"
        if _default_ai_provider() == "cloudflare"
        else "llama3.2:latest"
    )


def _default_embedding_model() -> str:
    configured_model = os.getenv("EMBEDDING_MODEL")
    if configured_model:
        return configured_model
    return (
        "@cf/baai/bge-base-en-v1.5"
        if _default_ai_provider() == "cloudflare"
        else "nomic-embed-text"
    )


class AppConfig(BaseModel):
    """Application settings schema."""
    
    # Database
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    db_host: str = Field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = Field(default_factory=lambda: int(os.getenv("DB_PORT", "5433")))
    db_name: str = Field(default_factory=lambda: os.getenv("DB_NAME", "healthcaredb"))
    db_user: str = Field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgrespassword"))

    # AI provider
    ai_provider: Literal["ollama", "cloudflare"] = Field(default_factory=_default_ai_provider)
    cloudflare_account_id: str = Field(default_factory=lambda: os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    cloudflare_api_token: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("CLOUDFLARE_API_TOKEN", ""))
    )
    cloudflare_ai_gateway_id: str = Field(
        default_factory=lambda: (
            os.getenv("CLOUDFLARE_AI_GATEWAY_ID")
            or os.getenv("CLOUDFLARE_GATEWAY_ID")
            or os.getenv("AI_GATEWAY_ID", "")
        )
    )
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = Field(default_factory=_default_llm_model)
    embedding_model: str = Field(default_factory=_default_embedding_model)
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

    @property
    def cloudflare_ai_base_url(self) -> str:
        """Return the Cloudflare Workers AI OpenAI-compatible API base URL."""
        if not self.cloudflare_account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID is required when AI_PROVIDER=cloudflare.")
        return (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.cloudflare_account_id}/ai/v1"
        )

    @property
    def cloudflare_headers(self) -> dict[str, str]:
        """Return authentication and AI Gateway routing headers."""
        token = self.cloudflare_api_token.get_secret_value()
        if not token:
            raise ValueError("CLOUDFLARE_API_TOKEN is required when AI_PROVIDER=cloudflare.")
        if not self.cloudflare_ai_gateway_id:
            raise ValueError(
                "CLOUDFLARE_AI_GATEWAY_ID is required to route Workers AI through AI Gateway."
            )
        return {
            "Authorization": f"Bearer {token}",
            "cf-aig-gateway-id": self.cloudflare_ai_gateway_id,
        }


# Global singleton instance
config = AppConfig()
