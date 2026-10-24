"""Embedding client with latency tracking and visual vector previews."""

import time
from typing import List, Optional
import requests
from src.config import config
from src.logger import log_embedding_details


class OllamaEmbedder:
    """Generates dense vector embeddings using the configured AI provider."""

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = config.ai_provider
        self.model_name = model_name or config.embedding_model
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.endpoint = (
            f"{config.cloudflare_ai_base_url}/embeddings"
            if self.provider == "cloudflare"
            else f"{self.base_url}/api/embeddings"
        )
        self.expected_dim = config.embedding_dim

    def get_embedding(self, text: str, verbose: bool = True) -> List[float]:
        """Request an embedding vector for a given string.
        
        Educational aspect:
          Text is converted by the neural network into an N-dimensional vector
          (e.g., 768 numbers) representing its semantic coordinates in latent space.
        """
        start_time = time.perf_counter()
        if self.provider == "cloudflare":
            payload = {"model": self.model_name, "input": text}
            headers = config.cloudflare_headers
        else:
            payload = {"model": self.model_name, "prompt": text}
            headers = None

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to request an embedding from {self.provider} at {self.endpoint}."
            ) from e

        if self.provider == "cloudflare":
            rows = data.get("data")
            if rows is None and isinstance(data.get("result"), dict):
                rows = data["result"].get("data")
            if rows and isinstance(rows[0], dict):
                embedding = rows[0].get("embedding", [])
            elif rows and isinstance(rows[0], list):
                embedding = rows[0]
            else:
                embedding = []
        else:
            embedding = data.get("embedding", [])

        if not embedding:
            raise ValueError(
                f"{self.provider} returned an empty embedding for model '{self.model_name}'."
            )
        embedding = [float(value) for value in embedding]
        if len(embedding) != self.expected_dim:
            raise ValueError(
                f"Embedding model '{self.model_name}' returned {len(embedding)} dimensions; "
                f"the database expects {self.expected_dim}."
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if verbose:
            log_embedding_details(
                text_snippet=text,
                model=self.model_name,
                dimension=len(embedding),
                sample_vector=embedding,
                latency_ms=elapsed_ms
            )

        return embedding

    def embed_batch(self, texts: List[str], verbose: bool = False) -> List[List[float]]:
        """Embed a list of text strings sequentially."""
        return [self.get_embedding(t, verbose=verbose) for t in texts]


# Global embedder instance
embedder = OllamaEmbedder()
