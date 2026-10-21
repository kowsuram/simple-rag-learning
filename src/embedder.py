"""Ollama embedding client with latency tracking and visual vector previews."""

import time
from typing import List, Optional
import requests
from src.config import config
from src.logger import log_embedding_details


class OllamaEmbedder:
    """Generates dense vector embeddings using local Ollama model."""

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name or config.embedding_model
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.endpoint = f"{self.base_url}/api/embeddings"
        self.expected_dim = config.embedding_dim

    def get_embedding(self, text: str, verbose: bool = True) -> List[float]:
        """Request embedding vector for a given string from Ollama.
        
        Educational aspect:
          Text is converted by the neural network into an N-dimensional vector
          (e.g., 768 numbers) representing its semantic coordinates in latent space.
        """
        start_time = time.perf_counter()
        payload = {
            "model": self.model_name,
            "prompt": text
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.endpoint}. "
                f"Ensure Ollama is running ('ollama serve') and model '{self.model_name}' is pulled."
            ) from e

        embedding = data.get("embedding", [])
        if not embedding:
            raise ValueError(f"Ollama returned an empty embedding array for model '{self.model_name}'.")

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
