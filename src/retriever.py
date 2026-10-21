"""Retrieval engine: Patient Query -> Ollama Embedding -> pgvector Cosine Search -> Context Assembly."""

import time
from typing import List, Dict, Any, Optional
from src.config import config
from src.db import db
from src.embedder import embedder
from src.logger import (
    console,
    log_step,
    log_retrieval_ranking
)


class HealthcareRetriever:
    """Handles semantic retrieval of medication chunks from pgvector."""

    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k or config.top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        medication_filter: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute full retrieval flow with step-by-step educational logs."""
        k = top_k or self.top_k

        # ----------------------------------------------------
        # RETRIEVAL STEP 1: EMBED THE PATIENT QUERY
        # ----------------------------------------------------
        if verbose:
            log_step(
                1,
                "Patient Query Embedding",
                f"Converting user query into a {config.embedding_dim}-dimensional vector representation using '{config.embedding_model}'."
            )

        query_embedding = embedder.get_embedding(query, verbose=verbose)

        # ----------------------------------------------------
        # RETRIEVAL STEP 2: PGVECTOR COSINE SIMILARITY SEARCH
        # ----------------------------------------------------
        if verbose:
            log_step(
                2,
                "pgvector Cosine Similarity Execution",
                "Querying PostgreSQL with SQL operator '<=>' (Cosine Distance). "
                "Ranking chunks by: Similarity = 1 - Cosine Distance."
            )

        start_time = time.perf_counter()
        results = db.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=k,
            medication_filter=medication_filter
        )
        search_latency_ms = (time.perf_counter() - start_time) * 1000

        if verbose:
            console.print(f"[dim]pgvector search completed in {search_latency_ms:.2f} ms[/dim]")
            log_retrieval_ranking(query, results)

        return results

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean, labeled reference context string."""
        if not chunks:
            return "NO RELEVANT MEDICATION INFORMATION FOUND IN KNOWLEDGE BASE."

        formatted_parts = []
        for i, chunk in enumerate(chunks, start=1):
            med_name = chunk.get("medication_name", "Unknown")
            section = chunk.get("section", "General")
            similarity = chunk.get("similarity", 0.0)
            content = chunk.get("content", "").strip()

            formatted_parts.append(
                f"--- REFERENCE CHUNK #{i} [Drug: {med_name} | Section: {section} | Match Confidence: {similarity * 100:.1f}%] ---\n"
                f"{content}\n"
            )

        return "\n".join(formatted_parts)


# Global retriever instance
retriever = HealthcareRetriever()
