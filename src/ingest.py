"""Ingestion pipeline: Document Loading -> Section Chunking -> Ollama Embedding -> pgvector Storage."""

import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import config
from src.db import db
from src.chunker import chunker
from src.embedder import embedder
from rich.panel import Panel
from src.logger import (
    console,
    log_header,
    log_step,
    log_document_loaded
)


def load_medication_files(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all medication JSON files from the data directory."""
    files = list(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No medication JSON files found in {data_dir}")

    monographs = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            monographs.append(data)
    return monographs


def run_ingestion(reset_db: bool = True, max_docs: int = None) -> int:
    """Run full educational ingestion pipeline."""
    log_header(
        "HEALTHCARE RAG: KNOWLEDGE INGESTION PIPELINE",
        "Converting Clinical Medication Monographs into pgvector Embeddings"
    )

    # ----------------------------------------------------
    # STEP 1: INITIALIZE DATABASE SCHEMA & PGVECTOR
    # ----------------------------------------------------
    log_step(
        1,
        "Database Initialization & pgvector Setup",
        "Enabling PostgreSQL 'vector' extension, setting up table schema with vector(768) column, and creating HNSW index."
    )
    db.init_db(reset=reset_db)

    # ----------------------------------------------------
    # STEP 2: LOAD SOURCE MEDICATION MONOGRAPHS
    # ----------------------------------------------------
    log_step(
        2,
        "Load Clinical Monographs (Source of Truth)",
        "Reading authoritative medication reference data from JSON monographs."
    )
    monographs = load_medication_files(config.data_dir)
    if max_docs:
        monographs = monographs[:max_docs]

    console.print(f"[bold green]Found {len(monographs)} medication monographs in {config.data_dir}[/bold green]\n")

    total_chunks_stored = 0

    for doc_idx, monograph in enumerate(monographs, start=1):
        med_name = monograph.get("medication_name", "Unknown")
        brand_names = monograph.get("brand_names", [])
        med_class = monograph.get("therapeutic_class", "")
        
        # Calculate total characters across all sections
        total_chars = sum(len(str(v)) for k, v in monograph.items() if isinstance(v, str))
        sections_count = len([k for k, v in monograph.items() if isinstance(v, str)])

        log_document_loaded(med_name, brand_names, med_class, sections_count, total_chars)

        # ----------------------------------------------------
        # STEP 3: SECTION-AWARE CHUNKING WITH OVERLAP
        # ----------------------------------------------------
        log_step(
            3,
            f"Chunking Document: {med_name}",
            f"Partitioning {med_name} into contextualized chunks (max size: {config.chunk_size} chars, overlap: {config.chunk_overlap} chars) "
            f"to prevent loss of clinical contraindications and dosage limits across boundaries."
        )
        chunks = chunker.chunk_monograph(monograph, verbose=True)
        console.print(f"[bold cyan]Generated {len(chunks)} chunks for {med_name}[/bold cyan]\n")

        # ----------------------------------------------------
        # STEP 4: EMBEDDING GENERATION
        # ----------------------------------------------------
        log_step(
            4,
            f"Generating Vector Embeddings: {med_name}",
            f"Sending each chunk to {config.ai_provider} model '{config.embedding_model}' to generate "
            f"{config.embedding_dim}-dimensional semantic vectors."
        )

        for i, chk in enumerate(chunks):
            # Generate vector embedding for the chunk text
            vector = embedder.get_embedding(chk["content"], verbose=True)
            chk["embedding"] = vector

        # ----------------------------------------------------
        # STEP 5: STORE CHUNKS & VECTORS IN PGVECTOR
        # ----------------------------------------------------
        log_step(
            5,
            f"Persisting into PostgreSQL + pgvector: {med_name}",
            f"Executing SQL INSERT statements to store text, metadata, and 768-D vectors into the 'medication_chunks' table."
        )

        for chk in chunks:
            chunk_id = db.insert_chunk(
                medication_name=chk["medication_name"],
                section=chk["section"],
                chunk_index=chk["chunk_index"],
                content=chk["content"],
                embedding=chk["embedding"],
                metadata=chk["metadata"],
                verbose=False
            )
            console.print(f"  [green][OK][/green] Saved Chunk #{chk['chunk_index'] + 1} ({chk['section']}) -> ID: [dim]{chunk_id}[/dim]")
            total_chunks_stored += 1

        console.print()

    # Final summary
    total_in_db = db.get_chunk_count()
    console.print(
        Panel(
            f"[bold green]Ingestion Complete![/bold green]\n"
            f"Total Monographs Ingested: {len(monographs)}\n"
            f"Total Chunks Stored: {total_chunks_stored}\n"
            f"Database Verification Count: {total_in_db} records in 'medication_chunks'",
            title="Ingestion Summary",
            border_style="bright_green"
        )
    )
    return total_chunks_stored


if __name__ == "__main__":
    run_ingestion(reset_db=True)
