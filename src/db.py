"""PostgreSQL + pgvector database manager."""

import json
from typing import List, Dict, Any, Optional
import psycopg
from pgvector.psycopg import register_vector
from src.config import config
from src.logger import log_sql_operation, console


class DatabaseManager:
    """Manages PostgreSQL connection, pgvector tables, and vector queries."""

    def __init__(self):
        self.dsn = config.postgres_dsn
        self.dim = config.embedding_dim

    def get_connection(self) -> psycopg.Connection:
        """Create and return a PostgreSQL connection with pgvector registered."""
        conn = psycopg.connect(self.dsn, autocommit=True)
        try:
            register_vector(conn)
        except Exception:
            # If vector extension is not enabled yet in fresh DB, enable it first
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)
        return conn

    def init_db(self, reset: bool = False) -> None:
        """Initialize pgvector extension, table schema, and HNSW vector index."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Enable pgvector extension
                sql_ext = "CREATE EXTENSION IF NOT EXISTS vector;"
                log_sql_operation("Enable pgvector Extension", sql_ext)
                cur.execute(sql_ext)

                if reset:
                    sql_drop = "DROP TABLE IF EXISTS medication_chunks CASCADE;"
                    log_sql_operation("Drop Existing Chunks Table (Reset)", sql_drop)
                    cur.execute(sql_drop)

                # 2. Create chunks table
                sql_table = f"""
                CREATE TABLE IF NOT EXISTS medication_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    medication_name VARCHAR(100) NOT NULL,
                    section VARCHAR(100) NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
                log_sql_operation("Create medication_chunks Table", sql_table)
                cur.execute(sql_table)

                # 3. Create HNSW vector index for ultra-fast cosine similarity search
                sql_index = """
                CREATE INDEX IF NOT EXISTS medication_chunks_embedding_idx 
                ON medication_chunks 
                USING hnsw (embedding vector_cosine_ops);
                """
                log_sql_operation("Create HNSW Vector Cosine Index", sql_index)
                cur.execute(sql_index)

    def insert_chunk(
        self,
        medication_name: str,
        section: str,
        chunk_index: int,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> str:
        """Insert a single text chunk with its embedding vector into pgvector."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                sql_insert = """
                INSERT INTO medication_chunks (
                    medication_name, section, chunk_index, content, embedding, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                meta_json = json.dumps(metadata or {})
                if verbose:
                    sample_vec_str = f"[{embedding[0]:.4f}, {embedding[1]:.4f}, ... (dim={len(embedding)})]"
                    param_desc = (
                        f"medication={medication_name}, section={section}, "
                        f"chunk_index={chunk_index}, vector={sample_vec_str}"
                    )
                    log_sql_operation("INSERT chunk into pgvector", sql_insert, param_desc)

                cur.execute(
                    sql_insert,
                    (medication_name, section, chunk_index, content, embedding, meta_json)
                )
                chunk_id = str(cur.fetchone()[0])
                return chunk_id

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        medication_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform Cosine Similarity vector search against stored medication chunks.
        
        pgvector operator:
          `<=>` is Cosine Distance (range: 0 to 2, where 0 is identical and 2 is opposite).
          Cosine Similarity = 1 - (embedding <=> query_vector).
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if medication_filter:
                    sql_search = """
                    SELECT id, medication_name, section, chunk_index, content, metadata,
                           (embedding <=> %s::vector) AS distance,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM medication_chunks
                    WHERE LOWER(medication_name) = LOWER(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """
                    params = (query_embedding, query_embedding, medication_filter, query_embedding, top_k)
                else:
                    sql_search = """
                    SELECT id, medication_name, section, chunk_index, content, metadata,
                           (embedding <=> %s::vector) AS distance,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM medication_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """
                    params = (query_embedding, query_embedding, query_embedding, top_k)

                sample_q_vec = f"[{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ... dim={len(query_embedding)}]"
                log_sql_operation(
                    f"Vector Cosine Similarity Query (Top-{top_k})",
                    sql_search,
                    f"query_vector={sample_q_vec}, limit={top_k}"
                )

                cur.execute(sql_search, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "id": str(row[0]),
                        "medication_name": row[1],
                        "section": row[2],
                        "chunk_index": row[3],
                        "content": row[4],
                        "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                        "distance": float(row[6]),
                        "similarity": float(row[7])
                    })
                return results

    def get_chunk_count(self) -> int:
        """Count total chunks stored in the database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT COUNT(*) FROM medication_chunks;")
                    return cur.fetchone()[0]
                except Exception:
                    return 0


# Global database manager instance
db = DatabaseManager()
