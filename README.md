# Healthcare Medication RAG Agent (Local Ollama + PostgreSQL pgvector)

A simple, transparent, and educational **Retrieval-Augmented Generation (RAG)** agent designed for healthcare medication inquiries. The agent answers patient questions grounded strictly in an authoritative repository of medication monographs (the source of truth), with verbose, visual logging of every stage in the pipeline.

---

## Architecture Overview

```
                                  INGESTION FLOW
+-------------------------+     +-------------------+     +---------------------+     +--------------------+
|  Medication Monographs  | --> | Section-Aware     | --> | Ollama Embedding    | --> | PostgreSQL +       |
|  (data/medications/*.json)     | Chunking + Overlap|     | (nomic-embed-text)  |     | pgvector Storage   |
+-------------------------+     +-------------------+     +---------------------+     +--------------------+
                                         |                            |                           |
                                  [Log Size, Span,            [Log 768-D Dims,            [Log SQL Table,
                                   Overlap preview]            Sample Floats, ms]          HNSW Index, UUID]

                                   RETRIEVAL FLOW
+-------------------------+     +-------------------+     +---------------------+     +--------------------+
|  Patient Enquiry        | --> | Ollama Embedding  | --> | pgvector Cosine     | --> | Guardrailed Prompt |
|  ("Can I take Advil     |     | Query Vector      |     | Similarity (<=>)    |     | + Ollama LLM       |
|  with stomach ulcer?")  |     | (768-D vector)    |     | Top-K Candidate Rank|     | (llama3.2:latest)  |
+-------------------------+     +-------------------+     +---------------------+     +--------------------+
                                         |                            |                           |
                                  [Log 768-D Query            [Log Distance,              [Log Injected Context,
                                   Vector Floats]              Similarity Table]           Clinical Answer]
```

---

## Core Concepts Explained

### 1. What is Chunking?
Medication leaflets contain dense clinical details (dosages, drug interactions, contraindications). If you ingest entire documents at once, embeddings get diluted and context windows fill up.
- **Section-Aware Chunking**: We divide monographs along medical section boundaries (`Indications`, `Dosage`, `Contraindications`, `Warnings`).
- **Sliding Window with Overlap**: If a section exceeds the chunk limit (e.g., 450 characters), a sliding window with an overlap (e.g., 80 characters) ensures critical instructions (like "DO NOT take if you have peptic ulcers") are not split across chunks.
- **Contextual Anchoring**: Every chunk prepends `[Medication: Ibuprofen | Section: Contraindications]` to guarantee the embedding vector retains both the drug name and clinical domain.

### 2. What are Embeddings?
An embedding model converts text into an array of floating-point numbers (a vector in high-dimensional space). 
- Using `nomic-embed-text`, each chunk becomes a **768-dimensional coordinate**:
  `[-0.0142, +0.0891, -0.0524, ..., +0.0118]`
- Semantically similar texts (e.g., "stomach ulcers" and "peptic ulcer disease") are mapped close together in this 768-dimensional space.

### 3. How pgvector Stores and Queries Embeddings
PostgreSQL with the `pgvector` extension provides native vector types and indexing:
- **Table Definition**:
  ```sql
  CREATE TABLE medication_chunks (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      medication_name VARCHAR(100),
      section VARCHAR(100),
      content TEXT,
      embedding vector(768),
      metadata JSONB
  );
  ```
- **HNSW Index**: Hierarchical Navigable Small World index accelerates nearest-neighbor vector search:
  ```sql
  CREATE INDEX ON medication_chunks USING hnsw (embedding vector_cosine_ops);
  ```
- **Cosine Similarity Query**:
  ```sql
  SELECT medication_name, section, content,
         1 - (embedding <=> %s::vector) AS similarity
  FROM medication_chunks
  ORDER BY embedding <=> %s::vector
  LIMIT 3;
  ```
  *(The `<=>` operator computes Cosine Distance; subtracting from 1 yields Cosine Similarity).*

### 4. Clinical Prompt Guardrails
To prevent medical hallucinations, the prompt explicitly instructs the LLM:
1. **Strict Grounding**: Base answers strictly on the retrieved context.
2. **Hallucination Refusal**: If a medication is not in the source of truth, explicitly state that it is not documented in the approved clinical database.
3. **Safety First**: Emphasize contraindications, maximum dosages, and high-risk interactions.
4. **Mandatory Disclaimer**: Include a clinical notice directing patients to consult licensed healthcare professionals.

---

---

## Quick Start Guide

### 🚀 Option A: One-Click Startup (Recommended for Windows)

Simply double-click or run:
```bat
start_all.bat
```

**What `start_all.bat` does automatically:**
1. ✅ Checks and starts the PostgreSQL + `pgvector` Docker container on port `5433`.
2. ✅ Verifies Ollama is running and automatically pulls `nomic-embed-text` and `llama3.2` if needed.
3. ✅ Checks if the database has chunks and runs knowledge base ingestion automatically if empty.
4. ✅ Launches the interactive Web Dashboard and opens `http://localhost:8505` in your browser.

---

### 🛠️ Option B: Step-by-Step Manual Startup

If you prefer to start each component manually:

#### 1. Start the PostgreSQL + pgvector Database
```powershell
docker compose up -d
```
*(Runs `pgvector/pgvector:pg16` on port `5433`)*

#### 2. Start Ollama and Pull Required Models
In a separate terminal window:
```powershell
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

#### 3. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

#### 4. Verify System Health Status
```powershell
python main.py status
```

#### 5. Ingest Medication Monographs into pgvector
Run ingestion with verbose educational logs:
```powershell
python main.py ingest
```

#### 6. Launch the Streamlit Web Dashboard & Database Explorer
```powershell
python -m streamlit run app.py --server.port 8505
```
Open **`http://localhost:8505`** in your browser to explore the database table, chat live, and inspect vector similarities!

---

## Command Line Interface (CLI) Guide

You can also run every feature directly from your terminal:

| Command | Description |
| :--- | :--- |
| `python main.py status` | Checks health of Ollama, models, and PostgreSQL pgvector connection. |
| `python main.py ingest` | Runs end-to-end ingestion (chunking, embedding, pgvector storage) with logs. |
| `python main.py inspect --limit 5` | Displays the raw 768-D vector coordinates and dimensions stored in PostgreSQL. |
| `python main.py ask "<question>"` | Runs a single query through the RAG pipeline with step-by-step trace. |
| `python main.py chat` | Starts an interactive real-time consultation chat loop. |
| `python -m unittest tests/test_pipeline.py` | Runs the automated 5-point clinical and vector test suite. |

### Example Clinical Inquiries to Try:
- **Contraindications**: `python main.py ask "Can I take Ibuprofen if I have active stomach ulcers?"`
- **Dosage**: `python main.py ask "What is the maximum daily dose of Paracetamol for adults?"`
- **Interactions**: `python main.py ask "Why shouldn't I drink grapefruit juice with Atorvastatin?"`
- **Allergies**: `python main.py ask "Can a patient allergic to Penicillin take Amoxicillin?"`
- **Hallucination Prevention**: `python main.py ask "Can a patient take WonderDrugXYZ?"` *(Refuses advice for unlisted medicines)*

---

## Database GUI & Visual Inspection Tools

To view and inspect the `public.medication_chunks` table visually:

1. **Streamlit Web GUI (Built-in)**: Go to the **🗄️ Database Explorer** tab at [http://localhost:8505](http://localhost:8505).
2. **DBeaver Community (Free Desktop App)**:
   - Host: `localhost` | Port: `5433` | Database: `healthcaredb` | User: `postgres` | Password: `postgrespassword`
3. **Direct SQL via Docker psql**:
   ```powershell
   docker exec -it healthcare-pgvector psql -U postgres -d healthcaredb -c "SELECT medication_name, section, vector_dims(embedding), substring(embedding::text from 1 for 40) || '...' as vector FROM medication_chunks LIMIT 5;"
   ```

---

## TODO

- [ ] Try unstructured ingestion from PDFs and approved online sources; extract the text, retain source and page metadata, and validate it before chunking and embedding.

---

## Deploying with Neon on Streamlit Community Cloud

1. In Neon, open **Connect**, select your database and branch, and copy the pooled connection string.
2. In Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

   ```toml
   DATABASE_URL = "postgresql://<user>:<password>@<host>/<database>?sslmode=require"
   ```

   Keep the connection string in Streamlit Secrets; do not commit it to the repository.
3. The app uses `DATABASE_URL` when present and keeps the existing `DB_*` settings as the local-development fallback.
4. In the Neon SQL Editor, enable pgvector once for the database:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

For local development, put `DATABASE_URL` in the ignored `.env` file. The existing Ollama service is separate from the database and must also be reachable by the deployed app for embeddings and LLM responses.

### Cloudflare Workers AI for cloud-hosted models

The Streamlit app supports Cloudflare Workers AI for both chat and embeddings. Create a Workers AI API token with Workers AI Read and Edit permissions, then add these values under the app's **Settings → Secrets**:

```toml
AI_PROVIDER = "cloudflare"
CLOUDFLARE_ACCOUNT_ID = "your-account-id"
CLOUDFLARE_API_TOKEN = "your-api-token"
```

The default models are `@cf/meta/llama-3.2-3b-instruct` for chat and `@cf/baai/bge-base-en-v1.5` for embeddings. The embedding model returns 768 dimensions, matching the database schema. After switching embedding models, click **Re-Ingest All Medications** once to rebuild the stored vectors; this operation resets the existing `medication_chunks` table.

### Optional Langfuse tracing

To trace RAG queries, retrieval, embeddings, and model-generation timing, add the following to Streamlit Community Cloud **Settings → Secrets**:

```toml
LANGFUSE_PUBLIC_KEY = "your-public-key"
LANGFUSE_SECRET_KEY = "your-secret-key"
LANGFUSE_BASE_URL = "https://your-Langfuse-host"
```

Set `LANGFUSE_BASE_URL` to the URL shown by Langfuse for your project. The app creates nested observations but does not capture inputs or outputs, avoiding sending query text, retrieved content, prescriptions, or generated answers to tracing. Tracing is disabled when the public or secret key is missing.

---

## Directory Structure
```
healthcare-rag/
├── data/
│   └── medications/              # Authoritative clinical monographs
│       ├── paracetamol.json
│       ├── ibuprofen.json
│       ├── amoxicillin.json
│       ├── metformin.json
│       ├── lisinopril.json
│       └── atorvastatin.json
├── src/
│   ├── config.py                 # Pydantic configuration & .env loader
│   ├── logger.py                 # Educational Rich-based terminal visualizer
│   ├── db.py                     # PostgreSQL + pgvector client & schema manager
│   ├── chunker.py                # Section-aware document chunker with overlap
│   ├── embedder.py               # Ollama vector embedding client
│   ├── ingest.py                 # End-to-end ingestion pipeline
│   ├── retriever.py              # Semantic similarity retriever
│   └── agent.py                  # Clinical RAG Agent with guardrails
├── tests/
│   └── test_pipeline.py          # Automated verification suite
├── .env                          # Local environment variables
├── .env.example                  # Template environment variables
├── docker-compose.yml            # PostgreSQL + pgvector container definition
├── requirements.txt              # Python package requirements
├── main.py                       # CLI application entrypoint
└── README.md                     # Documentation
```