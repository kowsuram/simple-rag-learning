"""Main entrypoint and interactive CLI for the Healthcare RAG Agent."""

import sys
import argparse
from rich.prompt import Prompt
from src.config import config
from src.logger import console, log_header
from src.db import db
from src.ingest import run_ingestion
from src.agent import agent


def show_status() -> None:
    """Check and display system health status."""
    import requests
    
    log_header("HEALTHCARE RAG: SYSTEM STATUS & HEALTH CHECK")
    
    # 1. Check Ollama
    ollama_ok = False
    models = []
    try:
        r = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            ollama_ok = True
            models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    console.print(f"[bold cyan]Ollama Base URL:[/bold cyan] {config.ollama_base_url}")
    if ollama_ok:
        console.print(f"  [green][OK] Connected[/green]")
        console.print(f"  Available Models: {', '.join(models)}")
        has_llm = any(config.llm_model.split(":")[0] in m for m in models)
        has_embed = any(config.embedding_model.split(":")[0] in m for m in models)
        console.print(f"  Configured LLM ({config.llm_model}): {'[green][OK] Ready[/green]' if has_llm else '[red][X] Not found (run ollama pull)'}")
        console.print(f"  Configured Embedder ({config.embedding_model}): {'[green][OK] Ready[/green]' if has_embed else '[red][X] Not found (run ollama pull)'}")
    else:
        console.print("  [red][X] Ollama is unreachable. Make sure 'ollama serve' is running.[/red]")

    console.print()

    # 2. Check Database & pgvector
    db_ok = False
    chunk_count = 0
    try:
        chunk_count = db.get_chunk_count()
        db_ok = True
    except Exception as e:
        db_error = str(e)

    console.print(f"[bold cyan]PostgreSQL + pgvector ({config.db_host}:{config.db_port}/{config.db_name}):[/bold cyan]")
    if db_ok:
        console.print(f"  [green][OK] Connected[/green]")
        console.print(f"  Chunks stored in 'medication_chunks': [bold yellow]{chunk_count}[/bold yellow]")
    else:
        console.print(f"  [red][X] Connection failed: {db_error}[/red]")

    console.print()


def interactive_chat() -> None:
    """Launch an interactive terminal chat session with the Healthcare Agent."""
    log_header(
        "HEALTHCARE MEDICATION ADVISOR: INTERACTIVE CHAT",
        "Type your medication inquiry below. Type 'exit' or 'quit' to end session."
    )
    
    # Quick chunk check
    try:
        count = db.get_chunk_count()
        if count == 0:
            console.print("[yellow]Warning: Database has 0 chunks! Run 'python main.py ingest' first.[/yellow]\n")
    except Exception as e:
        console.print(f"[red]Database connection error: {e}[/red]\n")
        return

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]Patient Enquiry[/bold green]")
            if not user_input or user_input.strip().lower() in ["exit", "quit", "q"]:
                console.print("[cyan]Ending consultation session. Stay safe![/cyan]")
                break
            
            agent.ask(user_input.strip(), verbose=True)
        except KeyboardInterrupt:
            console.print("\n[cyan]Session terminated.[/cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error processing enquiry: {e}[/red]")


def inspect_embeddings(limit: int = 5) -> None:
    """Query and display raw vector embeddings stored in PostgreSQL/pgvector."""
    from rich.table import Table
    log_header("PGVECTOR STORAGE INSPECTION: RAW EMBEDDINGS")

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT medication_name, section, 
                       vector_dims(embedding) as dims,
                       embedding::text as raw_vec,
                       content
                FROM medication_chunks 
                ORDER BY medication_name, chunk_index
                LIMIT %s;
            """, (limit,))
            rows = cur.fetchall()

    if not rows:
        console.print("[yellow]No embeddings found in 'medication_chunks'. Run 'python main.py ingest' first.[/yellow]")
        return

    table = Table(title=f"Stored pgvector Embeddings (First {len(rows)} Records)", show_lines=True)
    table.add_column("Medication", style="bold cyan", width=14)
    table.add_column("Section", style="magenta", width=22)
    table.add_column("Dims", justify="center", style="bold yellow", width=6)
    table.add_column("Vector Preview (768-D Float Coordinates)", style="green")
    table.add_column("Chunk Snippet", style="white", width=35)

    for row in rows:
        med, sec, dims, raw_vec_str, content = row[0], row[1], row[2], row[3], row[4]
        vec_floats = [float(x) for x in raw_vec_str.strip("[]").split(",")]
        first_6 = [f"{x:+.4f}" for x in vec_floats[:6]]
        last_3 = [f"{x:+.4f}" for x in vec_floats[-3:]]
        preview = f"[{', '.join(first_6)}, ... , {', '.join(last_3)}]"
        snippet = content.replace("\n", " ")[:60] + "..."
        table.add_row(med, sec, str(dims), preview, snippet)

    console.print(table)
    console.print(f"[dim]Total columns in table: id, medication_name, section, chunk_index, content, embedding vector({dims}), metadata[/dim]\n")


def main():
    parser = argparse.ArgumentParser(description="Healthcare RAG Agent with Ollama & pgvector")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ingest
    parser_ingest = subparsers.add_parser("ingest", help="Ingest medication monographs into pgvector")
    parser_ingest.add_argument("--no-reset", action="store_true", help="Do not drop existing table before ingestion")

    # Command: ask
    parser_ask = subparsers.add_parser("ask", help="Ask a single question and inspect the pipeline logs")
    parser_ask.add_argument("question", type=str, help="Patient medication inquiry")

    # Command: chat
    subparsers.add_parser("chat", help="Start an interactive chat loop")

    # Command: status
    subparsers.add_parser("status", help="Check status of Ollama and pgvector database")

    # Command: inspect
    parser_inspect = subparsers.add_parser("inspect", help="Inspect raw vector embeddings stored in pgvector")
    parser_inspect.add_argument("--limit", type=int, default=5, help="Number of records to display (default: 5)")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(reset_db=not args.no_reset)
    elif args.command == "ask":
        agent.ask(args.question, verbose=True)
    elif args.command == "chat":
        interactive_chat()
    elif args.command == "status":
        show_status()
    elif args.command == "inspect":
        inspect_embeddings(limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
