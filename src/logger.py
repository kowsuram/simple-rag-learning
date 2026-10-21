import sys
import time
from typing import List, Dict, Any

# Ensure Windows terminal handles UTF-8 gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax

console = Console(legacy_windows=False)


def log_header(title: str, subtitle: str = "") -> None:
    """Print an impactful section banner."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[italic dim]{subtitle}[/italic dim]"
    console.print()
    console.print(Panel(content, border_style="cyan", expand=False))


def log_step(step_num: int, title: str, explanation: str) -> None:
    """Print a step marker with educational explanation."""
    console.print()
    step_text = Text()
    step_text.append(f" STEP {step_num} ", style="bold black on bright_yellow")
    step_text.append(f"  {title}\n", style="bold white")
    step_text.append(f"{explanation}", style="dim")
    console.print(Panel(step_text, border_style="yellow", padding=(0, 1)))


def log_document_loaded(med_name: str, brand_names: List[str], class_name: str, sections_count: int, total_chars: int) -> None:
    """Log document ingestion status."""
    table = Table(title=f"Source Document: {med_name}", title_style="bold green", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Medication Name", med_name)
    table.add_row("Brand Names", ", ".join(brand_names))
    table.add_row("Therapeutic Class", class_name)
    table.add_row("Clinical Sections", str(sections_count))
    table.add_row("Total Characters", f"{total_chars:,} chars")
    
    console.print(table)


def log_chunk_details(
    medication: str,
    section: str,
    chunk_index: int,
    total_chunks: int,
    start_char: int,
    end_char: int,
    text: str,
    overlap_chars: int = 0
) -> None:
    """Log the chunking process for a specific text segment."""
    preview = text[:180] + ("..." if len(text) > 180 else "")
    chunk_info = (
        f"[bold cyan]Medication:[/bold cyan] {medication}  |  "
        f"[bold cyan]Section:[/bold cyan] {section}\n"
        f"[bold cyan]Chunk ID:[/bold cyan] #{chunk_index + 1} of {total_chunks}  |  "
        f"[bold cyan]Char Span:[/bold cyan] [{start_char}:{end_char}] ({len(text)} chars)  |  "
        f"[bold cyan]Overlap:[/bold cyan] {overlap_chars} chars\n\n"
        f"[bold yellow]Chunk Content Preview:[/bold yellow]\n[italic]{preview}[/italic]"
    )
    console.print(Panel(chunk_info, title=f"Chunk #{chunk_index + 1} Generated", border_style="blue", padding=(0, 1)))


def log_embedding_details(
    text_snippet: str,
    model: str,
    dimension: int,
    sample_vector: List[float],
    latency_ms: float
) -> None:
    """Log vector embedding generation with preview of float numbers."""
    first_few = [f"{v:+.4f}" for v in sample_vector[:6]]
    last_few = [f"{v:+.4f}" for v in sample_vector[-3:]]
    vector_preview = f"[{', '.join(first_few)}, ... , {', '.join(last_few)}]"
    
    info = (
        f"[bold green]Embedding Model:[/bold green] {model}\n"
        f"[bold green]Vector Dimension:[/bold green] {dimension} float numbers\n"
        f"[bold green]Generation Latency:[/bold green] {latency_ms:.1f} ms\n"
        f"[bold green]Vector Snapshot (768-D coordinates):[/bold green]\n"
        f"[magenta]{vector_preview}[/magenta]\n\n"
        f"[dim]Input snippet embedded: \"{text_snippet[:100]}...\"[/dim]"
    )
    console.print(Panel(info, title="Vector Embedding Generated", border_style="green", padding=(0, 1)))


def log_sql_operation(operation_name: str, sql_query: str, params_summary: str = "") -> None:
    """Log raw SQL execution for pgvector operations."""
    syntax = Syntax(sql_query.strip(), "sql", theme="monokai", line_numbers=False)
    panel_content = Text()
    panel_content.append(f"Operation: {operation_name}\n", style="bold cyan")
    if params_summary:
        panel_content.append(f"Parameters: {params_summary}\n\n", style="dim")
    
    console.print(Panel(panel_content, title=f"pgvector SQL Execution", border_style="magenta", padding=(0, 1)))
    console.print(syntax)


def log_retrieval_ranking(query: str, retrieved_chunks: List[Dict[str, Any]]) -> None:
    """Display an educational comparison table of retrieved chunks and their similarity scores."""
    console.print()
    console.print(f"[bold cyan]Query:[/bold cyan] [italic]\"{query}\"[/italic]")
    
    table = Table(
        title="pgvector Cosine Similarity Search Results (Top Chunks)",
        title_style="bold green",
        header_style="bold magenta",
        show_lines=True
    )
    table.add_column("Rank", justify="center", style="bold yellow", width=6)
    table.add_column("Medication", style="cyan", width=14)
    table.add_column("Section", style="blue", width=18)
    table.add_column("Similarity", justify="center", style="bold green", width=12)
    table.add_column("Cosine Distance (<=>)", justify="center", style="dim", width=14)
    table.add_column("Matched Content Snippet", style="white")

    for i, chunk in enumerate(retrieved_chunks, start=1):
        similarity = chunk.get("similarity", 0.0)
        distance = chunk.get("distance", 1.0 - similarity)
        content_preview = chunk.get("content", "").strip()[:140] + "..."
        sim_percentage = f"{similarity * 100:.2f}%"
        
        table.add_row(
            f"#{i}",
            chunk.get("medication_name", "Unknown"),
            chunk.get("section", "General"),
            sim_percentage,
            f"{distance:.4f}",
            content_preview
        )
    
    console.print(table)


def log_augmented_prompt(retrieved_context: str, system_guardrails: str, user_question: str) -> None:
    """Log the prompt constructed for the LLM."""
    content = (
        f"[bold yellow]=== CLINICAL GUARDRAILS (System Prompt) ===[/bold yellow]\n"
        f"{system_guardrails}\n\n"
        f"[bold green]=== RETRIEVED GROUND TRUTH (pgvector Context) ===[/bold green]\n"
        f"{retrieved_context.strip()}\n\n"
        f"[bold cyan]=== PATIENT QUESTION ===[/bold cyan]\n"
        f"{user_question}"
    )
    console.print(Panel(content, title="Augmented Prompt Sent to Ollama LLM", border_style="yellow", padding=(0, 1)))


def log_llm_response(response_text: str, latency_ms: float) -> None:
    """Log the final clinical agent response."""
    footer = f"\n\n[dim]Generation latency: {latency_ms:.1f} ms | Grounded in pgvector clinical database[/dim]"
    console.print(Panel(response_text + footer, title="Clinical Agent Response (Ollama llama3.2)", border_style="bright_green"))
