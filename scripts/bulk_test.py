from __future__ import annotations

import sys
import os
import json
import uuid
from pathlib import Path

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

"""Bulk testing utility for the recipe chatbot agent.

Reads a CSV file containing user queries, fires them against the `/chat`
endpoint concurrently, and stores the results for later manual evaluation.
"""

import argparse
import csv
import datetime as dt
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from braintrust.otel import BraintrustSpanProcessor

from backend.utils import get_agent_response, SYSTEM_PROMPT

# -----------------------------------------------------------------------------
# Bootstrap OpenTelemetry + Braintrust
# -----------------------------------------------------------------------------
provider = TracerProvider()
provider.add_span_processor(BraintrustSpanProcessor())  # picks up BRAINTRUST_* env vars
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_CSV: Path = Path("data/sample_queries.csv")
RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MAX_WORKERS = 32

# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def process_query_sync(
    query_id: str,
    query: str,
    clarification: str
) -> Tuple[str, str, str, str]:
    """
    Runs a two-turn conversation in one trace:
      1) Ask initial query → get clarifier question
      2) Provide clarification → get final recipe

    Returns (query_id, query, clarifier_response, final_response)
    """
    # Initial user message
    history: List[Dict[str, str]] = [{"role": "user", "content": query}]
    clarifier_response = ""
    final_response = ""

    # Unique conversation identifier
    conv_id = str(uuid.uuid4())

    # Root span encompassing both turns
    with tracer.start_as_current_span("bulk_test_conversation") as span:
        span.set_attribute("bulk_test.tool", "bulk_testing_utility")
        span.set_attribute("bulk_test.query_id", query_id)
        span.set_attribute("conversation.id", conv_id)

        # ─── Clarification event ─────────────────────────────────────────────
        span.add_event(
            "clarification_requested",
            {"prompt": json.dumps(history)}
        )
        history = get_agent_response(history)
        clarifier_response = history[-1]["content"]
        span.add_event(
            "clarification_received",
            {"response": clarifier_response}
        )

        # Append user's clarity answer
        history.append({"role": "user", "content": clarification})

        # ─── Final recipe event ───────────────────────────────────────────────
        span.add_event(
            "recipe_requested",
            {"prompt": json.dumps(history)}
        )
        history = get_agent_response(history)
        final_response = history[-1]["content"]
        span.add_event(
            "recipe_received",
            {"response": final_response}
        )

    return query_id, query, clarifier_response, final_response


def run_bulk_test(csv_path: Path, num_workers: int = MAX_WORKERS) -> None:
    """Main entry point for bulk testing (synchronous version)."""
    # Read input CSV
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        input_data: List[Dict[str, str]] = [
            row for row in reader
            if row.get("id") and row.get("query") and row.get("clarification")
        ]

    if not input_data:
        raise ValueError(
            "No valid data (with 'id', 'query', and 'clarification') found in the provided CSV file."
        )

    console = Console()
    results_data: List[Tuple[str, str, str, str]] = []

    # Execute queries in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_data = {
            executor.submit(
                process_query_sync,
                item["id"],
                item["query"],
                item["clarification"],
            ): item
            for item in input_data
        }

        console.print(f"[bold blue]Submitting {len(input_data)} queries...[/bold blue]")
        for i, future in enumerate(as_completed(future_to_data)):
            item = future_to_data[future]
            try:
                qid, q, clarifier, final = future.result()
                results_data.append((qid, q, clarifier, final))

                panel_group = Group(
                    Text(f"ID: {qid}\n", style="bold magenta"),
                    Text("Query:\n", style="bold yellow"),
                    Text(q + "\n\n"),
                    Markdown("**Clarifier Question:**"),
                    Text(clarifier + "\n\n"),
                    Markdown("**Final Recipe:**"),
                    Markdown(final),
                )
                console.print(
                    Panel(
                        panel_group,
                        title=f"Result {i+1}/{len(input_data)} - ID: {qid}",
                        border_style="cyan",
                    )
                )

            except Exception as exc:
                console.print(
                    Panel(
                        f"[bold red]Error ID {item['id']}: {exc}",
                        border_style="red"
                    )
                )
                results_data.append((item["id"], item["query"], f"Error: {exc}", ""))

        console.print("[bold blue]All queries processed.[/bold blue]")

    # Write results CSV
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{timestamp}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "query", "clarifier_response", "final_response"])
        writer.writerows(results_data)

    console.print(f"[bold green]Saved {len(results_data)} results to {out_path}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk test the recipe chatbot with two-turn conversations"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to CSV with 'id','query','clarification' columns."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of worker threads (default: {MAX_WORKERS})."
    )
    args = parser.parse_args()
    run_bulk_test(args.csv, args.workers)
