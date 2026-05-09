"""eval/sweep_deterministic.py

Quick sweep: run the deterministic authority_partition config against every
seed in eval/records/seeds/ and produce a summary table. Free (no LLM calls);
takes well under 1 second for all 9 seeds.

This is the predicate-layer correctness test. Every REJECT seed should be
localized to the predicate corresponding to its violation_family. Every
aligned seed should PASS all predicates.

Run from project root:
    uv run python -m eval.sweep_deterministic
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from eval.configs.authority_partition import AuthorityPartitionConfig  # noqa: E402

# Map violation_family -> expected primary_failure_node for the deterministic layer.
EXPECTED_NODE_BY_FAMILY = {
    "captured":     "authority_partition.captured_approver",
    "colluded":     "authority_partition.collusion_proposer_approver",
    "undisclosed":  "authority_partition.undisclosed_conflict_of_interest",
    "forged":       "authority_partition.signature_forgery",
    "coerced":      "authority_partition.signature_coercion",
    "sod":          "authority_partition.separation_of_duties_violation",
    "drift":        "authority_partition.designation_drift",
    "aligned":      None,
}


def main() -> None:
    seeds_dir = PROJECT_ROOT / "eval" / "records" / "seeds"
    seed_files = sorted(p for p in seeds_dir.glob("*.json") if not p.name.startswith("_"))
    config = AuthorityPartitionConfig()

    console = Console()
    table = Table(title=f"Deterministic authority_partition sweep ({len(seed_files)} seeds)",
                  show_header=True, header_style="bold cyan")
    table.add_column("seed_id", style="dim")
    table.add_column("family")
    table.add_column("gold")
    table.add_column("pred")
    table.add_column("predicted_node")
    table.add_column("expected_node")
    table.add_column("outcome", justify="center")
    table.add_column("ms", justify="right")

    correct = 0
    total = 0
    rows = []

    for path in seed_files:
        raw = json.loads(path.read_text())
        record = raw.get("record", raw)
        meta = raw.get("metadata", {})
        family = meta.get("violation_family", "?")
        seed_id = meta.get("seed_id") or record.get("sample_id", path.stem)

        verdict = config.evaluate(record)

        gold_label = record.get("label", "?")
        pred_label = verdict.status
        pred_node = verdict.primary_failure_node
        expected_node = EXPECTED_NODE_BY_FAMILY.get(family)

        binary_match = (pred_label == gold_label)
        node_match = (pred_node == expected_node)
        ok = binary_match and node_match

        outcome = "[green]✓[/green]" if ok else "[red]✗[/red]"
        rows.append({
            "seed_id": seed_id,
            "family": family,
            "gold": gold_label,
            "pred": pred_label,
            "predicted_node": pred_node or "(none)",
            "expected_node": expected_node or "(none)",
            "outcome": ok,
            "ms": verdict.latency_ms,
            "reasoning": verdict.reasoning,
        })

        table.add_row(
            seed_id, family, gold_label, pred_label,
            pred_node or "[dim](none)[/dim]",
            expected_node or "[dim](none)[/dim]",
            outcome, str(verdict.latency_ms),
        )
        if ok:
            correct += 1
        total += 1

    console.print(table)

    pct = (correct / total * 100) if total else 0.0
    summary_color = "bold green" if correct == total else "bold yellow"
    console.print(f"\n[{summary_color}]correct: {correct}/{total} ({pct:.1f}%)[/{summary_color}]")

    if correct < total:
        console.print("\n[bold]Failed seeds:[/bold]")
        for row in rows:
            if not row["outcome"]:
                console.print(
                    f"  [yellow]{row['seed_id']}[/yellow] (family={row['family']})\n"
                    f"    expected: {row['expected_node']}\n"
                    f"    got:      {row['predicted_node']}\n"
                    f"    reason:   {row['reasoning']}"
                )

    # Persist
    out_path = PROJECT_ROOT / "eval" / "results" / "sweep__authority_partition__predicate-graph-v0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": "authority_partition",
        "model": "predicate-graph-v0",
        "n_seeds": total,
        "correct": correct,
        "rows": rows,
    }, indent=2))
    console.print(f"\n[dim]Persisted to:[/dim] {out_path}")


if __name__ == "__main__":
    main()
