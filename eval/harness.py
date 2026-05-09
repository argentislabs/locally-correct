"""eval/harness.py

Minimal driver. Loads a record from disk, looks up a verifier config by name,
runs evaluation, prints a structured comparison between predicted and gold
verdicts.

Usage:
    uv run python -m eval.harness eval/records/handcrafted/type_d_smoke.json \\
        --config gpt_no_approver

Multiple records and multiple configs come later. This is the smoke driver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env from project root before anything else touches os.environ
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from eval.configs.base import RecordSummary, Verdict  # noqa: E402

console = Console()


def load_config(name: str, model: str | None = None):
    """Lazy-import config classes so we don't pay OpenAI-client cost for unused configs."""
    if name == "gpt_no_approver":
        from eval.configs.gpt_no_approver import GPTNoApproverConfig
        return GPTNoApproverConfig(model=model)
    if name == "gpt_with_approver":
        from eval.configs.gpt_with_approver import GPTWithApproverConfig
        return GPTWithApproverConfig(model=model)
    if name == "authority_partition":
        from eval.configs.authority_partition import AuthorityPartitionConfig
        if model is not None:
            console.print(
                f"[yellow]Note:[/yellow] authority_partition is deterministic; "
                f"--model {model!r} ignored."
            )
        return AuthorityPartitionConfig()
    if name == "composed":
        from eval.configs.composed import ComposedConfig
        return ComposedConfig(alignment_model=model)
    raise ValueError(
        f"Unknown config: {name}. Known: "
        "gpt_no_approver, gpt_with_approver, authority_partition, composed"
    )


def load_record(path: Path) -> tuple[dict, dict]:
    """Returns (record, metadata). Handles both seed-envelope and bare-record formats."""
    raw = json.loads(path.read_text())
    if "record" in raw and isinstance(raw["record"], dict):
        return raw["record"], raw.get("metadata", {})
    return raw, {}


TYPE_D_FAMILIES = {
    "captured", "colluded", "undisclosed", "forged",
    "coerced", "sod", "drift",
}


def render_verdict(verdict: Verdict, record: dict, meta: dict, config_name: str) -> None:
    summary = RecordSummary.from_record(record, family=meta.get("violation_family"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("field", style="dim")
    table.add_column("value")

    gold_label = record.get("label", "?")
    gold_node = (
        record.get("reason", {}).get("failed_validation_node", {}).get("node_id")
        if isinstance(record.get("reason"), dict)
        else None
    )
    pred_status = verdict.status
    pred_node = verdict.primary_failure_node

    table.add_row("sample_id", summary.sample_id)
    table.add_row("violation_family", summary.violation_family or "(none)")
    table.add_row("intent", summary.intent_preview)
    table.add_row("", "")
    table.add_row("[bold]gold label[/bold]", gold_label)
    table.add_row("[bold]gold failure node[/bold]", gold_node or "(n/a)")
    table.add_row("", "")
    table.add_row("[bold]predicted status[/bold]", pred_status)
    table.add_row("[bold]predicted failure node[/bold]", pred_node or "(none)")
    table.add_row("confidence", f"{verdict.confidence:.2f}" if verdict.confidence is not None else "-")
    table.add_row("latency", f"{verdict.latency_ms} ms")

    if verdict.error:
        table.add_row("[red]error[/red]", verdict.error)

    console.print(table)
    if verdict.reasoning:
        console.print(Panel(verdict.reasoning, title="reasoning", border_style="dim"))

    # Architectural commentary
    if pred_status == "ERROR":
        console.print("[red]ERROR — config failed to produce a verdict.[/red]")
        return

    is_type_d = (
        gold_label == "REJECT"
        and meta.get("violation_family") in TYPE_D_FAMILIES
    )
    has_approver_access = config_name in {"gpt_with_approver", "authority_partition", "composed"}
    is_llm_with_approver = config_name == "gpt_with_approver"
    is_graph = config_name == "authority_partition"
    is_composed = config_name == "composed"

    if is_type_d and not has_approver_access:
        # Alignment-only configs SHOULD approve Type D records — that is the
        # categorical-state distinction in action.
        if pred_status == "PASS":
            console.print(
                "[green]PREDICTION CONFIRMED[/green] — alignment-only configuration "
                "approved a record whose gold label is REJECT, where the failure "
                "lives in institutional designation state. This is the categorical-"
                "state distinction in action."
            )
        else:
            console.print(
                "[yellow]UNEXPECTED REJECT[/yellow] — alignment-only config rejected "
                "a Type D record without approver context. Inspect reasoning: it may "
                "be latching on to a stylistic cue or a weak transaction-local "
                "signal. Worth a stylistic-distance check on the seed."
            )

    elif is_type_d and is_llm_with_approver:
        # The existential test: LLM-as-judge with full approver context.
        if pred_status == "PASS":
            console.print(
                "[bold green]ARCHITECTURE STRONGLY JUSTIFIED[/bold green] — config had "
                "approver context and STILL approved the Type D record. The model "
                "either ignored the approver block or failed to reason structurally "
                "from the designation chain. This is exactly what a deterministic "
                "predicate-based graph addresses."
            )
        elif pred_status == "REJECT" and pred_node == gold_node:
            console.print(
                "[yellow]BINARY + LOCALIZATION MATCH ON THIS RECORD[/yellow] — "
                "LLM-with-approver returned correct verdict and correct family. "
                "Honest reading: the LLM can read explicit structural cues when "
                "they are present. This does NOT yet challenge the architectural "
                "claim, which rests on (a) variance stability across reruns at "
                "T=0.1, (b) records whose failure requires structural reasoning "
                "rather than literal field-reading (e.g., no null roots, no stale "
                "disclosures), and (c) discrimination across all 7 authority "
                "families. Single-record success on an explicit failure is "
                "consistent with literal field-reading; the existential test "
                "lives on the harder seed set."
            )
        elif pred_status == "REJECT" and pred_node != gold_node:
            console.print(
                "[yellow]REJECT WITH FAMILY MIS-LOCALIZATION[/yellow] — config "
                f"rejected (good) but labeled it [bold]{pred_node}[/bold] when "
                f"gold is [bold]{gold_node}[/bold]. Binary verdict matches; family "
                "localization fails. This is the predicted pattern — LLM-as-judge "
                "can flag SOMETHING is wrong but cannot reliably name WHICH "
                "structural failure mode applies. The graph architecture's claim "
                "rests on stable family localization, not binary verdict alone."
            )

    elif is_type_d and is_graph:
        # The deterministic graph getting the right answer IS the architecture.
        if pred_status == "REJECT" and pred_node == gold_node:
            console.print(
                "[bold green]GRAPH BEHAVED AS DESIGNED[/bold green] — deterministic "
                "predicate-based authority partition correctly localized the failure "
                f"to [bold]{pred_node}[/bold] in 0ms with no LLM cost. This is what "
                "the architecture is for; the load-bearing question is whether "
                "LLM-as-judge can replicate this localization stably (test via "
                "gpt_with_approver across 5 runs)."
            )
        elif pred_status == "REJECT" and pred_node != gold_node:
            console.print(
                "[red]GRAPH MIS-LOCALIZED[/red] — predicates fired but the wrong "
                f"family. Investigate node ordering: gold={gold_node!r}, "
                f"predicted={pred_node!r}. May indicate a record with multiple "
                "concurrent failures and the ordering chose the wrong primary."
            )
        elif pred_status == "PASS":
            console.print(
                "[red]GRAPH MISSED FAILURE[/red] — predicates all passed but gold "
                "label is REJECT. Investigate: either the predicate set is "
                "incomplete for this family, or the seed's approver block is "
                "missing the structural marker the predicate depends on."
            )

    elif is_type_d and is_composed:
        # Composition: authority subgraph should fire; alignment subgraph
        # should pass on this clean transaction.
        if pred_status == "REJECT" and pred_node == gold_node:
            console.print(
                "[green]COMPOSITION BEHAVED AS DESIGNED[/green] — alignment "
                "subgraph PASSED, authority subgraph REJECTED, composed verdict "
                f"took the authority result and localized to [bold]{pred_node}[/bold]. "
                "Authority half is model-tier independent (deterministic dispatch); "
                "alignment half varies with model. This demonstrates coverage "
                "across both alignment and authorization failure surfaces FOR THIS "
                "RECORD CLASS. Whether the composition is load-bearing at scale "
                "depends on harder families and variance — that's the next experiment."
            )
        else:
            console.print(
                f"[yellow]COMPOSITION OUTCOME[/yellow] — pred={pred_status} "
                f"node={pred_node}, gold=REJECT/{gold_node}. Inspect "
                "alignment_subgraph and authority_subgraph in the persisted "
                "raw_response to see which subgraph drove the verdict."
            )

    else:
        match = "MATCH" if pred_status == gold_label else "DIVERGE"
        color = "green" if match == "MATCH" else "red"
        console.print(f"[{color}]{match}[/{color}] — predicted {pred_status}, gold {gold_label}.")


def aggregate_stats(verdicts: list[Verdict]) -> dict:
    """Compute aggregate statistics across N runs of the same config on the same record.

    The protocol critique mandates: majority verdict, primary-node distribution,
    variance/divergence flag, latency median + min/max. This is the function that
    produces the numbers we'll quote in Section 6.
    """
    from collections import Counter

    statuses = [v.status for v in verdicts]
    nodes = [v.primary_failure_node for v in verdicts]
    latencies = sorted(v.latency_ms for v in verdicts)

    status_counts = Counter(statuses)
    node_counts = Counter(nodes)
    n = len(verdicts)
    median = latencies[n // 2] if n % 2 == 1 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2

    return {
        "n_runs": n,
        "majority_status": status_counts.most_common(1)[0][0],
        "majority_status_share": status_counts.most_common(1)[0][1] / n,
        "status_distribution": dict(status_counts),
        "majority_primary_node": node_counts.most_common(1)[0][0],
        "majority_primary_node_share": node_counts.most_common(1)[0][1] / n,
        "primary_node_distribution": dict(node_counts),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "latency_ms_median": median,
        "verdict_diverged": len(set(statuses)) > 1,
        "primary_node_diverged": len(set(nodes)) > 1,
    }


def render_aggregate(verdicts: list[Verdict], record: dict, meta: dict, config_name: str) -> None:
    """Multi-run aggregate rendering. Mirrors the architectural commentary in
    render_verdict but uses majority verdict + variance to decide what to say.
    """
    summary = RecordSummary.from_record(record, family=meta.get("violation_family"))
    stats = aggregate_stats(verdicts)

    gold_label = record.get("label", "?")
    gold_node = (
        record.get("reason", {}).get("failed_validation_node", {}).get("node_id")
        if isinstance(record.get("reason"), dict) else None
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_row("sample_id", summary.sample_id)
    table.add_row("violation_family", summary.violation_family or "(none)")
    table.add_row("intent", summary.intent_preview)
    table.add_row("", "")
    table.add_row("[bold]gold label[/bold]", gold_label)
    table.add_row("[bold]gold failure node[/bold]", gold_node or "(n/a)")
    table.add_row("", "")
    table.add_row("[bold]majority status[/bold]",
                  f"{stats['majority_status']}  ({stats['majority_status_share']:.0%} of {stats['n_runs']} runs)")
    table.add_row("[bold]majority primary node[/bold]",
                  f"{stats['majority_primary_node'] or '(none)'}  "
                  f"({stats['majority_primary_node_share']:.0%} of {stats['n_runs']} runs)")
    table.add_row("status distribution", str(stats["status_distribution"]))
    table.add_row("node distribution", str(stats["primary_node_distribution"]))
    table.add_row("verdict diverged", "[red]yes[/red]" if stats["verdict_diverged"] else "[green]no[/green]")
    table.add_row("primary node diverged",
                  "[red]yes[/red]" if stats["primary_node_diverged"] else "[green]no[/green]")
    table.add_row("latency (median)", f"{stats['latency_ms_median']:.0f} ms")
    table.add_row("latency (min/max)", f"{stats['latency_ms_min']} / {stats['latency_ms_max']} ms")
    console.print(table)

    # Architectural commentary using the MAJORITY verdict
    pred_status = stats["majority_status"]
    pred_node = stats["majority_primary_node"]
    is_type_d = (
        gold_label == "REJECT"
        and meta.get("violation_family") in TYPE_D_FAMILIES
    )
    is_llm_with_approver = config_name == "gpt_with_approver"
    is_graph = config_name == "authority_partition"

    if is_type_d and is_llm_with_approver:
        if stats["primary_node_diverged"] or stats["verdict_diverged"]:
            console.print(
                f"[bold red]LOCALIZATION INSTABILITY[/bold red] — across {stats['n_runs']} reruns "
                f"at T=0.1, the LLM-with-approver disagreed with itself "
                f"(verdict diverged: {stats['verdict_diverged']}, primary node diverged: "
                f"{stats['primary_node_diverged']}). Distribution: {stats['primary_node_distribution']}. "
                "This is the predicted pattern: LLM-as-judge produces opportunistic correctness "
                "without stable structural localization. The deterministic graph wins on "
                "exactly this dimension by construction."
            )
        elif pred_status == "REJECT" and pred_node == gold_node:
            console.print(
                f"[yellow]STABLE BINARY+LOCALIZATION ACROSS {stats['n_runs']} RUNS[/yellow] — "
                "LLM-with-approver returned correct verdict and correct family in every rerun. "
                "Consistent with literal field-reading on this record. The architectural claim "
                "lives on harder seeds where the cue requires structural reasoning, not "
                "single-record stability on explicit cues."
            )
        else:
            console.print(
                f"[yellow]STABLE BUT INCORRECT[/yellow] — {stats['n_runs']} reruns agree on "
                f"{pred_status}/{pred_node} when gold is {gold_label}/{gold_node}."
            )

    elif is_type_d and is_graph:
        if not stats["verdict_diverged"] and pred_status == "REJECT" and pred_node == gold_node:
            console.print(
                f"[green]DETERMINISTIC AS EXPECTED[/green] — {stats['n_runs']} reruns produced "
                "identical verdicts and primary nodes (variance is zero by construction)."
            )

    elif not is_type_d:
        if pred_status == gold_label:
            console.print(f"[green]MAJORITY MATCHES GOLD[/green] across {stats['n_runs']} runs.")
        else:
            console.print(f"[red]MAJORITY DIVERGES FROM GOLD[/red] across {stats['n_runs']} runs.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a verifier configuration against a single record.")
    parser.add_argument("record_path", type=Path, help="Path to record JSON")
    parser.add_argument("--config", required=True, help="Config name (e.g., gpt_no_approver)")
    parser.add_argument("--model", default=None,
                        help="Optional model override (e.g., gpt-5.4, gpt-5.4-mini, gpt-5.5). "
                             "Falls back to OPENAI_MODEL env var, then to config default.")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of reruns at T=0.1 (default 1 for smoke; 5 for variance per "
                             "protocol critique). Deterministic configs ignore this for compute "
                             "but still emit N identical entries for uniform downstream analysis.")
    args = parser.parse_args()

    if not args.record_path.exists():
        console.print(f"[red]Record not found:[/red] {args.record_path}")
        return 1
    if args.n_runs < 1:
        console.print("[red]--n-runs must be >= 1[/red]")
        return 1

    record, meta = load_record(args.record_path)
    config = load_config(args.config, model=args.model)
    model_used = getattr(config, "model", "(unknown)")

    console.print(Panel.fit(
        f"[bold]config:[/bold] {config.name}\n"
        f"[bold]model:[/bold]  {model_used}\n"
        f"[bold]record:[/bold] {args.record_path}\n"
        f"[bold]n_runs:[/bold] {args.n_runs}",
        border_style="cyan",
    ))

    verdicts: list[Verdict] = []
    for i in range(args.n_runs):
        if args.n_runs > 1:
            console.print(f"[dim]run {i + 1}/{args.n_runs}…[/dim]", end=" ")
        v = config.evaluate(record)
        verdicts.append(v)
        if args.n_runs > 1:
            console.print(f"{v.status} ({v.primary_failure_node or 'no-node'})  [dim]{v.latency_ms} ms[/dim]")

    if args.n_runs == 1:
        render_verdict(verdicts[0], record, meta, config_name=config.name)
    else:
        render_aggregate(verdicts, record, meta, config_name=config.name)

    # Persist for later metric work. Filename includes model so cross-model
    # comparisons are trivially organized; n_runs surfaces in the JSON content.
    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_id = record.get("sample_id", "unknown")
    safe_model = model_used.replace("/", "_").replace(":", "_")
    suffix = f"__n{args.n_runs}" if args.n_runs > 1 else ""
    out_path = out_dir / f"smoke__{config.name}__{safe_model}__{sample_id}{suffix}.json"
    out_path.write_text(json.dumps({
        "config": config.name,
        "model": model_used,
        "n_runs": args.n_runs,
        "record_path": str(args.record_path),
        "sample_id": sample_id,
        "gold_label": record.get("label"),
        "gold_failure_node": (
            record.get("reason", {}).get("failed_validation_node", {}).get("node_id")
            if isinstance(record.get("reason"), dict) else None
        ),
        "violation_family": meta.get("violation_family"),
        "verdicts": [v.model_dump() for v in verdicts],
        "aggregate": aggregate_stats(verdicts) if args.n_runs > 1 else None,
    }, indent=2))
    console.print(f"\n[dim]Persisted result to:[/dim] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
