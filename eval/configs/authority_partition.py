"""eval/configs/authority_partition.py

VerifierConfig adapter for the deterministic authority-partition layer.

This config has zero LLM cost. Latency is microseconds (predicate dispatch).
Its job is to evaluate the `approver` field group against the 7 authority
predicates and emit a Verdict in the same shape as the LLM-based configs.

Architectural note: this config IGNORES alignment state entirely. It cannot
detect Type A/B/C failures. Composing it with an alignment-validation config
(see `composed.py`) produces coverage across both failure surfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from authority_partition import ...` since src/ is sibling, not on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from authority_partition import evaluate_authority  # noqa: E402

from .base import Verdict


class AuthorityPartitionConfig:
    """Deterministic predicate-based authority partition. Standalone.

    No LLM, no model parameter. Pure structural evaluation of approver state.
    Latency reported in milliseconds (rounded up from microseconds for the
    Verdict shape, so very-fast results show as 0 ms).
    """

    name = "authority_partition"
    model = "deterministic"  # so harness has something to print

    def evaluate(self, record: dict) -> Verdict:
        approver = record.get("approver")
        v = evaluate_authority(approver)
        latency_ms = max(1, v.latency_us // 1000) if v.latency_us > 0 else 0

        if v.status == "PASS":
            return Verdict(
                status="PASS",
                primary_failure_node=None,
                confidence=1.0,
                reasoning=f"All {v.evaluated} authority predicates passed (deterministic dispatch).",
                latency_ms=latency_ms,
                raw_response={"evaluated_predicates": v.evaluated, "subgraph": "authority"},
            )
        # REJECT
        r = v.reason
        assert r is not None
        return Verdict(
            status="REJECT",
            primary_failure_node=str(r.node),
            confidence=1.0,
            reasoning=f"{r.message} (field={r.field}, value={r.value!r})",
            latency_ms=latency_ms,
            raw_response={
                "evaluated_predicates": v.evaluated,
                "subgraph": "authority",
                "field": r.field,
                "value": r.value,
                "node": str(r.node),
            },
        )
