"""eval/configs/base.py

Shared types, base interface, and the canonical failure-node taxonomy.

The failure-node taxonomy is load-bearing: configs that produce a
`primary_failure_node` MUST draw from these constants so cross-config
match-score evaluation is meaningful.

ALIGNMENT_FAILURE_NODES mirrors Yao et al. (NDSS 2026) graph node families.
AUTHORITY_FAILURE_NODES is our extension — the 7 Type D failure modes.
ALL_FAILURE_NODES is the union, used by full-context configs.

A VerifierConfig takes a `record` dict (the schema-locked record from yesterday's
work) and produces a Verdict. All configs return the same Verdict shape so that
metrics and composition logic can operate uniformly.

The Record dict has shape:
{
  "sample_id":    str,
  "human_intent": str,
  "proposed_tx":  { protocol, operation, asset, amount, transactions, gas, ... },
  "metadata":     { block_time, tx_hash, actor_type, ... },
  "label":        "PASS" | "REJECT",                 # gold truth
  "reason":       { simple_reason, failed_validation_node },
  "approver":     { ... }                            # the new field group
}

Configs may choose to ignore certain fields (e.g., gpt_no_approver ignores
record["approver"]) — that's exactly what we're measuring.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


VerdictStatus = Literal["PASS", "REJECT", "ERROR"]


# --- Canonical failure-node taxonomy (load-bearing for match-score) ---

ALIGNMENT_FAILURE_NODES: list[str] = [
    "intent_alignment.operation_matching",
    "intent_alignment.token_matching",
    "intent_alignment.amount_matching",
    "intent_alignment.parameter_matching",
    "technical_invariants.format_validation",
    "technical_invariants.protocol_compatibility",
    "legal_compliance.sanctions_screening",
    "adversarial_detection.intent_manipulation",
    "adversarial_detection.parameter_manipulation",
    "adversarial_detection.mev_risky_parameters",
]

AUTHORITY_FAILURE_NODES: list[str] = [
    "authority_partition.captured_approver",
    "authority_partition.collusion_proposer_approver",
    "authority_partition.undisclosed_conflict_of_interest",
    "authority_partition.signature_forgery",
    "authority_partition.signature_coercion",
    "authority_partition.separation_of_duties_violation",
    "authority_partition.designation_drift",
]

ALL_FAILURE_NODES: list[str] = ALIGNMENT_FAILURE_NODES + AUTHORITY_FAILURE_NODES


class Verdict(BaseModel):
    """Standardized output for any verifier configuration.

    `primary_failure_node` is the load-bearing localization signal — the
    config's best guess at WHICH check failed. For PASS it's None.
    """

    status: VerdictStatus
    primary_failure_node: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    latency_ms: int = 0
    raw_response: dict | None = None
    error: str | None = None


@runtime_checkable
class VerifierConfig(Protocol):
    """All verifier configurations implement this interface."""

    name: str

    def evaluate(self, record: dict) -> Verdict: ...


class RecordSummary(BaseModel):
    """Light-touch view of a record for logging and debugging."""

    sample_id: str
    label: str
    violation_family: str | None = None
    intent_preview: str = Field(default="", max_length=120)

    @classmethod
    def from_record(cls, record: dict, family: str | None = None) -> "RecordSummary":
        intent = record.get("human_intent", "") or ""
        truncated = intent[:119] + "…" if len(intent) > 119 else intent
        return cls(
            sample_id=record.get("sample_id", "(no id)"),
            label=record.get("label", "(no label)"),
            violation_family=family,
            intent_preview=truncated,
        )
