"""eval/configs/_taxonomy.py

Shared failure-node taxonomy for all verifier configurations.

Two categorically distinct sets:

  * ALIGNMENT_FAILURE_NODES  — failures detectable from the transaction artifact
                                + intent. Mirrors Yao et al. NDSS 2026 graph families.
  * AUTHORITY_FAILURE_NODES  — failures detectable only from institutional
                                designation state (the `approver` field group).

The split is the load-bearing categorical-state distinction. Configs that
ignore the approver field group draw only from ALIGNMENT_FAILURE_NODES; configs
that consume the approver field group may draw from either set.

When `gpt_with_approver` is asked to localize a failure, it has the full
combined vocabulary at its disposal — this is the strongest possible LLM-as-judge
baseline, and is the existential test described in the protocol critique.
"""
from __future__ import annotations


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


COMBINED_FAILURE_NODES: list[str] = ALIGNMENT_FAILURE_NODES + AUTHORITY_FAILURE_NODES


# Convenience: violation_family → expected primary node for gold-label match scoring
FAMILY_TO_NODE: dict[str, str] = {
    "captured":    "authority_partition.captured_approver",
    "colluded":    "authority_partition.collusion_proposer_approver",
    "undisclosed": "authority_partition.undisclosed_conflict_of_interest",
    "forged":      "authority_partition.signature_forgery",
    "coerced":     "authority_partition.signature_coercion",
    "sod":         "authority_partition.separation_of_duties_violation",
    "drift":       "authority_partition.designation_drift",
}
