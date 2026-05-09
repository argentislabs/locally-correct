"""eval/configs/composed.py

The full composed configuration: alignment subgraph + authority subgraph,
composing AT THE VERDICT LAYER ONLY. The two subgraphs never interleave at
the node layer.

Composition rule:
  - If either subgraph REJECTs, composed verdict is REJECT.
  - If both PASS, composed verdict is PASS.
  - Primary failure node is the FIRST failing subgraph's primary, in this
    architectural order:
        1. authority subgraph  (deterministic, microsecond)
        2. alignment subgraph  (LLM, seconds)

The architectural order (authority before alignment) reflects upstream-ness:
authority failures invalidate the request itself; alignment failures concern
how the request was operationalized. Putting authority first means a
captured-approver record is rejected before any LLM call is made. This is
also the cheapest evaluation path in production — most authority-failed
records short-circuit before the alignment call.

This config is the load-bearing demonstration of the article's central claim:
adding the authority-partition layer to alignment validation produces coverage
across both alignment and authorization failure surfaces that neither layer
alone achieves.
"""
from __future__ import annotations

import time

from .authority_partition import AuthorityPartitionConfig
from .base import Verdict
from .gpt_no_approver import GPTNoApproverConfig


class ComposedConfig:
    """Authority-partition + GPT alignment validation, composed at verdict layer.

    The alignment subgraph is gpt_no_approver INTENTIONALLY — composition with
    a non-approver-aware alignment system is what demonstrates that authority
    context is not derivable from transaction artifact + intent alone. Using
    gpt_with_approver in the composition would conflate the two subgraphs and
    obscure the categorical-state distinction.
    """

    name = "composed"

    def __init__(self, alignment_model: str | None = None) -> None:
        self.alignment = GPTNoApproverConfig(model=alignment_model)
        self.authority = AuthorityPartitionConfig()
        self.model = self.alignment.model  # surfaced for logging

    def evaluate(self, record: dict) -> Verdict:
        t0 = time.perf_counter()

        # Authority subgraph first (cheap, deterministic, fast-fail).
        auth_verdict = self.authority.evaluate(record)

        # Alignment subgraph runs unconditionally so we capture full latency
        # for cross-config comparison and so the composed result includes
        # alignment evidence even when authority short-circuits the verdict.
        align_verdict = self.alignment.evaluate(record)

        total_latency_ms = int((time.perf_counter() - t0) * 1000)

        # Composition rule
        if auth_verdict.status == "REJECT":
            primary = auth_verdict.primary_failure_node
            reasoning = (
                f"AUTHORITY subgraph REJECTED → {primary}. "
                f"Alignment subgraph independently returned {align_verdict.status}."
            )
            return Verdict(
                status="REJECT",
                primary_failure_node=primary,
                confidence=auth_verdict.confidence,
                reasoning=reasoning,
                latency_ms=total_latency_ms,
                raw_response={
                    "authority_subgraph": auth_verdict.model_dump(),
                    "alignment_subgraph": align_verdict.model_dump(),
                    "decision_source": "authority",
                },
            )

        if align_verdict.status == "REJECT":
            primary = align_verdict.primary_failure_node
            reasoning = (
                f"ALIGNMENT subgraph REJECTED → {primary}. "
                f"Authority subgraph PASSED."
            )
            return Verdict(
                status="REJECT",
                primary_failure_node=primary,
                confidence=align_verdict.confidence,
                reasoning=reasoning,
                latency_ms=total_latency_ms,
                raw_response={
                    "authority_subgraph": auth_verdict.model_dump(),
                    "alignment_subgraph": align_verdict.model_dump(),
                    "decision_source": "alignment",
                },
            )

        # Both PASS
        return Verdict(
            status="PASS",
            primary_failure_node=None,
            confidence=min(
                auth_verdict.confidence or 1.0,
                align_verdict.confidence or 1.0,
            ),
            reasoning="Both subgraphs PASSED. No alignment or authority failure detected.",
            latency_ms=total_latency_ms,
            raw_response={
                "authority_subgraph": auth_verdict.model_dump(),
                "alignment_subgraph": align_verdict.model_dump(),
                "decision_source": "both_pass",
            },
        )
