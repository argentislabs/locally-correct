"""src/authority_partition/verdict.py

Runner that composes predicate outputs into a verdict.

Iterates DISPATCH in canonical order, returns the first Reason found, else None.
This is the architectural entry point used by both the standalone authority-
partition VerifierConfig and the composed config.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .nodes import DISPATCH, AuthorityNode, Reason


@dataclass(frozen=True)
class AuthorityVerdict:
    """Internal verdict shape from the authority subgraph.

    `reason` is None iff status == "PASS".
    `evaluated` is the count of predicates evaluated before either a hit or
    full pass — useful for latency and audit logs.
    """

    status: str  # "PASS" | "REJECT"
    reason: Optional[Reason]
    evaluated: int
    latency_us: int


def evaluate_authority(approver: dict) -> AuthorityVerdict:
    """Run all predicates in canonical order; return first failure or PASS.

    Pure deterministic dispatch. No fuzzy matching anywhere — every predicate
    reads typed structural state and either returns None or a Reason.
    """
    if approver is None or not isinstance(approver, dict):
        # Treat absent approver as a structural failure: nothing to authorize against.
        return AuthorityVerdict(
            status="REJECT",
            reason=Reason(
                node=AuthorityNode.CAPTURED_APPROVER,
                field="approver",
                value=None,
                message="No approver field group present; nothing to authorize against.",
            ),
            evaluated=0,
            latency_us=0,
        )

    t0 = time.perf_counter_ns()
    evaluated = 0
    for node, predicate in DISPATCH.items():
        evaluated += 1
        reason = predicate(approver)
        if reason is not None:
            return AuthorityVerdict(
                status="REJECT",
                reason=reason,
                evaluated=evaluated,
                latency_us=(time.perf_counter_ns() - t0) // 1000,
            )
    return AuthorityVerdict(
        status="PASS",
        reason=None,
        evaluated=evaluated,
        latency_us=(time.perf_counter_ns() - t0) // 1000,
    )
