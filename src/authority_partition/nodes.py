"""src/authority_partition/nodes.py

Deterministic predicate-based authority-partition layer. NO fuzzy matching,
NO string heuristics, NO LLM reasoning. Pure enum dispatch over typed
structural designation state.

Architecture:
  - AuthorityNode is a closed enum of the 7 authority-partition failure modes.
  - Each predicate is a pure function: (approver: dict) -> Optional[Reason].
  - The DISPATCH table maps AuthorityNode → predicate, in canonical evaluation
    order. The runner iterates this table; first non-None Reason wins.
  - Predicates dispatch on EXPLICIT typed fields ONLY:
      approver["approval"]["approval_method"]    : enum
      approver["approval"]["signature_valid"]    : bool
      approver["designation_chain"][i]["designation_root"]    : enum | None
      approver["disclosed_interests"][i]["disclosure_status"] : enum
      approver["structural_flags"][<flag>]                    : bool

Rationale: the architectural claim is that authority-partition is deterministic
predicate evaluation, not semantic inference. This module's IMPLEMENTATION
must reflect that: every condition is a typed equality or boolean read.

If a predicate ever needs to inspect a free-form string semantically, that is
a SCHEMA FAILURE — promote the marker to a structural flag instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional


class AuthorityNode(StrEnum):
    """Closed taxonomy of authority-partition failure modes.

    Order matters: this is the canonical evaluation order. When multiple
    predicates would fire on the same record, the first one in this order
    is selected as the primary failure node. Order reflects upstream-ness:
    forgery and coercion (approval-level) before designation chain failures
    before composite organizational failures.
    """

    SIGNATURE_FORGERY = "authority_partition.signature_forgery"
    SIGNATURE_COERCION = "authority_partition.signature_coercion"
    CAPTURED_APPROVER = "authority_partition.captured_approver"
    UNDISCLOSED_CONFLICT_OF_INTEREST = "authority_partition.undisclosed_conflict_of_interest"
    COLLUSION_PROPOSER_APPROVER = "authority_partition.collusion_proposer_approver"
    SEPARATION_OF_DUTIES_VIOLATION = "authority_partition.separation_of_duties_violation"
    DESIGNATION_DRIFT = "authority_partition.designation_drift"


@dataclass(frozen=True)
class Reason:
    """Structured rejection reason emitted by a predicate.

    `evidence` is the typed field-and-value tuple that triggered the predicate.
    Auditors can trace any REJECT to a literal field read in the approver
    structure.
    """

    node: AuthorityNode
    field: str
    value: object
    message: str


# ---------------------------------------------------------------------------
# Predicate functions — each is a pure (approver) -> Optional[Reason] mapping.
# ---------------------------------------------------------------------------


def check_signature_forgery(approver: dict) -> Optional[Reason]:
    approval = approver.get("approval", {})
    if approval.get("approval_method") == "forged":
        return Reason(
            node=AuthorityNode.SIGNATURE_FORGERY,
            field="approval.approval_method",
            value="forged",
            message="Approval method is forged: the signature did not originate from the principal it appears to.",
        )
    if approval.get("signature_valid") is False:
        return Reason(
            node=AuthorityNode.SIGNATURE_FORGERY,
            field="approval.signature_valid",
            value=False,
            message="Cryptographic signature did not validate against the principal's key.",
        )
    return None


def check_signature_coercion(approver: dict) -> Optional[Reason]:
    approval = approver.get("approval", {})
    if approval.get("approval_method") == "coerced":
        return Reason(
            node=AuthorityNode.SIGNATURE_COERCION,
            field="approval.approval_method",
            value="coerced",
            message="Approval was issued under coercion; signature is technically valid but the conditions for the authority were not met.",
        )
    return None


def check_captured_approver(approver: dict) -> Optional[Reason]:
    flags = approver.get("structural_flags", {})
    if flags.get("captured_approver") is True:
        return Reason(
            node=AuthorityNode.CAPTURED_APPROVER,
            field="structural_flags.captured_approver",
            value=True,
            message="Designation chain is formally present but substantively compromised, OR designation_root is null.",
        )
    # Secondary literal check: any chain link with designation_root explicitly null.
    chain = approver.get("designation_chain", []) or []
    for i, link in enumerate(chain):
        if link.get("designation_root") is None:
            return Reason(
                node=AuthorityNode.CAPTURED_APPROVER,
                field=f"designation_chain[{i}].designation_root",
                value=None,
                message="Designation chain link does not terminate at an externally-rooted authority.",
            )
    return None


def check_undisclosed_conflict_of_interest(approver: dict) -> Optional[Reason]:
    flags = approver.get("structural_flags", {})
    if flags.get("undisclosed_conflict_of_interest") is True:
        return Reason(
            node=AuthorityNode.UNDISCLOSED_CONFLICT_OF_INTEREST,
            field="structural_flags.undisclosed_conflict_of_interest",
            value=True,
            message="Authorization was issued under incomplete material disclosure.",
        )
    # Secondary literal check: any disclosed interest with stale status.
    interests = approver.get("disclosed_interests", []) or []
    for i, interest in enumerate(interests):
        if interest.get("disclosure_status") == "stale":
            return Reason(
                node=AuthorityNode.UNDISCLOSED_CONFLICT_OF_INTEREST,
                field=f"disclosed_interests[{i}].disclosure_status",
                value="stale",
                message="A disclosed interest is stale; the authorization context is no longer current.",
            )
    return None


def check_collusion_proposer_approver(approver: dict) -> Optional[Reason]:
    flags = approver.get("structural_flags", {})
    if flags.get("collusion_proposer_approver") is True:
        return Reason(
            node=AuthorityNode.COLLUSION_PROPOSER_APPROVER,
            field="structural_flags.collusion_proposer_approver",
            value=True,
            message="Proposer and approver coordinated against the structural separation the designation requires.",
        )
    return None


def check_separation_of_duties_violation(approver: dict) -> Optional[Reason]:
    flags = approver.get("structural_flags", {})
    if flags.get("separation_of_duties_violation") is True:
        return Reason(
            node=AuthorityNode.SEPARATION_OF_DUTIES_VIOLATION,
            field="structural_flags.separation_of_duties_violation",
            value=True,
            message="The same principal holds incompatible roles whose combination violates structural separation.",
        )
    return None


def check_designation_drift(approver: dict) -> Optional[Reason]:
    flags = approver.get("structural_flags", {})
    if flags.get("designation_drift") is True:
        return Reason(
            node=AuthorityNode.DESIGNATION_DRIFT,
            field="structural_flags.designation_drift",
            value=True,
            message="The cumulative authority structure has drifted from its externally-rooted origin without holistic reauthorization.",
        )
    return None


# ---------------------------------------------------------------------------
# Dispatch table. Iteration order = canonical evaluation order.
# ---------------------------------------------------------------------------

DISPATCH: dict[AuthorityNode, Callable[[dict], Optional[Reason]]] = {
    AuthorityNode.SIGNATURE_FORGERY: check_signature_forgery,
    AuthorityNode.SIGNATURE_COERCION: check_signature_coercion,
    AuthorityNode.CAPTURED_APPROVER: check_captured_approver,
    AuthorityNode.UNDISCLOSED_CONFLICT_OF_INTEREST: check_undisclosed_conflict_of_interest,
    AuthorityNode.COLLUSION_PROPOSER_APPROVER: check_collusion_proposer_approver,
    AuthorityNode.SEPARATION_OF_DUTIES_VIOLATION: check_separation_of_duties_violation,
    AuthorityNode.DESIGNATION_DRIFT: check_designation_drift,
}
