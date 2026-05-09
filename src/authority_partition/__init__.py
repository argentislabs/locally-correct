"""src/authority_partition

Deterministic predicate-based authority-partition layer for evaluating
institutional designation state. Composes with LLM-based alignment validation
at the verdict layer to produce coverage across both alignment and authorization
failure surfaces.
"""
from .nodes import DISPATCH, AuthorityNode, Reason
from .verdict import AuthorityVerdict, evaluate_authority

__all__ = [
    "AuthorityNode",
    "AuthorityVerdict",
    "DISPATCH",
    "Reason",
    "evaluate_authority",
]
