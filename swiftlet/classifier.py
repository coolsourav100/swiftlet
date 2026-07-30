"""
Workload classification — buckets a request into a discrete "shape" so the
learned config store can generalize across similar requests instead of
learning a separate config for every possible prompt length.

This directly addresses the gap BB-CEP flagged: prefill (large batch, many
tokens at once) is compute-bound and wants GPU-heavy placement; decode
(single-token generation) is memory-bound and benefits from a CPU/GPU split.
A single static llama.cpp launch config can't straddle both well in one
session — classifying by shape is what lets swiftlet pick differently.
"""

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    PREFILL_HEAVY = "prefill_heavy"   # long prompt, short expected generation
    DECODE_HEAVY = "decode_heavy"     # short prompt, long expected generation
    BALANCED = "balanced"             # neither dominates


# Bucket boundaries — deliberately coarse. Fine-grained buckets would mean
# more distinct signatures, which means more cold-start exploration before
# the learned store has useful data for any of them. Coarse buckets converge
# faster; that trade-off is worth being explicit about, not just picking
# arbitrary numbers.
PROMPT_LEN_BUCKETS = [(0, 256), (256, 2048), (2048, float("inf"))]
GEN_LEN_BUCKETS = [(0, 128), (128, 1024), (1024, float("inf"))]


def _bucket_index(value: int, buckets: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


@dataclass(frozen=True)
class WorkloadSignature:
    """
    A hashable, coarse-grained description of a request's shape.
    Two requests with the same signature are assumed to benefit from
    the same CPU/GPU configuration.
    """
    prompt_bucket: int
    gen_bucket: int

    def __str__(self) -> str:
        return f"prompt_b{self.prompt_bucket}_gen_b{self.gen_bucket}"

    @property
    def phase(self) -> Phase:
        # Prefill dominates when the prompt is large relative to expected
        # generation; decode dominates in the opposite case. This is a
        # heuristic proxy for "which roofline regime this request sits in"
        # (see BB-CEP Section 3) — it's not a measurement, just a reasonable
        # prior to pick an initial config before real data accumulates.
        if self.prompt_bucket >= 1 and self.gen_bucket == 0:
            return Phase.PREFILL_HEAVY
        if self.prompt_bucket == 0 and self.gen_bucket >= 1:
            return Phase.DECODE_HEAVY
        return Phase.BALANCED


def classify(prompt_tokens: int, expected_gen_tokens: int) -> WorkloadSignature:
    """
    Classify a request into a workload signature.

    expected_gen_tokens: if unknown, use a reasonable default (e.g. 256)
    rather than 0 — a signature of (large_prompt, 0) will incorrectly
    read as pure-prefill when it might really be a long document Q&A
    expecting a substantial answer.
    """
    if prompt_tokens < 0 or expected_gen_tokens < 0:
        raise ValueError("token counts must be non-negative")

    return WorkloadSignature(
        prompt_bucket=_bucket_index(prompt_tokens, PROMPT_LEN_BUCKETS),
        gen_bucket=_bucket_index(expected_gen_tokens, GEN_LEN_BUCKETS),
    )
