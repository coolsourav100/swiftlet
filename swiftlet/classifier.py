"""
Workload classification — buckets a request into a discrete "shape" so the
learned config store can generalize across similar requests instead of
learning a separate config for every possible prompt length.
"""

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    PREFILL_HEAVY = "prefill_heavy"
    DECODE_HEAVY = "decode_heavy"
    BALANCED = "balanced"


PROMPT_LEN_BUCKETS = [(0, 256), (256, 2048), (2048, float("inf"))]
GEN_LEN_BUCKETS = [(0, 128), (128, 1024), (1024, float("inf"))]


def _bucket_index(value: int, buckets: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


# FIX #6: The signature now exposes its features so the GP can learn
# across signatures.  A single GP with features [gpu, moe, batch,
# prompt_bucket, gen_bucket] sees ALL observations and learns that
# "gpu=99 is fast everywhere" instead of rediscovering it per-signature.

@dataclass(frozen=True)
class WorkloadSignature:
    prompt_bucket: int
    gen_bucket: int

    def __str__(self) -> str:
        return f"prompt_b{self.prompt_bucket}_gen_b{self.gen_bucket}"

    @property
    def phase(self) -> Phase:
        if self.prompt_bucket >= 1 and self.gen_bucket == 0:
            return Phase.PREFILL_HEAVY
        if self.prompt_bucket == 0 and self.gen_bucket >= 1:
            return Phase.DECODE_HEAVY
        return Phase.BALANCED

    def to_features(self) -> list[float]:
        """
        Normalized feature vector for the GP.
        prompt_bucket ∈ {0,1,2} → /2.0, gen_bucket ∈ {0,1,2} → /2.0.
        This lets the RBF kernel correlate similar signatures:
        (prompt=2, gen=0) and (prompt=1, gen=0) are "nearby" in
        feature space, so the GP transfers knowledge between them.
        """
        max_bucket = max(len(PROMPT_LEN_BUCKETS), len(GEN_LEN_BUCKETS)) - 1
        return [
            self.prompt_bucket / max_bucket,
            self.gen_bucket / max_bucket,
        ]


def classify(prompt_tokens: int, expected_gen_tokens: int) -> WorkloadSignature:
    if prompt_tokens < 0 or expected_gen_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return WorkloadSignature(
        prompt_bucket=_bucket_index(prompt_tokens, PROMPT_LEN_BUCKETS),
        gen_bucket=_bucket_index(expected_gen_tokens, GEN_LEN_BUCKETS),
    )
