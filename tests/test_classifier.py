import pytest
from swiftlet.classifier import classify, WorkloadSignature, Phase


def test_short_prompt_short_gen_is_balanced_or_decode():
    sig = classify(prompt_tokens=50, expected_gen_tokens=50)
    assert sig.prompt_bucket == 0
    assert sig.gen_bucket == 0
    assert sig.phase == Phase.BALANCED


def test_long_prompt_short_gen_is_prefill_heavy():
    sig = classify(prompt_tokens=5000, expected_gen_tokens=50)
    assert sig.phase == Phase.PREFILL_HEAVY


def test_short_prompt_long_gen_is_decode_heavy():
    sig = classify(prompt_tokens=50, expected_gen_tokens=2000)
    assert sig.phase == Phase.DECODE_HEAVY


def test_signature_is_hashable_and_stable():
    sig1 = classify(100, 100)
    sig2 = classify(100, 100)
    assert sig1 == sig2
    assert hash(sig1) == hash(sig2)
    assert str(sig1) == str(sig2)


def test_negative_tokens_raises():
    with pytest.raises(ValueError):
        classify(-1, 100)
    with pytest.raises(ValueError):
        classify(100, -1)


def test_bucket_boundaries_are_inclusive_lower_exclusive_upper():
    # 256 should fall into the SECOND bucket, not the first, per boundary definition
    sig_at_boundary = classify(256, 0)
    sig_below_boundary = classify(255, 0)
    assert sig_at_boundary.prompt_bucket == 1
    assert sig_below_boundary.prompt_bucket == 0
