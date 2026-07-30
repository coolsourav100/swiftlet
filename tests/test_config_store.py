import json
import pytest
from swiftlet.classifier import classify
from swiftlet.config_store import LearnedConfigStore, EngineConfig


def test_no_data_forces_exploration(tmp_path):
    store = LearnedConfigStore(tmp_path / "store.json", seed=0)
    sig = classify(50, 2000)  # decode-heavy
    config, is_exploration = store.choose_config(sig)
    assert is_exploration is True
    assert config.n_gpu_layers == 99


def test_recording_persists_across_instances(tmp_path):
    path = tmp_path / "store.json"
    store1 = LearnedConfigStore(path, seed=0)
    sig = classify(50, 2000)
    config = EngineConfig(n_gpu_layers=99, n_cpu_moe=8)
    store1.record_result(sig, config, tok_per_sec=25.0)

    assert path.exists()

    store2 = LearnedConfigStore(path, seed=0)
    best = store2.best_known(sig)
    assert best is not None
    assert best.config.n_cpu_moe == 8
    assert best.mean_tok_per_sec == 25.0


def test_exploitation_picks_best_after_all_candidates_tried(tmp_path):
    store = LearnedConfigStore(tmp_path / "store.json", epsilon=0.0, seed=0)
    sig = classify(50, 2000)  # decode-heavy -> 5 candidate configs

    from swiftlet.config_store import _default_configs_for_phase
    from swiftlet.classifier import Phase
    candidates = _default_configs_for_phase(Phase.DECODE_HEAVY)

    # Try every candidate once, with n_cpu_moe=8 clearly the best performer
    for cfg in candidates:
        measured = 40.0 if cfg.n_cpu_moe == 8 else 20.0
        store.record_result(sig, cfg, measured)

    # With epsilon=0 and all candidates already tried, should exploit the best
    config, is_exploration = store.choose_config(sig)
    assert is_exploration is False
    assert config.n_cpu_moe == 8


def test_mean_tok_per_sec_averages_multiple_trials(tmp_path):
    store = LearnedConfigStore(tmp_path / "store.json", seed=0)
    sig = classify(50, 2000)
    config = EngineConfig(n_gpu_layers=99, n_cpu_moe=8)

    store.record_result(sig, config, 20.0)
    store.record_result(sig, config, 30.0)

    best = store.best_known(sig)
    assert best.trials == 2
    assert best.mean_tok_per_sec == 25.0


def test_different_signatures_are_independent(tmp_path):
    store = LearnedConfigStore(tmp_path / "store.json", seed=0)
    sig_decode = classify(50, 2000)
    sig_prefill = classify(5000, 50)

    store.record_result(sig_decode, EngineConfig(99, 8), 40.0)
    store.record_result(sig_prefill, EngineConfig(99, 0), 15.0)

    assert store.best_known(sig_decode).config.n_cpu_moe == 8
    assert store.best_known(sig_prefill).config.n_cpu_moe == 0
