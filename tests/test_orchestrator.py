import pytest
from swiftlet.config_store import LearnedConfigStore, EngineConfig
from swiftlet.orchestrator import Orchestrator, ServerPool, ServerHandle


def make_fake_launcher(launched: list):
    def launcher(config, port):
        launched.append(config)
        return ServerHandle(config=config, port=port, started_at=0.0)
    return launcher


def test_pool_reuses_existing_server_for_same_config(tmp_path):
    launched = []
    pool = ServerPool(max_size=2, launcher=make_fake_launcher(launched))
    config = EngineConfig(99, 8)

    h1 = pool.get_or_launch(config)
    h2 = pool.get_or_launch(config)

    assert h1 is h2
    assert len(launched) == 1  # only launched once, reused second time


def test_pool_evicts_lru_when_full(tmp_path):
    launched = []
    pool = ServerPool(max_size=1, launcher=make_fake_launcher(launched))

    pool.get_or_launch(EngineConfig(99, 0))
    pool.get_or_launch(EngineConfig(99, 8))  # should evict the first

    assert len(launched) == 2
    assert len(pool.active_configs()) == 1
    assert pool.active_configs()[0].n_cpu_moe == 8


def test_orchestrator_records_and_reroutes_over_time(tmp_path):
    launched = []
    store = LearnedConfigStore(tmp_path / "store.json", epsilon=0.0, seed=1)
    pool = ServerPool(max_size=5, launcher=make_fake_launcher(launched))
    orch = Orchestrator(store, pool)

    from swiftlet.config_store import _default_configs_for_phase
    from swiftlet.classifier import Phase
    candidates = _default_configs_for_phase(Phase.DECODE_HEAVY)

    # Simulate trying every candidate for a decode-heavy workload
    for _ in range(len(candidates)):
        sig, config, handle, exploring = orch.route(prompt_tokens=50, expected_gen_tokens=2000)
        assert exploring is True  # still exploring until all candidates tried
        measured = 50.0 if config.n_cpu_moe == 8 else 10.0
        orch.record(sig, config, measured)

    # Now it should have tried everything and exploit the best (n_cpu_moe=8)
    sig, config, handle, exploring = orch.route(prompt_tokens=55, expected_gen_tokens=2100)
    assert exploring is False
    assert config.n_cpu_moe == 8


def test_unimplemented_launcher_raises_clear_error():
    pool = ServerPool(max_size=1)  # no launcher provided
    with pytest.raises(NotImplementedError, match="No real llama-server launcher"):
        pool.get_or_launch(EngineConfig(99, 0))
