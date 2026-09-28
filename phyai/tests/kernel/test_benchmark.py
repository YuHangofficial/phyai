"""The default autotune benchmark: env-only wiring, synthesis, and scope.

``profile: autotune`` works from configuration alone: ``initialize_kernel_system``
injects ``default_benchmark`` when the policy asks for tuning and no hook was
given; the hook synthesizes inputs for ops that declare ``bench_args`` (GEMM,
the norms), measures on the query's device, persists winners, and declines the
attention family instead of guessing.
"""

from __future__ import annotations

import json

import pytest

from phyai.kernel.benchmark import default_benchmark
from phyai.kernel.bootstrap import initialize_kernel_system, resolve_policy
from phyai.kernel.config import KernelConfig
from phyai.kernel.facts import Facts
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


def rmsnorm_query() -> KernelQuery:
    return KernelQuery.build(
        "rmsnorm",
        role="bench",
        dtype={"input": "fp32", "weight": "fp32"},
        shape={"tokens": 8, "hidden": 64},
        attrs={"variant": "rms"},
    )


def gemm_query() -> KernelQuery:
    return KernelQuery.build(
        "gemm",
        role="bench",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "bf16"},
        shape={"M": 8, "N": 64, "K": 64},
    )


def autotune_selector(tmp_path) -> Selector:
    catalog = build_catalog()
    config = KernelConfig(
        profile="autotune", autotune_cache=str(tmp_path / "tune.json")
    )
    return Selector(
        catalog,
        resolve_policy(config, catalog),
        device="cpu",
        benchmark=default_benchmark(catalog),
        autotune_cache=config.autotune_cache,
    )


def test_env_only_wiring_installs_the_default_hook_for_autotune_only(tmp_path):
    """PHYAI_KERNEL_PROFILE=autotune must tune without any code."""
    tuned = initialize_kernel_system(
        KernelConfig(profile="autotune", autotune_cache=str(tmp_path / "t.json")),
        device="cpu",
    )
    assert tuned.benchmark is not None
    assert tuned.select(rmsnorm_query()).kernel_id == "torch.rmsnorm"
    assert list(json.loads((tmp_path / "t.json").read_text()).values()) == [
        "torch.rmsnorm"
    ]
    assert initialize_kernel_system(KernelConfig(), device="cpu").benchmark is None


def test_norms_and_gemm_are_measured_cached_and_explained_on_cpu(tmp_path):
    selector = autotune_selector(tmp_path)
    # explain() must not report a cache select() could not have written yet.
    assert not selector.explain(rmsnorm_query()).autotuned
    assert selector.select(rmsnorm_query()).kernel_id == "torch.rmsnorm"
    assert selector.select(gemm_query()).kernel_id == "torch.gemm.bf16"
    cache = json.loads((tmp_path / "tune.json").read_text())
    assert sorted(cache.values()) == ["torch.gemm.bf16", "torch.rmsnorm"]

    trace = selector.explain(rmsnorm_query())
    assert trace.autotuned and trace.selected == "torch.rmsnorm"
    measured = {c.kernel_id: c.benchmark_ms for c in trace.candidates}
    assert measured["torch.rmsnorm"] is not None and measured["torch.rmsnorm"] > 0
    # Without a benchmark hook the same profile never consults the cache.
    catalog = build_catalog()
    bare = Selector(
        catalog, resolve_policy(KernelConfig(profile="autotune"), catalog), device="cpu"
    )
    bare.select(rmsnorm_query())
    assert not bare.explain(rmsnorm_query()).autotuned


def test_the_hook_declines_what_it_cannot_synthesize(tmp_path):
    """No ``bench_args`` (attention) and quantized GEMM (scale layouts cannot be
    faked) raise, so priority order stands and nothing is cached."""
    catalog = build_catalog()
    assert catalog.op("attention_paged").bench_args is None
    selector = autotune_selector(tmp_path)
    query = KernelQuery.build(
        "attention",
        role="bench",
        dtype={"input": "bf16", "key": "bf16", "value": "bf16", "output": "bf16"},
        shape={"tokens": 8, "kv_tokens": 8, "heads": 2, "kv_heads": 2, "head_dim": 64},
        attrs={"layout": "padded", "causal": False},
    )
    selection = selector.select(query)
    with pytest.raises(NotImplementedError, match="bench_args"):
        selector.benchmark(selection.impl, selection.facts, selection)
    assert not (tmp_path / "tune.json").exists()
    facts = Facts(
        values={"quant.format": "fp8_e4m3", "shape.M": 8, "shape.N": 64, "shape.K": 64}
    )
    with pytest.raises(NotImplementedError, match="dense"):
        catalog.op("gemm").bench_args(facts, "cpu")
