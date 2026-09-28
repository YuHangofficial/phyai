"""Select, prepare, and cache kernel implementations."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Callable, Sequence
from pathlib import Path
from dataclasses import field, replace, dataclass

from phyai.kernel.facts import Facts, device_facts, facts_from_query
from phyai.kernel.types import KernelMode, KernelQuery, ModelContext
from phyai.kernel.opspec import Impl, OpSpec, resolve_param_dtypes
from phyai.kernel.policy import Policy, Decision, PolicyError
from phyai.kernel.library import library_facts
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import is_false


class NoKernelError(RuntimeError):
    """Raised when no kernel implementation can satisfy a call."""


@dataclass(frozen=True)
class CandidateTrace:
    """Store one candidate's assessment and preparation result."""

    kernel_id: str
    eligible: bool
    #: Rendered capability predicate.
    when: str = ""
    #: Predicate failure and observed value.
    reason: str = ""
    #: Predicates skipped because optional facts were absent.
    skipped: tuple[str, ...] = ()
    prepared: bool = False
    prepare_error: str | None = None
    benchmark_ms: float | None = None


@dataclass(frozen=True)
class SelectionTrace:
    """Store the trace for one kernel selection."""

    op: str
    role: str
    facts: Mapping[str, object]
    policy_profile: str
    matched_rules: tuple[str, ...]
    candidates: tuple[CandidateTrace, ...]
    selected: str | None = None
    autotuned: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "op": self.op,
            "role": self.role,
            "facts": {key: _plain(value) for key, value in sorted(self.facts.items())},
            "policy_profile": self.policy_profile,
            "matched_rules": list(self.matched_rules),
            "selected": self.selected,
            "autotuned": self.autotuned,
            "candidates": [
                {
                    "id": item.kernel_id,
                    "eligible": item.eligible,
                    "when": item.when,
                    "reason": item.reason,
                    "skipped": list(item.skipped),
                    "prepared": item.prepared,
                    "prepare_error": item.prepare_error,
                    "benchmark_ms": item.benchmark_ms,
                }
                for item in self.candidates
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, default=str)

    def explain(self) -> str:
        """Render the selection trace for logs and errors."""

        lines = [f"op={self.op!r} role={self.role!r} selected={self.selected!r}"]
        if self.matched_rules:
            lines.append(f"  policy rules: {', '.join(self.matched_rules)}")
        for item in self.candidates:
            mark = "ok  " if item.eligible else "no  "
            lines.append(f"  {mark}{item.kernel_id}: {item.when}")
            if not item.eligible:
                lines.append(f"        rejected: {item.reason}")
            if item.prepare_error:
                lines.append(f"        prepare failed: {item.prepare_error}")
            for skipped in item.skipped:
                lines.append(f"        vacuous: {skipped}")
        return "\n".join(lines)


def _plain(value: object) -> object:
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class Selection:
    """Store a prepared implementation and its selection context."""

    impl: Impl
    query: KernelQuery
    facts: Facts
    implementation: Any
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kernel_id(self) -> str:
        return self.impl.kernel_id

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.implementation(*args, **kwargs)


BenchmarkFn = Callable[[Impl, Facts, "Selection"], float]


class Selector:
    """Select kernels from a catalog using an engine-scoped policy."""

    def __init__(
        self,
        catalog: Catalog,
        policy: Policy | None = None,
        *,
        model: ModelContext | None = None,
        device: object | None = None,
        benchmark: BenchmarkFn | None = None,
        autotune_cache: str | Path | None = None,
    ) -> None:
        from phyai.kernel.device import probe_device

        self.catalog = catalog
        self.policy = policy or Policy()
        self.model = model or ModelContext()
        # probe_device accepts live devices and canonical ``vendor:arch``
        # strings, and probes the active device when none is supplied.
        self.device = probe_device(device)
        self.benchmark = benchmark
        # Bumped whenever cached selections may no longer be valid; call
        # sites fold it into their staleness fingerprint.
        self._epoch = 0
        self._cache: dict[tuple[object, ...], Selection] = {}
        self._autotune: dict[str, str] = {}
        # Per-tune-key measurements from this process, for explain().
        self._measurements: dict[str, dict[str, float]] = {}
        self._autotune_path = (
            Path(autotune_cache).expanduser() if autotune_cache else None
        )
        self._load_autotune()

    @property
    def epoch(self) -> int:
        """Increase whenever previously cached selections may be stale."""

        return self._epoch

    def normalize(self, query: KernelQuery) -> KernelQuery:
        updates: dict[str, object] = {}
        if query.model == ModelContext() and self.model != ModelContext():
            updates["model"] = self.model
        # Replace only implicit query devices with the selector profile.
        if self.device is not None and not query.device_explicit:
            updates["device"] = self.device
        return replace(query, **updates) if updates else query

    def facts_for(self, query: KernelQuery, spec: OpSpec) -> Facts:
        # library_facts memoizes per import name, so building the mapping per
        # call is a dict comprehension over one or two cached booleans.
        return facts_from_query(
            query, spec, libraries=library_facts(self.catalog.libraries(spec.name))
        )

    def assess(
        self, spec: OpSpec, facts: Facts, mode: KernelMode
    ) -> tuple[list[Impl], list[CandidateTrace]]:
        """Split the operation's implementations into eligible and rejected."""

        eligible: list[Impl] = []
        traces: list[CandidateTrace] = []
        graph = mode in {KernelMode.CAPTURE, KernelMode.REPLAY}

        for impl in self.catalog.impls(spec.name):
            rendered = impl.when.render()
            # Graph modes require implementations declared capture-safe.
            if graph and not impl.capture_safe:
                traces.append(
                    CandidateTrace(
                        impl.kernel_id,
                        False,
                        rendered,
                        f"not usable inside a CUDA graph ({mode.value})",
                    )
                )
                continue
            failure = impl.when.eval(facts)
            if failure is None:
                eligible.append(impl)
                traces.append(
                    CandidateTrace(
                        impl.kernel_id,
                        True,
                        rendered,
                        "eligible",
                        tuple(item.predicate for item in impl.when.skipped(facts)),
                    )
                )
            else:
                traces.append(
                    CandidateTrace(impl.kernel_id, False, rendered, str(failure))
                )
        return eligible, traces

    def order(
        self,
        eligible: Sequence[Impl],
        decision: Decision,
        prefer: Sequence[str] = (),
    ) -> list[Impl]:
        """Order eligible implementations by policy, preference, and fallback."""

        by_id = {impl.kernel_id: impl for impl in eligible}

        if not decision.candidates:
            if not prefer:
                return list(eligible)  # already priority-ordered by the catalog
            hinted = [by_id[k] for k in prefer if k in by_id]
            named = {impl.kernel_id for impl in hinted}
            return hinted + [i for i in eligible if i.kernel_id not in named]

        ordered = [
            by_id[kernel_id] for kernel_id in decision.candidates if kernel_id in by_id
        ]
        if decision.strict or self.policy.fallback == "error":
            return ordered

        # Policy fallback adds only reference implementations.
        listed = {impl.kernel_id for impl in ordered}
        ordered.extend(
            impl for impl in eligible if impl.reference and impl.kernel_id not in listed
        )
        return ordered

    def explain(
        self,
        query: KernelQuery,
        *,
        prefer: Sequence[str] = (),
    ) -> SelectionTrace:
        query = self.normalize(query)
        spec = self.catalog.op(query.op)
        facts = self.facts_for(query, spec)
        decision = self.policy.decide(facts, self.catalog)
        eligible, traces = self.assess(spec, facts, query.mode)
        ordered = self.order(eligible, decision, prefer)

        selected = ordered[0].kernel_id if ordered else None
        autotuned = False
        if self._tuning(decision, query.mode):
            tune_key = self._tune_key(facts, decision)
            cached = self._autotune.get(tune_key)
            if cached is not None and any(i.kernel_id == cached for i in ordered):
                selected, autotuned = cached, True
            for kernel_id, elapsed in self._measurements.get(tune_key, {}).items():
                _mark(traces, kernel_id, benchmark_ms=elapsed)
        if decision.strict and (
            not decision.candidates or selected != decision.candidates[0]
        ):
            selected = None

        return SelectionTrace(
            op=query.op,
            role=query.role,
            facts=facts.values,
            policy_profile=self.policy.profile,
            matched_rules=decision.matched_rules,
            candidates=tuple(traces),
            selected=selected,
            autotuned=autotuned,
        )

    def select(
        self,
        query: KernelQuery,
        *,
        prefer: Sequence[str] = (),
    ) -> Selection:
        query = self.normalize(query)
        spec = self.catalog.op(query.op)
        facts = self.facts_for(query, spec)
        decision = self.policy.decide(facts, self.catalog)

        key = (
            self.catalog.version,
            self.policy.version,
            self.policy.profile,
            self.policy.fallback,
            query.cache_key,
            decision.matched_rules,
            decision.candidates,
            tuple(sorted((str(k), str(v)) for k, v in decision.params.items())),
            tuple(prefer),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        eligible, traces = self.assess(spec, facts, query.mode)
        ordered = self.order(eligible, decision, prefer)

        if decision.strict:
            wanted = decision.candidates[0] if decision.candidates else None
            if wanted is None:
                raise self._fail(
                    "a strict override named no kernel", query, facts, decision, traces
                )
            if not any(impl.kernel_id == wanted for impl in ordered):
                raise self._fail(
                    f"override requires {wanted!r}, which cannot handle this call",
                    query,
                    facts,
                    decision,
                    traces,
                )
            ordered = [impl for impl in ordered if impl.kernel_id == wanted]

        if not ordered:
            raise self._fail(
                f"no kernel can handle op={query.op!r} role={query.role!r}",
                query,
                facts,
                decision,
                traces,
            )

        selection = self._prepare_first(spec, query, facts, decision, ordered, traces)
        self._cache[key] = selection
        return selection

    def param_dtypes(
        self,
        op: str,
        *,
        activation: str,
        known: Mapping[str, object] | None = None,
        preferred: Mapping[str, str] | None = None,
        prefer: Sequence[str] = (),
    ) -> dict[str, str]:
        """Choose parameter dtypes from facts known at construction time."""

        spec = self.catalog.op(op)
        # Restrict capabilities with all facts known before tensor allocation.
        construction: dict[str, object] = {"op": spec.name}
        construction.update(device_facts(self.device))
        construction.update(library_facts(self.catalog.libraries(spec.name)))
        construction.update(known or {})

        viable = [
            impl
            for impl in self.catalog.impls(spec.name)
            if not is_false(impl.when.restrict(construction))
        ]
        if not viable:
            viable = list(self.catalog.impls(spec.name))
        return resolve_param_dtypes(
            spec,
            viable,
            activation=activation,
            preferred=preferred,
            prefer=prefer,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        # Call sites fold the epoch into their staleness fingerprint.
        self._epoch += 1

    def _fail(
        self,
        message: str,
        query: KernelQuery,
        facts: Facts,
        decision: Decision,
        traces: Sequence[CandidateTrace],
    ) -> NoKernelError:
        trace = SelectionTrace(
            op=query.op,
            role=query.role,
            facts=facts.values,
            policy_profile=self.policy.profile,
            matched_rules=decision.matched_rules,
            candidates=tuple(traces),
        )
        return NoKernelError(f"{message}\n{trace.explain()}")

    def _prepare_first(
        self,
        spec: OpSpec,
        query: KernelQuery,
        facts: Facts,
        decision: Decision,
        ordered: Sequence[Impl],
        traces: list[CandidateTrace],
    ) -> Selection:
        tuning = self._tuning(decision, query.mode)
        prepared: list[Selection] = []
        errors: list[str] = []

        for impl in ordered:
            # Rule params target the kernels the rule names; reference
            # fallbacks appended by order() run without them.
            params = (
                dict(decision.params) if impl.kernel_id in decision.candidates else {}
            )
            try:
                implementation = impl.prepare(facts, params)
            except PolicyError:
                # A rule parameter its target cannot accept is a configuration
                # error; never fall back past it to another kernel.
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors.append(f"{impl.kernel_id}: {message}")
                _mark(traces, impl.kernel_id, prepare_error=message)
                if decision.strict:
                    raise self._fail(
                        f"override requires {impl.kernel_id!r}, which failed to "
                        f"prepare: {message}",
                        query,
                        facts,
                        decision,
                        traces,
                    ) from exc
                continue

            selection = Selection(
                impl=impl,
                query=query,
                facts=facts,
                implementation=implementation,
                params=params,
            )
            _mark(traces, impl.kernel_id, prepared=True)
            if not tuning:
                return selection
            prepared.append(selection)

        if not prepared:
            raise self._fail(
                "every eligible kernel failed to prepare: " + "; ".join(errors),
                query,
                facts,
                decision,
                traces,
            )
        return self._tune(facts, decision, prepared, traces)

    def _tuning(self, decision: Decision, mode: KernelMode) -> bool:
        # The benchmark gate keeps select() and explain() consistent: a cache
        # entry is only honoured when tuning could actually have produced it.
        return (
            not decision.strict
            and self.policy.profile == "autotune"
            and self.benchmark is not None
            and mode is KernelMode.EAGER
        )

    def _tune(
        self,
        facts: Facts,
        decision: Decision,
        prepared: Sequence[Selection],
        traces: list[CandidateTrace],
    ) -> Selection:
        key = self._tune_key(facts, decision)
        cached = self._autotune.get(key)
        if cached is not None:
            for selection in prepared:
                if selection.kernel_id == cached:
                    return selection

        measured: list[tuple[float, str, Selection]] = []
        for selection in prepared:
            started = time.perf_counter()
            try:
                elapsed = float(
                    self.benchmark(selection.impl, selection.facts, selection)
                )
            except Exception:
                continue
            if elapsed < 0:
                elapsed = (time.perf_counter() - started) * 1000.0
            _mark(traces, selection.kernel_id, benchmark_ms=elapsed)
            measured.append((elapsed, selection.kernel_id, selection))
        if not measured:
            return prepared[0]
        self._measurements[key] = {kid: ms for ms, kid, _ in measured}
        _, _, winner = min(measured, key=lambda item: (item[0], item[1]))
        self._autotune[key] = winner.kernel_id
        self._save_autotune()
        return winner

    def _tune_key(self, facts: Facts, decision: Decision) -> str:
        return json.dumps(
            {
                "catalog": self.catalog.version,
                "policy": self.policy.version,
                "profile": self.policy.profile,
                "fallback": self.policy.fallback,
                "facts": {k: _plain(v) for k, v in sorted(facts.values.items())},
                "rules": list(decision.matched_rules),
                "candidates": list(decision.candidates),
                "params": {str(k): str(v) for k, v in sorted(decision.params.items())},
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _load_autotune(self) -> None:
        if self._autotune_path is None or not self._autotune_path.exists():
            return
        try:
            data = json.loads(self._autotune_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, Mapping):
            # Version-prefixed keys leave stale entries unused.
            self._autotune.update({str(k): str(v) for k, v in data.items()})

    def _save_autotune(self) -> None:
        if self._autotune_path is None:
            return
        try:
            self._autotune_path.parent.mkdir(parents=True, exist_ok=True)
            self._autotune_path.write_text(
                json.dumps(self._autotune, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            return


def _mark(traces: list[CandidateTrace], kernel_id: str, **changes: object) -> None:
    for index, item in enumerate(traces):
        if item.kernel_id == kernel_id:
            traces[index] = replace(item, **changes)  # type: ignore[arg-type]
            return


__all__ = [
    "BenchmarkFn",
    "CandidateTrace",
    "NoKernelError",
    "Selection",
    "SelectionTrace",
    "Selector",
]
