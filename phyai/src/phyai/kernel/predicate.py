"""Define composable predicates over typed kernel facts."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, Callable, Iterable, Sequence
from dataclasses import dataclass

from phyai.kernel.facts import (
    MISSING,
    Fact,
    Facts,
    FactKind,
    no_truth_value,
)
from phyai.kernel.types import Arch

# Numeric comparator syntax accepted by ordered YAML facts.
_COMPARATOR_RE = re.compile(r"^(<=|>=|==|!=|<|>)?\s*(-?\d+)$")
# Divisibility syntax accepted by ordered YAML facts.
_MODULO_RE = re.compile(r"^%\s*(\d+)\s*(?:==\s*(\d+))?$")


@dataclass(frozen=True)
class Failure:
    """Store a predicate failure and its observed detail."""

    predicate: str
    detail: str

    def __str__(self) -> str:
        return f"{self.predicate} failed: {self.detail}"


@dataclass(frozen=True)
class Skipped:
    """Store a predicate skipped because an optional fact was absent."""

    predicate: str


class Predicate(ABC):
    """A composable, renderable, partially-evaluable boolean expression."""

    __bool__ = no_truth_value("a predicate")

    @abstractmethod
    def eval(self, facts: Facts) -> Failure | None:
        """Return ``None`` when satisfied, else why it was not."""

    @abstractmethod
    def render(self) -> str:
        """Return canonical text for selection traces and plan manifests."""

    @abstractmethod
    def facts_used(self) -> frozenset[str]:
        """Return every fact path read by this expression."""

    @abstractmethod
    def restrict(self, known: Mapping[str, object]) -> "Predicate":
        """Substitute known facts and fold constant branches."""

    def skipped(self, facts: Facts) -> tuple[Skipped, ...]:
        """Return leaves skipped because an optional fact was absent."""

        return ()

    def __and__(self, other: "Predicate") -> "Predicate":
        return And((self, other))

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.render()


class Const(Predicate):
    """Represent a predicate branch folded to a boolean constant."""

    __slots__ = ("value",)

    def __init__(self, value: bool) -> None:
        self.value = bool(value)

    def eval(self, facts: Facts) -> Failure | None:
        return None if self.value else Failure(self.render(), "always false")

    def render(self) -> str:
        return "true" if self.value else "false"

    def facts_used(self) -> frozenset[str]:
        return frozenset()

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        return self


TRUE = Const(True)
FALSE = Const(False)


def is_false(predicate: Predicate) -> bool:
    """Return whether a predicate is a false constant."""

    return isinstance(predicate, Const) and not predicate.value


class Leaf(Predicate):
    """Implement shared evaluation for single-fact predicates."""

    __slots__ = ("fact",)

    def __init__(self, fact: Fact) -> None:
        self.fact = fact

    @abstractmethod
    def test(self, value: object) -> bool:
        """Evaluate a normalized, non-``MISSING`` value."""

    def describe_actual(self, value: object) -> str:
        if value is None:
            return f"{self.fact.path} is unknown"
        return f"got {value!r}"

    def eval(self, facts: Facts) -> Failure | None:
        raw = facts.lookup(self.fact.path)
        if raw is MISSING:
            if facts.is_optional(self.fact.path):
                return None
            return Failure(self.render(), f"{self.fact.path} was not provided")
        value = self.fact.kind.normalize(raw)
        if self.test(value):
            return None
        return Failure(self.render(), self.describe_actual(value))

    def skipped(self, facts: Facts) -> tuple[Skipped, ...]:
        if facts.lookup(self.fact.path) is MISSING and facts.is_optional(
            self.fact.path
        ):
            return (Skipped(self.render()),)
        return ()

    def facts_used(self) -> frozenset[str]:
        return frozenset({self.fact.path})

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        if self.fact.path not in known:
            return self
        value = self.fact.kind.normalize(known[self.fact.path])
        return Const(self.test(value))


class Cmp(Leaf):
    """``fact <op> literal`` or ``fact <op> other_fact``."""

    __slots__ = ("op", "other", "other_fact")

    _OPS: Mapping[str, Callable[[Any, Any], bool]] = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
    }

    def __init__(self, fact: Fact, op: str, other: object) -> None:
        super().__init__(fact)
        if op not in self._OPS:
            raise ValueError(f"unknown comparison operator {op!r}")
        if op in {"<", "<=", ">", ">="} and not fact.kind.ordered:
            raise TypeError(
                f"{fact.path} is {fact.kind.value}, which has no ordering; "
                f"use == / in_ instead of {op}"
            )
        self.op = op
        if isinstance(other, Fact):
            self.other_fact: Fact | None = other
            self.other: object = None
        else:
            self.other_fact = None
            self.other = fact.kind.normalize(other)

    def test(self, value: object) -> bool:
        # Fact-to-fact comparisons use their own two-value evaluation path.
        if value is None:
            return self.op == "!=" and self.other is not None
        try:
            return self._OPS[self.op](value, self.other)
        except TypeError:
            return False

    def render(self) -> str:
        # Fact objects cannot be coerced to bool.
        if self.other_fact is not None:
            return f"{self.fact.path} {self.op} {self.other_fact.path}"
        return f"{self.fact.path} {self.op} {_literal(self.other)}"

    def facts_used(self) -> frozenset[str]:
        if self.other_fact is None:
            return frozenset({self.fact.path})
        return frozenset({self.fact.path, self.other_fact.path})

    def eval(self, facts: Facts) -> Failure | None:
        if self.other_fact is None:
            return super().eval(facts)
        left_raw = facts.lookup(self.fact.path)
        right_raw = facts.lookup(self.other_fact.path)
        for fact, raw in ((self.fact, left_raw), (self.other_fact, right_raw)):
            if raw is MISSING:
                if facts.is_optional(fact.path):
                    return None
                return Failure(self.render(), f"{fact.path} was not provided")
        left = self.fact.kind.normalize(left_raw)
        right = self.other_fact.kind.normalize(right_raw)
        if left is None or right is None:
            unknown = self.fact if left is None else self.other_fact
            return Failure(self.render(), f"{unknown.path} is unknown")
        try:
            if self._OPS[self.op](left, right):
                return None
        except TypeError:
            pass
        return Failure(self.render(), f"got {left!r} vs {right!r}")

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        if self.other_fact is None:
            return super().restrict(known)
        if self.fact.path not in known or self.other_fact.path not in known:
            return self
        left = self.fact.kind.normalize(known[self.fact.path])
        right = self.other_fact.kind.normalize(known[self.other_fact.path])
        if left is None or right is None:
            return FALSE
        try:
            return Const(bool(self._OPS[self.op](left, right)))
        except TypeError:
            return FALSE


class In(Leaf):
    """Match a fact against a set of values."""

    __slots__ = ("values",)

    def __init__(self, fact: Fact, values: Iterable[object]) -> None:
        super().__init__(fact)
        if isinstance(values, (str, bytes)):
            values = (values,)
        self.values = frozenset(fact.kind.normalize(item) for item in values)

    def test(self, value: object) -> bool:
        return value in self.values

    def render(self) -> str:
        return f"{self.fact.path} in {{{_render_set(self.values)}}}"


class ArchAtLeast(Leaf):
    """Match architectures at or above a bound within the same series."""

    __slots__ = ("bound",)

    def __init__(self, fact: Fact, bound: object) -> None:
        super().__init__(fact)
        if fact.kind is not FactKind.ARCH:
            raise TypeError(
                f"{fact.path} is {fact.kind.value}; architecture ordering needs "
                f"an arch fact"
            )
        self.bound = Arch.parse_bound(bound)

    def test(self, value: object) -> bool:
        if value is None:
            return False
        return Arch.parse(value).at_least(self.bound)

    def describe_actual(self, value: object) -> str:
        if value is None:
            return super().describe_actual(value)
        arch = Arch.parse(value)
        if arch.series != self.bound.series:
            return (
                f"got {arch.name!r}, which is not "
                f"{'an' if self.bound.series[0] in 'aeiou' else 'a'} "
                f"{self.bound.series!r} architecture"
            )
        return f"got {arch.name!r}"

    def render(self) -> str:
        return f"{self.fact.path} >= {self.bound.name}"


class ArchFamilyIn(Leaf):
    """Match an architecture against discrete generation families."""

    __slots__ = ("families",)

    def __init__(self, fact: Fact, families: Iterable[object]) -> None:
        super().__init__(fact)
        if fact.kind is not FactKind.ARCH:
            raise TypeError(
                f"{fact.path} is {fact.kind.value}; architecture generations need "
                f"an arch fact"
            )
        if isinstance(families, (str, bytes)):
            families = (families,)
        self.families = frozenset(Arch.parse_family(item) for item in families)
        if not self.families:
            raise ValueError(f"{fact.path} family_in needs at least one generation")

    def test(self, value: object) -> bool:
        if value is None:
            return False
        return Arch.parse(value).family in self.families

    def describe_actual(self, value: object) -> str:
        if value is None:
            return super().describe_actual(value)
        arch = Arch.parse(value)
        if arch.family is None:
            return f"got {arch.name!r}, which has no generation"
        return f"got {arch.name!r} (generation {arch.family})"

    def render(self) -> str:
        # Sort families by series and numeric major version.
        ordered = sorted(self.families, key=_family_sort_key)
        return f"{self.fact.path} family in {{{', '.join(ordered)}}}"


def _family_sort_key(family: str) -> tuple[str, int]:
    match = _ARCH_FAMILY_RE.match(family)
    if match is None:  # pragma: no cover - parse_family already rejected these
        return (family, 0)
    return (match.group(1), int(match.group(2)))


_ARCH_FAMILY_RE = re.compile(r"^([a-z]+)(\d+)$")


class Mod(Leaf):
    """Match a fact by modulus and remainder."""

    __slots__ = ("modulus", "remainder")

    def __init__(self, fact: Fact, modulus: int, remainder: int = 0) -> None:
        super().__init__(fact)
        if not fact.kind.ordered:
            raise TypeError(
                f"{fact.path} is {fact.kind.value}; divisibility needs an int fact"
            )
        self.modulus = int(modulus)
        self.remainder = int(remainder)

    def test(self, value: object) -> bool:
        if value is None:
            return False
        return int(value) % self.modulus == self.remainder

    def render(self) -> str:
        return f"{self.fact.path} % {self.modulus} == {self.remainder}"


class IsNone(Leaf):
    """Explicitly require a fact to be known-absent (``quant: null``)."""

    __slots__ = ()

    def test(self, value: object) -> bool:
        return value is None

    def describe_actual(self, value: object) -> str:
        return f"got {value!r}"

    def render(self) -> str:
        return f"{self.fact.path} is none"


class IsSet(Leaf):
    """Require a fact to be present with a non-``None`` value."""

    __slots__ = ()

    def test(self, value: object) -> bool:
        return value is not None

    def describe_actual(self, value: object) -> str:
        return f"{self.fact.path} is unknown"

    def render(self) -> str:
        return f"{self.fact.path} is set"


class IsTruthy(Leaf):
    """A boolean fact used directly as a predicate."""

    __slots__ = ()

    def test(self, value: object) -> bool:
        return bool(value)

    def render(self) -> str:
        return self.fact.path


class SetHas(Leaf):
    """``value in fact`` for set-valued facts."""

    __slots__ = ("value",)

    def __init__(self, fact: Fact, value: object) -> None:
        super().__init__(fact)
        self.value = str(value).lower()

    def test(self, value: object) -> bool:
        return bool(value) and self.value in value  # type: ignore[operator]

    def render(self) -> str:
        return f"{self.value!r} in {self.fact.path}"


class SetIntersects(Leaf):
    """Require at least one listed member in a set-valued fact."""

    __slots__ = ("values",)

    def __init__(self, fact: Fact, values: Iterable[object]) -> None:
        super().__init__(fact)
        if isinstance(values, str):
            values = (values,)
        self.values = frozenset(str(item).lower() for item in values)

    def test(self, value: object) -> bool:
        return bool(value) and bool(self.values & value)  # type: ignore[operator]

    def render(self) -> str:
        inner = ", ".join(sorted(repr(item) for item in self.values))
        return f"{self.fact.path} intersects {{{inner}}}"


class And(Predicate):
    """All children must hold. Reports the first failure, in written order."""

    __slots__ = ("children",)

    def __init__(self, children: Sequence[Predicate]) -> None:
        flat: list[Predicate] = []
        for child in children:
            # Flatten nested conjunctions into one ordered sequence.
            if isinstance(child, And):
                flat.extend(child.children)
            else:
                flat.append(child)
        self.children = tuple(flat)

    def eval(self, facts: Facts) -> Failure | None:
        for child in self.children:
            failure = child.eval(facts)
            if failure is not None:
                return failure
        return None

    def skipped(self, facts: Facts) -> tuple[Skipped, ...]:
        out: list[Skipped] = []
        for child in self.children:
            out.extend(child.skipped(facts))
        return tuple(out)

    def render(self) -> str:
        return " & ".join(
            f"({child.render()})"
            if isinstance(child, (Or, Implies))
            else child.render()
            for child in self.children
        )

    def facts_used(self) -> frozenset[str]:
        return (
            frozenset().union(*(c.facts_used() for c in self.children)) or frozenset()
        )

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        kept: list[Predicate] = []
        for child in self.children:
            reduced = child.restrict(known)
            if isinstance(reduced, Const):
                if not reduced.value:
                    return FALSE
                continue  # true is the identity for AND
            kept.append(reduced)
        if not kept:
            return TRUE
        return kept[0] if len(kept) == 1 else And(tuple(kept))


class Or(Predicate):
    """At least one child must hold."""

    __slots__ = ("children",)

    def __init__(self, children: Sequence[Predicate]) -> None:
        flat: list[Predicate] = []
        for child in children:
            if isinstance(child, Or):
                flat.extend(child.children)
            else:
                flat.append(child)
        self.children = tuple(flat)

    def eval(self, facts: Facts) -> Failure | None:
        failures: list[Failure] = []
        for child in self.children:
            failure = child.eval(facts)
            if failure is None:
                return None
            failures.append(failure)
        return Failure(
            self.render(),
            "; ".join(f.detail for f in failures) or "no alternative held",
        )

    def render(self) -> str:
        return " | ".join(child.render() for child in self.children)

    def facts_used(self) -> frozenset[str]:
        return (
            frozenset().union(*(c.facts_used() for c in self.children)) or frozenset()
        )

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        kept: list[Predicate] = []
        for child in self.children:
            reduced = child.restrict(known)
            if isinstance(reduced, Const):
                if reduced.value:
                    return TRUE
                continue  # false is the identity for OR
            kept.append(reduced)
        if not kept:
            return FALSE
        return kept[0] if len(kept) == 1 else Or(tuple(kept))


class Not(Predicate):
    __slots__ = ("child",)

    def __init__(self, child: Predicate) -> None:
        self.child = child

    def eval(self, facts: Facts) -> Failure | None:
        if self.child.eval(facts) is None:
            return Failure(self.render(), f"{self.child.render()} held")
        return None

    def render(self) -> str:
        return f"!({self.child.render()})"

    def facts_used(self) -> frozenset[str]:
        return self.child.facts_used()

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        reduced = self.child.restrict(known)
        if isinstance(reduced, Const):
            return Const(not reduced.value)
        return Not(reduced)


class Implies(Predicate):
    """Require a predicate only when its condition holds."""

    __slots__ = ("condition", "requirement")

    def __init__(self, condition: Predicate, requirement: Predicate) -> None:
        self.condition = condition
        self.requirement = requirement

    def eval(self, facts: Facts) -> Failure | None:
        if self.condition.eval(facts) is not None:
            return None
        failure = self.requirement.eval(facts)
        if failure is None:
            return None
        return Failure(self.render(), failure.detail)

    def skipped(self, facts: Facts) -> tuple[Skipped, ...]:
        if self.condition.eval(facts) is not None:
            return ()
        return self.requirement.skipped(facts)

    def render(self) -> str:
        return f"{self.condition.render()} -> {self.requirement.render()}"

    def facts_used(self) -> frozenset[str]:
        return self.condition.facts_used() | self.requirement.facts_used()

    def restrict(self, known: Mapping[str, object]) -> Predicate:
        condition = self.condition.restrict(known)
        requirement = self.requirement.restrict(known)
        if isinstance(condition, Const):
            return TRUE if not condition.value else requirement
        if isinstance(requirement, Const) and requirement.value:
            return TRUE
        return Implies(condition, requirement)


def _as_predicate(value: object) -> Predicate:
    """Accept a bare boolean fact where a predicate is expected."""

    if isinstance(value, Predicate):
        return value
    if isinstance(value, Fact):
        return value.as_predicate()
    raise TypeError(f"expected a predicate or boolean fact, got {value!r}")


def all_of(*parts: object) -> Predicate:
    """Return the conjunction of all supplied predicates."""

    if not parts:
        return TRUE
    return And(tuple(_as_predicate(part) for part in parts))


def any_of(*parts: object) -> Predicate:
    if not parts:
        return FALSE
    return Or(tuple(_as_predicate(part) for part in parts))


def none_of(*parts: object) -> Predicate:
    return Not(any_of(*parts))


def implies(condition: object, requirement: object) -> Predicate:
    return Implies(_as_predicate(condition), _as_predicate(requirement))


def same(*facts: Fact) -> Predicate:
    """Require all supplied facts to have the same value."""

    if len(facts) < 2:
        raise ValueError("same() needs at least two facts")
    first, *rest = facts
    return And(tuple(Cmp(first, "==", other) for other in rest))


def _literal(value: object) -> str:
    """Render a literal the way policy YAML would spell it."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _render_set(values: frozenset[object]) -> str:
    """Render set values with numbers sorted numerically before other values."""

    numbers = sorted(
        item
        for item in values
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    )
    rest = sorted(_literal(item) for item in values if item not in numbers)
    return ", ".join([*(_literal(item) for item in numbers), *rest])


def _is_glob(value: object) -> bool:
    """Return whether a literal contains shell-pattern syntax."""

    return isinstance(value, str) and any(char in value for char in "*?[")


def _reject_pattern(fact: Fact, item: str) -> ValueError:
    """Pattern literals are an error, not a match.

    A pattern names a *shape* of values where a rule should name the values it
    was actually validated on. Generations are short fixed lists anyway --
    every Hopper datacenter part reports sm90 -- so enumeration costs nothing
    and keeps the rule honest about its coverage.
    """

    example = "[sm90, sm100]" if fact.kind is FactKind.ARCH else "[one, other]"
    return ValueError(
        f"{item!r} contains pattern syntax, and pattern matching is not "
        f"supported -- enumerate the exact values instead, e.g. "
        f"'{fact.path}: {example}'"
    )


def _arch_comparator(fact: Fact, text: str) -> Predicate:
    """Parse a same-series architecture lower-bound expression."""

    body = text[2:].strip() if text.startswith(">=") else ""
    if not text.startswith(">=") or not body:
        raise ValueError(
            f"{text!r} is not a supported architecture comparison; only '>=' is, "
            f"as in '>=sm100'"
        )
    # The policy loader adds rule context to invalid-bound errors.
    return ArchAtLeast(fact, body)


def predicate_from_literal(fact: Fact, literal: object) -> Predicate:
    """Compile a typed fact predicate from a YAML literal."""

    if literal is None:
        return IsNone(fact)

    if isinstance(literal, (list, tuple, set, frozenset)):
        if fact.kind is FactKind.STRSET:
            return SetIntersects(fact, literal)
        for item in literal:
            if _is_glob(item):
                raise _reject_pattern(fact, item)
        return In(fact, literal)

    if fact.kind is FactKind.STRSET:
        return SetHas(fact, literal)

    if isinstance(literal, str):
        text = literal.strip()
        if fact.kind is FactKind.ARCH and text.startswith((">", "<", "=", "!")):
            return _arch_comparator(fact, text)
        if fact.kind.ordered:
            modulo = _MODULO_RE.match(text)
            if modulo is not None:
                modulus, remainder = modulo.groups()
                return Mod(fact, int(modulus), int(remainder or 0))
            comparator = _COMPARATOR_RE.match(text)
            if comparator is not None:
                op, number = comparator.groups()
                return Cmp(fact, op or "==", int(number))
        elif _is_glob(text):
            raise _reject_pattern(fact, text)

    if fact.kind is FactKind.ARCH and isinstance(literal, (int, float)):
        raise ValueError(
            f"{literal!r} has no series prefix, so it cannot be matched -- "
            f"write 'sm{literal}' (or 'gfx{literal}')."
        )

    return Cmp(fact, "==", literal)


__all__ = [
    "And",
    "ArchAtLeast",
    "ArchFamilyIn",
    "Cmp",
    "Const",
    "FALSE",
    "Failure",
    "Implies",
    "In",
    "IsNone",
    "IsSet",
    "IsTruthy",
    "Leaf",
    "Mod",
    "Not",
    "Or",
    "Predicate",
    "SetHas",
    "SetIntersects",
    "Skipped",
    "TRUE",
    "all_of",
    "any_of",
    "implies",
    "is_false",
    "none_of",
    "predicate_from_literal",
    "same",
]
