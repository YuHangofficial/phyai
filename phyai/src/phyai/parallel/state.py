"""Mode + ContextVar + default-mesh hook.

A ContextVar set on the host before ``with torch.cuda.graph(g):`` is
observable inside the capture block, and graph replay does not re-enter
Python — so :func:`graph_capture` toggles dispatcher behaviour during
capture without affecting the recorded graph.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phyai.parallel.mesh import Mesh


class Mode(Enum):
    EAGER = "eager"
    GRAPH_CAPTURING = "graph_capturing"


_mode: ContextVar[Mode] = ContextVar("phyai_dist_mode", default=Mode.EAGER)


def current_mode() -> Mode:
    return _mode.get()


@contextmanager
def graph_capture():
    """Mark the current scope as cuda-graph capture for dispatcher decisions.

    Usage:
        with phyai.parallel.graph_capture(), torch.cuda.graph(g):
            model(real_input)
    """
    tok = _mode.set(Mode.GRAPH_CAPTURING)
    try:
        yield
    finally:
        _mode.reset(tok)


# Single default mesh registered by name; ``use_mesh`` rebinds the default
# mesh name within a scope (a no-op when only one mesh is registered).
_meshes: dict[str, "Mesh"] = {}
_default_mesh_name: ContextVar[str] = ContextVar("phyai_default_mesh", default="model")


def register_mesh(mesh: "Mesh") -> None:
    _meshes[mesh.name] = mesh


def clear_meshes() -> None:
    """Forget every registered mesh (used by ``phyai.parallel.shutdown``)."""
    _meshes.clear()


def registered_meshes() -> tuple["Mesh", ...]:
    return tuple(_meshes.values())


def resolve_mesh(arg: "str | Mesh") -> "Mesh":
    """Resolve a ``Mesh`` instance or registered name to a ``Mesh`` object.

    The literal name ``"model"`` resolves to whatever mesh is currently
    bound by :func:`use_mesh` (defaulting to the registered ``"model"``
    mesh).
    """
    from phyai.parallel.mesh import Mesh as MeshType

    if isinstance(arg, MeshType):
        return arg
    name = arg if arg != "model" else _default_mesh_name.get()
    if name not in _meshes:
        raise KeyError(
            f"Unknown mesh '{name}'. Known meshes: {list(_meshes)}. "
            f"Did you call phyai.parallel.init(...)?"
        )
    return _meshes[name]


def default_mesh() -> "Mesh":
    """Quick-access for the current default mesh."""
    return resolve_mesh("model")


@contextmanager
def use_mesh(name: str):
    """Temporarily switch the default mesh.

    Useful for speculative decoding (target + draft model with separate meshes).
    """
    tok = _default_mesh_name.set(name)
    try:
        yield
    finally:
        _default_mesh_name.reset(tok)
